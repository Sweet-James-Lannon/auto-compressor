"""PDF compression using Ghostscript with optional splitting."""

import logging
import os
import shutil
import math
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from PyPDF2 import PdfReader

import compress_ghostscript
import split_pdf
from exceptions import (
    EncryptionError,
    StructureError,
    MetadataCorruptionError,
    SplitError,
)
from utils import env_bool, env_float, env_int, get_file_size_mb, get_effective_cpu_count

logger = logging.getLogger(__name__)


# Threshold for switching to parallel compression (MB)
# Files below this use serial, above use parallel (faster for large files)
# Set to 30MB - serial compression can timeout on files 40MB+ due to slow Ghostscript settings
PARALLEL_THRESHOLD_MB = 30.0

# Files at or below this size stay on the serial path to avoid over-splitting (keeps mid-size PDFs to fewer parts).
PARALLEL_SERIAL_CUTOFF_MB = 100.0

# Force parallel compression for very high page counts, even when file size is modest.
PARALLEL_PAGE_THRESHOLD = env_int("PARALLEL_PAGE_THRESHOLD", 600)
# Allow page-count forced parallel even for modest-sized files (helps very page-dense PDFs).
PARALLEL_PAGE_MIN_MB = env_float("PARALLEL_PAGE_MIN_MB", 0.0)
# Minimum size to force parallel purely by page count (keeps small, dense files on serial first).
PARALLEL_PAGE_FORCE_MIN_MB = PARALLEL_THRESHOLD_MB
PDF_PRECHECK_ENABLED = env_bool("PDF_PRECHECK_ENABLED", True)

# Cap workers to env or available CPU to avoid thrash on small instances
def _resolve_parallel_workers(effective_cpu: Optional[int] = None) -> Tuple[int, Optional[int]]:
    """Resolve max parallel worker count at call time to respect dynamic CPU limits."""
    if effective_cpu is None:
        effective_cpu = get_effective_cpu_count()

    env_value = os.environ.get("PARALLEL_MAX_WORKERS")
    env_workers = None
    if env_value:
        try:
            env_workers = max(1, int(env_value))
        except ValueError:
            logger.warning("[compress] Invalid PARALLEL_MAX_WORKERS=%s; using effective CPU", env_value)
            env_workers = None

    if env_workers is not None:
        return max(1, env_workers), env_workers

    max_workers = env_workers if env_workers is not None else effective_cpu
    return max(1, min(max_workers, effective_cpu)), env_workers

# Skip compression for very small files (already optimized)
MIN_COMPRESSION_SIZE_MB = float(os.environ.get("MIN_COMPRESSION_SIZE_MB", "1.0"))
COMPRESSION_MODE = os.environ.get("COMPRESSION_MODE", "aggressive").lower()
ALLOW_LOSSY_COMPRESSION = os.environ.get("ALLOW_LOSSY_COMPRESSION", "1").lower() in ("1", "true", "yes")
SCANNED_CONFIDENCE_FOR_AGGRESSIVE = float(os.environ.get("SCANNED_CONFIDENCE_FOR_AGGRESSIVE", "70"))
DEFAULT_TARGET_CHUNK_MB = float(os.environ.get("TARGET_CHUNK_MB", "40"))
DEFAULT_MAX_CHUNK_MB = float(os.environ.get("MAX_CHUNK_MB", str(DEFAULT_TARGET_CHUNK_MB * 1.5)))


def resolve_compression_mode(input_path: Path) -> str:
    """Choose compression mode based on config and PDF characteristics."""
    mode = COMPRESSION_MODE
    if mode not in ("lossless", "aggressive", "adaptive"):
        logger.warning(f"[compress] Unknown COMPRESSION_MODE '{mode}', defaulting to lossless")
        mode = "lossless"

    if mode == "aggressive" and not ALLOW_LOSSY_COMPRESSION:
        logger.info("[compress] Aggressive mode requested but lossy compression disabled; using lossless")
        return "lossless"

    if mode != "adaptive":
        return mode

    if not ALLOW_LOSSY_COMPRESSION:
        return "lossless"

    try:
        from pdf_diagnostics import detect_already_compressed, detect_scanned_document

        already_compressed, _ = detect_already_compressed(input_path)
        if already_compressed:
            return "lossless"

        is_scanned, confidence = detect_scanned_document(input_path)
        if is_scanned and confidence >= SCANNED_CONFIDENCE_FOR_AGGRESSIVE:
            return "aggressive"
    except Exception as e:
        logger.warning(f"[compress] Adaptive mode analysis failed: {e}")

    return "lossless"


def compress_pdf(
    input_path: str,
    working_dir: Optional[Path] = None,
    split_threshold_mb: Optional[float] = None,
    split_trigger_mb: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None
) -> Dict:
    """
    Compress a PDF file using Ghostscript, optionally splitting if too large.

    For large files (>40MB), uses parallel compression for 3-4x faster processing.

    Args:
        input_path: Path to input PDF.
        working_dir: Directory for output file (default: same as input).
        split_threshold_mb: If set, split output into parts under this size.
        split_trigger_mb: If set, only split when output exceeds this size.
        progress_callback: Optional callback(percent, stage, message) for progress updates.

    Returns:
        Dict with: output_path(s), original_size_mb, compressed_size_mb,
                   reduction_percent, success, compression_method,
                   was_split, output_paths (list if split).

    Raises:
        FileNotFoundError: If input file doesn't exist.
        EncryptionError: If PDF is password-protected.
        StructureError: If PDF is corrupted or malformed.
        MetadataCorruptionError: If PDF metadata is corrupted.
        SplitError: If PDF cannot be split into small enough parts.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Helper to report progress
    def report_progress(percent: int, stage: str, message: str):
        if progress_callback:
            progress_callback(percent, stage, message)

    probe_info = None
    probe_bad = False
    bytes_per_page = None
    composition = {
        "already_compressed": False,
        "already_reason": "",
        "scanned": False,
        "scan_confidence": 0.0,
    }

    # Validate PDF before processing - catch encrypted files early
    page_count = None
    if PDF_PRECHECK_ENABLED:
        try:
            with open(input_path, 'rb') as f:
                reader = PdfReader(f, strict=False)
                # Do not block on "encrypted" flag; attempt to read pages regardless.
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                        logger.info(f"{input_path.name} flagged encrypted; attempted empty password and continuing.")
                    except Exception as de:
                        logger.warning(
                            f"{input_path.name} flagged encrypted; decrypt attempt failed ({de}), continuing anyway."
                        )

                page_count = len(reader.pages)
        except Exception as e:
            # Let Ghostscript try - it might handle edge cases
            logger.warning(f"PDF pre-validation warning (will continue): {e}")

    working_dir = working_dir or input_path.parent
    original_size = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size
    compression_mode = resolve_compression_mode(input_path)

    # ---------------------------------------------------------------------
    # Composition-aware mode overrides (avoid aggressive on vector/text PDFs
    # or on files already compressed by upstream tooling).
    # ---------------------------------------------------------------------
    try:
        from pdf_diagnostics import detect_already_compressed, detect_scanned_document

        composition["already_compressed"], composition["already_reason"] = detect_already_compressed(input_path)
        composition["scanned"], composition["scan_confidence"] = detect_scanned_document(input_path)
    except Exception as exc:  # diagnostics are best-effort
        logger.warning(f"[compress] Composition detection failed (continuing): {exc}")

    if composition["already_compressed"]:
        compression_mode = "lossless"
        logger.info("[compress] Detected already-compressed PDF (%s); forcing lossless path", composition["already_reason"])
    elif compression_mode == "aggressive" and not composition["scanned"]:
        # Mixed/vector-heavy PDFs bloat when forced through /screen; prefer lossless.
        compression_mode = "lossless"
        logger.info("[compress] Not scanned (or low confidence); preferring lossless to avoid inflation")
    elif compression_mode == "aggressive" and composition["scanned"] and composition["scan_confidence"] < SCANNED_CONFIDENCE_FOR_AGGRESSIVE:
        compression_mode = "lossless"
        logger.info(
            "[compress] Scanned but low confidence (%.1f%% < %.1f%%); using lossless",
            composition["scan_confidence"],
            SCANNED_CONFIDENCE_FOR_AGGRESSIVE,
        )

    logger.info(f"[compress] Compression mode: {compression_mode} (allow_lossy={ALLOW_LOSSY_COMPRESSION})")
    effective_split_trigger = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb
    split_requested = split_threshold_mb is not None and split_threshold_mb > 0
    bytes_per_page = (original_bytes / max(page_count, 1)) if page_count else None
    quality_mode = "aggressive_150dpi" if compression_mode == "aggressive" else "lossless"

    def _augment(resp: Dict) -> Dict:
        resp.setdefault("quality_mode", quality_mode)
        resp.setdefault(
            "analysis",
            {
                "page_count": page_count,
                "bytes_per_page": bytes_per_page,
                "probe": probe_info,
                "probe_bad": probe_bad,
                "composition": composition,
            },
        )
        return resp

    # Skip compression for very small files - usually already optimized
    if original_size < MIN_COMPRESSION_SIZE_MB:
        logger.info(f"[compress] Skipping {input_path.name}: {original_size:.2f}MB < {MIN_COMPRESSION_SIZE_MB}MB threshold")

        # Still split if above threshold (rare for small files)
        if split_threshold_mb and effective_split_trigger and original_size > effective_split_trigger:
            output_paths = split_pdf.split_for_delivery(
                input_path, working_dir, input_path.stem,
                threshold_mb=split_threshold_mb,
                progress_callback=progress_callback,
                prefer_binary=True,
                skip_optimization_under_threshold=True,
            )
            return _augment({
                "output_path": str(output_paths[0]),
                "output_paths": [str(p) for p in output_paths],
                "original_size_mb": round(original_size, 2),
                "compressed_size_mb": round(original_size, 2),
                "reduction_percent": 0.0,
                "compression_method": "skipped",
                "compression_mode": compression_mode,
                "was_split": len(output_paths) > 1,
                "total_parts": len(output_paths),
                "success": True,
                "note": "File under compression threshold, split only",
                "page_count": page_count,
                "part_sizes": [p.stat().st_size for p in output_paths]
            })

        return _augment({
            "output_path": str(input_path),
            "output_paths": [str(input_path)],
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(original_size, 2),
            "reduction_percent": 0.0,
            "compression_method": "skipped",
            "compression_mode": compression_mode,
            "was_split": False,
            "total_parts": 1,
            "success": True,
            "note": f"File under {MIN_COMPRESSION_SIZE_MB}MB threshold - already optimized",
            "page_count": page_count,
            "part_sizes": [original_bytes]
        })

    logger.info(f"[SIZE_CHECK] Input: {input_path.name} = {original_bytes} bytes ({original_size:.2f}MB)")

    # Optional micro-probe to predict inflation/throughput on aggressive mode
    probe_info = None
    probe_bad = False
    if compression_mode == "aggressive" and original_size >= PARALLEL_THRESHOLD_MB and original_size <= 400 and page_count:
        try:
            # Micro-probe always uses lossless to avoid inflate-y aggressive test runs
            probe_info = compress_ghostscript.run_micro_probe(input_path, "lossless")
            probe_bad = (
                not probe_info.get("success")
                or probe_info.get("delta_pct", 0) > 3
                or probe_info.get("elapsed", 0) > compress_ghostscript.PROBE_TIME_BUDGET_SEC
            )
            logger.info(
                "[compress] Micro-probe pages=%s delta=%.1f%% elapsed=%.1fs success=%s",
                probe_info.get("pages"),
                probe_info.get("delta_pct", 0.0),
                probe_info.get("elapsed", 0.0),
                probe_info.get("success"),
            )
        except Exception as exc:
            logger.warning("[compress] Micro-probe failed (will ignore): %s", exc)

    # If probe indicates risk, downgrade to lossless path to protect quality/SLA.
    if probe_bad and compression_mode == "aggressive":
        logger.info("[compress] Probe indicated risk; downgrading to lossless path")
        compression_mode = "lossless"
        quality_mode = "lossless"

    # =========================================================================
    # ROUTE: Large files use parallel compression for speed
    # Mid-size files (<~100MB or only 1–2 chunks) stay serial to avoid extra parts
    # =========================================================================
    est_chunks = math.ceil(original_size / DEFAULT_TARGET_CHUNK_MB) if DEFAULT_TARGET_CHUNK_MB > 0 else 3
    page_parallel_candidate = (
        page_count is not None
        and PARALLEL_PAGE_THRESHOLD > 0
        and page_count >= PARALLEL_PAGE_THRESHOLD
        and original_size >= PARALLEL_PAGE_MIN_MB
    )
    force_parallel_by_pages = page_parallel_candidate and original_size >= PARALLEL_PAGE_FORCE_MIN_MB
    # For no-split flows, be more conservative about parallelizing midsize files to avoid
    # heavy split/merge overhead. Require >150MB before size-based parallel if split is off.
    size_parallel = (
        original_size > PARALLEL_THRESHOLD_MB
        and (
            (split_requested and original_size > PARALLEL_SERIAL_CUTOFF_MB)
            or (not split_requested and original_size > 200.0)
        )
        and est_chunks > 3
    )
    use_parallel = force_parallel_by_pages or size_parallel or page_parallel_candidate

    # If probe shows inflation/slow path and we weren't forced by pages, prefer serial to protect quality/SLA.
    if probe_bad and not force_parallel_by_pages:
        use_parallel = False

    if page_parallel_candidate and not force_parallel_by_pages:
        logger.info(
            "[compress] Page count %s >= %s but size %.1fMB < %.1fMB; trying serial first",
            page_count,
            PARALLEL_PAGE_THRESHOLD,
            original_size,
            PARALLEL_PAGE_FORCE_MIN_MB,
        )

    def run_parallel(reason: str) -> Dict:
        if reason == "page_count":
            logger.info(
                "[PARALLEL] Page count %s >= %s; using parallel compression",
                page_count,
                PARALLEL_PAGE_THRESHOLD,
            )
        elif reason == "size":
            logger.info(
                "[PARALLEL] Large file %.1fMB (est_chunks=%s); using parallel compression",
                original_size,
                est_chunks,
            )
        elif reason == "timeout":
            logger.info("[PARALLEL] Serial compression timed out; switching to parallel")
        else:
            logger.info("[PARALLEL] Using parallel compression (reason=%s)", reason)
        effective_cpu = get_effective_cpu_count()
        max_workers, env_workers = _resolve_parallel_workers(effective_cpu)
        env_label = env_workers if env_workers is not None else "unset"
        logger.info(
            "[PARALLEL] Worker cap: max_workers=%s (cpu=%s, PARALLEL_MAX_WORKERS=%s)",
            max_workers,
            effective_cpu,
            env_label,
        )
        target_chunk_mb = DEFAULT_TARGET_CHUNK_MB
        max_chunk_mb = DEFAULT_MAX_CHUNK_MB
        if compression_mode == "lossless":
            target_chunk_mb = max(DEFAULT_TARGET_CHUNK_MB, 60.0)
            max_chunk_mb = max(DEFAULT_MAX_CHUNK_MB, target_chunk_mb * 1.5)
            logger.info(
                "[compress] Lossless mode: larger chunks (target %.1fMB, max %.1fMB)",
                target_chunk_mb,
                max_chunk_mb,
            )
        else:
            # If we were forced into parallel due to page count, keep chunks small so
            # each Ghostscript run handles fewer pages and avoids long timeouts.
            if reason == "page_count":
                target_chunk_mb = max(DEFAULT_TARGET_CHUNK_MB, 40.0)
                max_chunk_mb = max(DEFAULT_MAX_CHUNK_MB, target_chunk_mb * 1.5)
            # Otherwise, no-split path can favor fewer, larger chunks to speed merge.
            elif not split_requested and original_size <= 150.0:
                target_chunk_mb = max(target_chunk_mb, 80.0)
                max_chunk_mb = max(max_chunk_mb, target_chunk_mb * 1.5)
            logger.info(
                "[compress] Aggressive mode: standard chunks (target %.1fMB, max %.1fMB)",
                target_chunk_mb,
                max_chunk_mb,
            )
        return compress_ghostscript.compress_parallel(
            input_path=input_path,
            working_dir=working_dir,
            base_name=input_path.stem,
            split_threshold_mb=split_threshold_mb,
            split_trigger_mb=effective_split_trigger,
            progress_callback=progress_callback,
            max_workers=max_workers,
            compression_mode=compression_mode,
            target_chunk_mb=target_chunk_mb,
            max_chunk_mb=max_chunk_mb,
            input_page_count=page_count,
        )

    if force_parallel_by_pages:
        logger.info(
            "[compress] Page count %s >= %s; using parallel compression to avoid serial timeouts",
            page_count,
            PARALLEL_PAGE_THRESHOLD,
        )
        return _augment(run_parallel("page_count"))
    if size_parallel:
        if split_requested:
            logger.info("[compress] Large file with split requested; using parallel compression for speed")
        else:
            logger.info("[compress] Large file detected; using parallel compression for speed")
        return _augment(run_parallel("size"))
    if split_requested:
        logger.info(
            "[compress] Split requested; trying serial compression first for maximum savings "
            "and falling back to parallel on timeout",
        )

    # =========================================================================
    # ROUTE: Small files use serial compression (existing logic)
    # =========================================================================
    logger.info(f"Compressing: {input_path.name} ({original_size:.1f}MB)")
    output_path = working_dir / f"{input_path.stem}_compressed.pdf"

    report_progress(15, "compressing", "Compressing PDF...")

    # Try Ghostscript compression
    compress_fn = (
        compress_ghostscript.compress_pdf_lossless
        if compression_mode == "lossless"
        else compress_ghostscript.compress_pdf_with_ghostscript
    )
    success, message = compress_fn(input_path, output_path)

    if not success:
        msg_lower = message.lower()
        timed_out = "timeout" in msg_lower
        if timed_out and split_requested:
            logger.warning(
                "[compress] Serial compression timed out; retrying in parallel for %s",
                input_path.name,
            )
            result = run_parallel("timeout")
            parts = result.get("total_parts") or len(result.get("output_paths", []))
            out_mb = result.get("compressed_size_mb", 0.0)
            method = result.get("compression_method", "unknown")
            logger.info(
                "[compress] Parallel retry complete: method=%s parts=%s out=%.1fMB",
                method,
                parts,
                out_mb,
            )
            return _augment(result)
        if timed_out and not split_requested and use_parallel:
            logger.warning(
                "[compress] Serial compression timed out; falling back to parallel for %s",
                input_path.name,
            )
            result = run_parallel("timeout")
            parts = result.get("total_parts") or len(result.get("output_paths", []))
            out_mb = result.get("compressed_size_mb", 0.0)
            method = result.get("compression_method", "unknown")
            logger.info(
                "[compress] Parallel fallback complete: method=%s parts=%s out=%.1fMB",
                method,
                parts,
                out_mb,
            )
            return _augment(result)
        # Map error message to specific exception type
        if 'password' in msg_lower or 'encrypt' in msg_lower or 'locked' in msg_lower:
            raise EncryptionError.for_file(input_path.name)
        elif 'corrupted' in msg_lower or 'damaged' in msg_lower or 'structure' in msg_lower:
            raise StructureError.for_file(input_path.name, message)
        elif 'metadata' in msg_lower or 'internal data' in msg_lower:
            raise MetadataCorruptionError.for_file(input_path.name)
        else:
            raise StructureError.for_file(input_path.name, message)

    # Ensure Ghostscript actually produced an output file
    if not output_path.exists():
        raise StructureError.for_file(input_path.name, "Compression output not created")

    compressed_size = get_file_size_mb(output_path)
    compressed_bytes = output_path.stat().st_size

    logger.info(f"[SIZE_CHECK] After GS: {output_path.name} = {compressed_bytes} bytes ({compressed_size:.2f}MB)")

    report_progress(60, "compressing", "Compression complete, verifying...")

    # If compression made it bigger, return original
    if compressed_size >= original_size:
        bloat_pct = ((compressed_size - original_size) / original_size) * 100 if original_size > 0 else 0.0
        logger.warning(
            "Compression increased size (%.2fMB -> %.2fMB, +%.1f%%), returning original",
            original_size,
            compressed_size,
            bloat_pct,
        )
        output_path.unlink()
        shutil.copy2(input_path, output_path)
        compressed_size = original_size

        # Check if we still need to split the original
        if split_threshold_mb and effective_split_trigger and compressed_size > effective_split_trigger:
            output_paths = split_pdf.split_for_delivery(
                output_path, working_dir, input_path.stem,
                threshold_mb=split_threshold_mb,
                progress_callback=progress_callback,
                prefer_binary=True,
                skip_optimization_under_threshold=True,
            )
            return _augment({
                "output_path": str(output_paths[0]),
                "output_paths": [str(p) for p in output_paths],
                "original_size_mb": round(original_size, 2),
                "compressed_size_mb": round(original_size, 2),
                "reduction_percent": 0.0,
                "compression_method": "none",
                "compression_mode": compression_mode,
                "was_split": len(output_paths) > 1,
                "total_parts": len(output_paths),
                "success": True,
                "note": "Already optimized, split for size",
                "page_count": page_count,
                "part_sizes": [p.stat().st_size for p in output_paths],
                "bloat_detected": True,
                "bloat_pct": round(bloat_pct, 1),
                "bloat_action": "return_original",
            })

        single_size = output_path.stat().st_size
        return _augment({
            "output_path": str(output_path),
            "output_paths": [str(output_path)],
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(original_size, 2),
            "reduction_percent": 0.0,
            "compression_method": "none",
            "compression_mode": compression_mode,
            "was_split": False,
            "total_parts": 1,
            "success": True,
            "note": "Already optimized",
            "page_count": page_count,
            "part_sizes": [single_size],
            "bloat_detected": True,
            "bloat_pct": round(bloat_pct, 1),
            "bloat_action": "return_original",
        })

    reduction = ((original_size - compressed_size) / original_size) * 100

    logger.info(f"Done: {original_size:.1f}MB -> {compressed_size:.1f}MB ({reduction:.1f}%)")

    # Check if splitting is needed
    if split_threshold_mb and effective_split_trigger and compressed_size > effective_split_trigger:
        logger.info(f"Compressed file still {compressed_size:.1f}MB, splitting...")
        report_progress(70, "splitting", "Splitting into parts...")

        output_paths = split_pdf.split_for_delivery(
            output_path, working_dir, input_path.stem,
            threshold_mb=split_threshold_mb,
            progress_callback=progress_callback,
            prefer_binary=True,
            skip_optimization_under_threshold=True,
        )

        # Log size of each split part
        for i, part_path in enumerate(output_paths):
            part_bytes = part_path.stat().st_size
            part_mb = part_bytes / (1024 * 1024)
            logger.info(f"[SIZE_CHECK] Part {i+1}: {part_path.name} = {part_bytes} bytes ({part_mb:.2f}MB)")

        report_progress(95, "finalizing", "Finalizing...")
        return _augment({
            "output_path": str(output_paths[0]),
            "output_paths": [str(p) for p in output_paths],
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(compressed_size, 2),
            "reduction_percent": round(reduction, 1),
            "compression_method": "ghostscript",
            "compression_mode": compression_mode,
            "was_split": len(output_paths) > 1,
            "total_parts": len(output_paths),
            "success": True,
            "page_count": page_count,
            "part_sizes": [p.stat().st_size for p in output_paths]
        })

    # No splitting needed
    logger.info(f"[SIZE_CHECK] Final: {output_path.name} = {compressed_bytes} bytes ({compressed_size:.2f}MB)")
    report_progress(95, "finalizing", "Finalizing...")

    single_size = output_path.stat().st_size

    return _augment({
        "output_path": str(output_path),
        "output_paths": [str(output_path)],
        "original_size_mb": round(original_size, 2),
        "compressed_size_mb": round(compressed_size, 2),
        "reduction_percent": round(reduction, 1),
        "compression_method": "ghostscript",
        "compression_mode": compression_mode,
        "was_split": False,
        "total_parts": 1,
        "success": True,
        "page_count": page_count,
        "part_sizes": [single_size]
    })
