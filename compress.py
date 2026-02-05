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
    ProcessingTimeoutError,
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
PROBE_INFLATION_WARN_PCT = env_float("PROBE_INFLATION_WARN_PCT", 12.0)
PROBE_INFLATION_FORCE_LOSSLESS_PCT = env_float("PROBE_INFLATION_FORCE_LOSSLESS_PCT", 60.0)
ULTRA_FALLBACK_MIN_REDUCTION_PCT = env_float("ULTRA_FALLBACK_MIN_REDUCTION_PCT", 5.0)
ULTRA_FALLBACK_MIN_SIZE_MB = env_float("ULTRA_FALLBACK_MIN_SIZE_MB", 50.0)
ULTRA_FALLBACK_JPEGQ = env_int("ULTRA_FALLBACK_JPEGQ", 55)
ULTRA_FALLBACK_COLOR_DPI = env_int("ULTRA_FALLBACK_COLOR_DPI", 120)
ULTRA_FALLBACK_GRAY_DPI = env_int("ULTRA_FALLBACK_GRAY_DPI", 150)
ULTRA_FALLBACK_MONO_DPI = env_int("ULTRA_FALLBACK_MONO_DPI", 300)
LOSSLESS_ALREADY_TIMEOUT_SEC = env_int("LOSSLESS_ALREADY_TIMEOUT_SEC", 240)
ALLOW_PARALLEL_ON_ALREADY = env_bool("ALLOW_PARALLEL_ON_ALREADY", False)


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
    probe_force_lossless = False
    probe_action = None
    bytes_per_page = None
    composition = {
        "already_compressed": False,
        "already_reason": "",
        "scanned": False,
        "scan_confidence": 0.0,
    }
    force_serial_for_already = False
    serial_timeout_override = None

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
        logger.info("[compress] Detected already-compressed PDF (%s); proceeding with %s mode (advisory only)", composition["already_reason"], compression_mode)
    elif compression_mode == "adaptive" and composition["scanned"] and composition["scan_confidence"] >= SCANNED_CONFIDENCE_FOR_AGGRESSIVE:
        compression_mode = "aggressive"
        logger.info("[compress] Adaptive: high-confidence scanned; choosing aggressive")

    logger.info(f"[compress] Compression mode: {compression_mode} (allow_lossy={ALLOW_LOSSY_COMPRESSION})")
    effective_split_trigger = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb
    split_requested = split_threshold_mb is not None and split_threshold_mb > 0
    bytes_per_page = (original_bytes / max(page_count, 1)) if page_count else None
    quality_mode = "aggressive_72dpi" if compression_mode == "aggressive" else "lossless"

    logger.info(
        "[DIAG] GS settings: color_dpi=%s gray_dpi=%s mono_dpi=%s "
        "color_downsample=%s gray_downsample=%s quality_mode=%s",
        compress_ghostscript.GS_COLOR_IMAGE_RESOLUTION,
        compress_ghostscript.GS_GRAY_IMAGE_RESOLUTION,
        compress_ghostscript.GS_MONO_IMAGE_RESOLUTION,
        compress_ghostscript.GS_COLOR_DOWNSAMPLE_TYPE,
        compress_ghostscript.GS_GRAY_DOWNSAMPLE_TYPE,
        quality_mode,
    )

    def _augment(resp: Dict) -> Dict:
        resp.setdefault("quality_mode", quality_mode)
        resp.setdefault(
            "analysis",
            {
                "page_count": page_count,
                "bytes_per_page": bytes_per_page,
                "probe": probe_info,
                "probe_bad": probe_bad,
                "probe_action": probe_action,
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

    logger.info(
        "[DIAG] === JOB START === file=%s size_mb=%.2f pages=%s bytes_per_page=%s "
        "mode=%s allow_lossy=%s already_compressed=%s(%s) scanned=%s(%.0f%%) "
        "split_requested=%s split_threshold=%s",
        input_path.name, original_size, page_count,
        f"{bytes_per_page:.0f}" if bytes_per_page else "unknown",
        compression_mode, ALLOW_LOSSY_COMPRESSION,
        composition["already_compressed"], composition["already_reason"] or "n/a",
        composition["scanned"], composition["scan_confidence"],
        split_requested, split_threshold_mb,
    )

    # Optional micro-probe to predict inflation/throughput on aggressive mode
    probe_info = None
    probe_bad = False
    probe_delta_pct = 0.0
    probe_elapsed = 0.0
    if compression_mode == "aggressive" and original_size >= PARALLEL_THRESHOLD_MB and original_size <= 400 and page_count:
        try:
            # Micro-probe uses the SAME mode as actual compression to accurately predict inflation
            probe_info = compress_ghostscript.run_micro_probe(input_path, compression_mode)
            probe_delta_pct = probe_info.get("delta_pct", 0.0)
            probe_elapsed = probe_info.get("elapsed", 0.0)
            probe_force_lossless = (not probe_info.get("success")) or probe_delta_pct >= PROBE_INFLATION_FORCE_LOSSLESS_PCT
            # Threshold: mild inflation (>PROBE_INFLATION_WARN_PCT) signals risk; extreme inflation
            # forces a lossless path rather than skipping compression entirely.
            probe_bad = (
                not probe_info.get("success")
                or probe_delta_pct > PROBE_INFLATION_WARN_PCT
                or probe_elapsed > compress_ghostscript.PROBE_TIME_BUDGET_SEC * 1.5
            )
            logger.info(
                "[compress] Micro-probe pages=%s delta=%.1f%% elapsed=%.1fs success=%s",
                probe_info.get("pages"),
                probe_delta_pct,
                probe_elapsed,
                probe_info.get("success"),
            )
        except Exception as exc:
            logger.warning("[compress] Micro-probe failed (will ignore): %s", exc)

    # Micro-probe is advisory only — log the warning but let compression run.
    # The bloat handler after compression catches inflation and falls back.
    if probe_bad and original_size > PARALLEL_THRESHOLD_MB:
        delta_pct = probe_info.get("delta_pct", 0) if probe_info else 0
        probe_action = "warn_only"
        logger.info(
            "[compress] Probe warning (%.1f%%); proceeding — bloat handler will catch inflation",
            delta_pct,
        )

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
    # heavy split/merge overhead. Require >PARALLEL_SERIAL_CUTOFF_MB for size-based parallel.
    size_parallel = (
        original_size > PARALLEL_THRESHOLD_MB
        and original_size > PARALLEL_SERIAL_CUTOFF_MB
        and est_chunks > 3
    )
    use_parallel = force_parallel_by_pages or size_parallel or page_parallel_candidate

    logger.info(
        "[DIAG] Route decision: use_parallel=%s (size_parallel=%s force_by_pages=%s "
        "page_candidate=%s) est_chunks=%s file_mb=%.1f",
        use_parallel, size_parallel, force_parallel_by_pages,
        page_parallel_candidate, est_chunks, original_size,
    )

    # Note: force_serial_for_already removed — let normal size-based routing decide.
    # "Already compressed" detection is now advisory only.

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
            allow_lossy=ALLOW_LOSSY_COMPRESSION,
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
    success, message = compress_fn(input_path, output_path, timeout_override=serial_timeout_override)

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
        if timed_out and not split_requested:
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
        if timed_out:
            raise ProcessingTimeoutError.for_file(input_path.name)
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

    # If aggressive compression made it bigger, try lossless before giving up
    lossless_retry_worked = False
    if compressed_size >= original_size:
        bloat_pct = ((compressed_size - original_size) / original_size) * 100 if original_size > 0 else 0.0

        # Safety net: if aggressive bloated, try lossless
        if compression_mode == "aggressive":
            logger.warning(
                "Aggressive bloated (%.2fMB -> %.2fMB, +%.1f%%); retrying lossless",
                original_size,
                compressed_size,
                bloat_pct,
            )
            output_path.unlink(missing_ok=True)
            lossless_ok, lossless_msg = compress_ghostscript.compress_pdf_lossless(input_path, output_path)
            if lossless_ok and output_path.exists():
                compressed_size = get_file_size_mb(output_path)
                compressed_bytes = output_path.stat().st_size
                logger.info(
                    "[compress] Lossless retry: %.2fMB -> %.2fMB",
                    original_size,
                    compressed_size,
                )
                # If lossless worked, skip the "return original" block
                if compressed_size < original_size:
                    compression_mode = "lossless"
                    quality_mode = "lossless"
                    lossless_retry_worked = True
                else:
                    bloat_pct = ((compressed_size - original_size) / original_size) * 100 if original_size > 0 else 0.0
                    logger.warning(
                        "Lossless also bloated (%.2fMB -> %.2fMB, +%.1f%%); returning original",
                        original_size,
                        compressed_size,
                        bloat_pct,
                    )
                    output_path.unlink()
                    shutil.copy2(input_path, output_path)
                    compressed_size = original_size
            else:
                logger.warning("Lossless retry failed: %s; returning original", lossless_msg)
                output_path.unlink(missing_ok=True)
                shutil.copy2(input_path, output_path)
                compressed_size = original_size
        else:
            # Lossless mode bloated. If this file was flagged "already compressed" and
            # lossy is allowed, try aggressive compression before giving up — the
            # "already compressed" detection may have been a false positive.
            if composition["already_compressed"] and ALLOW_LOSSY_COMPRESSION:
                logger.info(
                    "[compress] Lossless bloated on 'already-compressed' file (%.2fMB -> %.2fMB, +%.1f%%); "
                    "trying aggressive fallback",
                    original_size,
                    compressed_size,
                    bloat_pct,
                )
                output_path.unlink(missing_ok=True)
                agg_ok, agg_msg = compress_ghostscript.compress_pdf_with_ghostscript(input_path, output_path)
                if agg_ok and output_path.exists():
                    agg_size = get_file_size_mb(output_path)
                    if agg_size < original_size:
                        compressed_size = agg_size
                        compressed_bytes = output_path.stat().st_size
                        compression_mode = "aggressive"
                        quality_mode = "aggressive_72dpi"
                        lossless_retry_worked = True
                        logger.info(
                            "[compress] Aggressive fallback succeeded: %.2fMB -> %.2fMB",
                            original_size,
                            agg_size,
                        )
                    else:
                        logger.warning(
                            "[compress] Aggressive fallback also bloated (%.2fMB -> %.2fMB); returning original",
                            original_size,
                            agg_size,
                        )
                        output_path.unlink(missing_ok=True)
                        shutil.copy2(input_path, output_path)
                        compressed_size = original_size
                else:
                    logger.warning("[compress] Aggressive fallback failed: %s; returning original", agg_msg)
                    output_path.unlink(missing_ok=True)
                    shutil.copy2(input_path, output_path)
                    compressed_size = original_size
            else:
                logger.warning(
                    "Compression increased size (%.2fMB -> %.2fMB, +%.1f%%), returning original",
                    original_size,
                    compressed_size,
                    bloat_pct,
                )
                output_path.unlink()
                shutil.copy2(input_path, output_path)
                compressed_size = original_size

        # If lossless retry worked, skip the "return original" block and continue to success path
        if not lossless_retry_worked:
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
    # If we barely saved anything and lossy is allowed, try an ultra fallback with lower DPI/JPEGQ.
    ultra_used = False
    if (
        ALLOW_LOSSY_COMPRESSION
        and original_size >= ULTRA_FALLBACK_MIN_SIZE_MB
        and reduction < ULTRA_FALLBACK_MIN_REDUCTION_PCT
    ):
        logger.warning(
            "[compress] Low savings (%.1f%%); running ultra fallback (JPEGQ=%s, DPI=%s/%s/%s)",
            reduction,
            ULTRA_FALLBACK_JPEGQ,
            ULTRA_FALLBACK_COLOR_DPI,
            ULTRA_FALLBACK_GRAY_DPI,
            ULTRA_FALLBACK_MONO_DPI,
        )
        # Save pre-ultra output so we can restore if ultra inflates.
        pre_ultra_path = working_dir / f"{input_path.stem}_pre_ultra.pdf"
        shutil.copy2(output_path, pre_ultra_path)
        pre_ultra_size = compressed_size
        pre_ultra_bytes = compressed_bytes

        ultra_ok, ultra_msg = compress_ghostscript.compress_ultra_fallback(
            input_path,
            output_path,
            jpeg_quality=ULTRA_FALLBACK_JPEGQ,
            color_res=ULTRA_FALLBACK_COLOR_DPI,
            gray_res=ULTRA_FALLBACK_GRAY_DPI,
            mono_res=ULTRA_FALLBACK_MONO_DPI,
        )
        if ultra_ok and output_path.exists():
            ultra_size = get_file_size_mb(output_path)
            # Guard: if ultra output is larger than original, discard it.
            if ultra_size >= original_size:
                logger.warning(
                    "[compress] Ultra fallback inflated (%.1fMB -> %.1fMB); restoring pre-ultra output",
                    original_size,
                    ultra_size,
                )
                output_path.unlink(missing_ok=True)
                shutil.move(str(pre_ultra_path), str(output_path))
                compressed_size = pre_ultra_size
                compressed_bytes = pre_ultra_bytes
                reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0.0
            else:
                compressed_size = ultra_size
                compressed_bytes = output_path.stat().st_size
                reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0.0
                compression_mode = "aggressive"
                quality_mode = "aggressive_72dpi"
                ultra_used = True
                logger.info("[compress] Ultra fallback: %.1fMB -> %.1fMB (%.1f%%)", original_size, compressed_size, reduction)
        else:
            logger.warning("[compress] Ultra fallback failed: %s; restoring pre-ultra output", ultra_msg)
            # Restore the pre-ultra output in case ultra corrupted the file.
            if pre_ultra_path.exists():
                output_path.unlink(missing_ok=True)
                shutil.move(str(pre_ultra_path), str(output_path))
                compressed_size = pre_ultra_size
                compressed_bytes = pre_ultra_bytes
                reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0.0

        pre_ultra_path.unlink(missing_ok=True)

    logger.info(
        "[DIAG] === JOB END === file=%s original_mb=%.2f compressed_mb=%.2f "
        "reduction=%.1f%% method=%s mode=%s quality=%s ultra=%s "
        "bloat_detected=%s split=%s parts=%s",
        input_path.name, original_size, compressed_size,
        reduction,
        "ghostscript_ultra" if ultra_used else "ghostscript",
        compression_mode, quality_mode, ultra_used,
        compressed_size >= original_size and not lossless_retry_worked,
        split_requested,
        1,
    )

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
        "compression_method": "ghostscript_ultra" if ultra_used else "ghostscript",
        "compression_mode": compression_mode,
        "was_split": False,
        "total_parts": 1,
        "success": True,
        "page_count": page_count,
        "part_sizes": [single_size]
    })
