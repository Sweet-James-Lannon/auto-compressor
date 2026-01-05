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
from utils import get_file_size_mb, get_effective_cpu_count

logger = logging.getLogger(__name__)


# Threshold for switching to parallel compression (MB)
# Files below this use serial, above use parallel (faster for large files)
# Set to 30MB - serial compression can timeout on files 40MB+ due to slow Ghostscript settings
PARALLEL_THRESHOLD_MB = 30.0

# Files at or below this size stay on the serial path to avoid over-splitting (keeps mid-size PDFs to fewer parts).
PARALLEL_SERIAL_CUTOFF_MB = 100.0

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

    report_progress(5, "validating", "Validating PDF...")

    # Validate PDF before processing - catch encrypted files early
    page_count = None
    try:
        with open(input_path, 'rb') as f:
            reader = PdfReader(f, strict=False)
            # Do not block on "encrypted" flag; attempt to read pages regardless.
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                    logger.info(f"{input_path.name} flagged encrypted; attempted empty password and continuing.")
                except Exception as de:
                    logger.warning(f"{input_path.name} flagged encrypted; decrypt attempt failed ({de}), continuing anyway.")

            page_count = len(reader.pages)
    except Exception as e:
        # Let Ghostscript try - it might handle edge cases
        logger.warning(f"PDF pre-validation warning (will continue): {e}")

    working_dir = working_dir or input_path.parent
    original_size = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size
    compression_mode = resolve_compression_mode(input_path)
    logger.info(f"[compress] Compression mode: {compression_mode} (allow_lossy={ALLOW_LOSSY_COMPRESSION})")
    effective_split_trigger = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb

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
            return {
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
            }

        return {
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
        }

    logger.info(f"[SIZE_CHECK] Input: {input_path.name} = {original_bytes} bytes ({original_size:.2f}MB)")

    # =========================================================================
    # ROUTE: Large files use parallel compression for speed
    # Mid-size files (<~100MB or only 1–2 chunks) stay serial to avoid extra parts
    # =========================================================================
    est_chunks = math.ceil(original_size / DEFAULT_TARGET_CHUNK_MB) if DEFAULT_TARGET_CHUNK_MB > 0 else 3
    use_parallel = (
        original_size > PARALLEL_THRESHOLD_MB
        and original_size > PARALLEL_SERIAL_CUTOFF_MB
        and est_chunks > 2
    )

    if use_parallel:
        logger.info(
            f"[PARALLEL] File {original_size:.1f}MB > {PARALLEL_THRESHOLD_MB}MB threshold "
            f"and > {PARALLEL_SERIAL_CUTOFF_MB}MB cutoff, using parallel compression (est_chunks={est_chunks})"
        )
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
            logger.info(
                "[compress] Aggressive mode: standard chunks (target %.1fMB, max %.1fMB)",
                target_chunk_mb,
                max_chunk_mb,
            )
        result = compress_ghostscript.compress_parallel(
            input_path=input_path,
            working_dir=working_dir,
            base_name=input_path.stem,
            split_threshold_mb=split_threshold_mb or 25.0,
            split_trigger_mb=effective_split_trigger,
            progress_callback=progress_callback,
            max_workers=max_workers,
            compression_mode=compression_mode,
            target_chunk_mb=target_chunk_mb,
            max_chunk_mb=max_chunk_mb,
        )
        try:
            if split_threshold_mb:
                expected_parts = math.ceil(result["compressed_size_mb"] / split_threshold_mb)
                if result.get("total_parts", 0) > expected_parts:
                    logger.warning(
                        "[PARALLEL] Parts (%s) exceeded expected ceil(%s/%.1f)=%s; sizes=%s",
                        result.get("total_parts"),
                        result.get("compressed_size_mb"),
                        split_threshold_mb,
                        expected_parts,
                        result.get("part_sizes"),
                    )
        except Exception:
            pass
        return result

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
        # Map error message to specific exception type
        msg_lower = message.lower()
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
        logger.warning(f"Compression increased size, returning original")
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
            return {
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
                "part_sizes": [p.stat().st_size for p in output_paths]
            }

        single_size = output_path.stat().st_size
        return {
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
            "part_sizes": [single_size]
        }

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
        return {
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
        }

    # No splitting needed
    logger.info(f"[SIZE_CHECK] Final: {output_path.name} = {compressed_bytes} bytes ({compressed_size:.2f}MB)")
    report_progress(95, "finalizing", "Finalizing...")

    single_size = output_path.stat().st_size

    return {
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
    }
