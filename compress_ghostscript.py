"""Ghostscript PDF compression for scanned legal documents."""

import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from exceptions import SplitError
from utils import get_effective_cpu_count

logger = logging.getLogger(__name__)

TARGET_CHUNK_MB = float(os.environ.get("TARGET_CHUNK_MB", "40"))
MAX_CHUNK_MB = float(os.environ.get("MAX_CHUNK_MB", str(TARGET_CHUNK_MB * 1.5)))
MAX_PARALLEL_CHUNKS = int(os.environ.get("MAX_PARALLEL_CHUNKS", "16"))
MAX_PAGES_PER_CHUNK = int(os.environ.get("MAX_PAGES_PER_CHUNK", "200"))
# Minimum chunk size to avoid spawning tiny Ghostscript jobs that add overhead.
MIN_CHUNK_MB = 20.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.environ.get(name) or "").strip()
    return raw if raw in allowed else default


GS_FAST_WEB_VIEW = _env_bool("GS_FAST_WEB_VIEW", True)
GS_COLOR_DOWNSAMPLE_TYPE = _env_choice(
    "GS_COLOR_DOWNSAMPLE_TYPE",
    "/Bicubic",
    ("/Subsample", "/Average", "/Bicubic"),
)
GS_GRAY_DOWNSAMPLE_TYPE = _env_choice(
    "GS_GRAY_DOWNSAMPLE_TYPE",
    "/Bicubic",
    ("/Subsample", "/Average", "/Bicubic"),
)
GS_BAND_HEIGHT = _env_int("GS_BAND_HEIGHT", 100)
GS_BAND_BUFFER_SPACE_MB = _env_int("GS_BAND_BUFFER_SPACE_MB", 500)
PARALLEL_SERIAL_FALLBACK = _env_bool("PARALLEL_SERIAL_FALLBACK", True)


def _resolve_gs_threads(num_threads: Optional[int]) -> int:
    """Resolve Ghostscript rendering threads, honoring explicit and env overrides."""
    if num_threads is not None:
        try:
            return max(1, int(num_threads))
        except (TypeError, ValueError):
            return 1

    env_threads = os.environ.get("GS_NUM_RENDERING_THREADS")
    if env_threads:
        try:
            return max(1, int(env_threads))
        except ValueError:
            return 1

    effective_cpu = get_effective_cpu_count()
    return max(1, min(4, effective_cpu))


def optimize_split_part(
    input_path: Path,
    output_path: Path,
    num_threads: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Lightweight optimization for split PDF parts - removes duplicate resources
    without re-encoding images.

    Use this for split parts where images are already compressed. The /default
    setting preserves image quality while removing PyPDF2's duplicated resources.

    Args:
        input_path: Source PDF (split part with duplicated resources)
        output_path: Destination for optimized PDF

    Returns:
        (success, message) tuple
    """
    return _run_lossless_ghostscript(
        input_path,
        output_path,
        "Optimizing split part",
        num_threads=num_threads,
    )


def _run_lossless_ghostscript(
    input_path: Path,
    output_path: Path,
    label: str,
    num_threads: Optional[int] = None,
) -> Tuple[bool, str]:
    """Run Ghostscript with lossless settings (no downsampling, pass-through images)."""
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use /default - preserves quality, just de-duplicates resources.
    threads = _resolve_gs_threads(num_threads)
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/default",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-dNumRenderingThreads={threads}",
        # Prevent any image downsampling or re-encoding.
        "-dDownsampleColorImages=false",
        "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dPassThroughJPEGImages=true",
        "-dPassThroughJPXImages=true",
        # Remove duplicates and subset fonts.
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        f"-sOutputFile={output_path}",
        str(input_path),
    ]

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = max(300, int(file_mb * 5))

        logger.info(f"{label} {input_path.name} ({file_mb:.1f}MB)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100 if file_mb > 0 else 0.0

        logger.info(f"Optimized: {file_mb:.1f}MB -> {out_mb:.1f}MB ({reduction:.1f}% reduction)")

        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as e:
        return False, str(e)


def compress_pdf_lossless(
    input_path: Path,
    output_path: Path,
    num_threads: Optional[int] = None,
) -> Tuple[bool, str]:
    """Lossless PDF optimization using Ghostscript (no downsampling)."""
    return _run_lossless_ghostscript(
        input_path,
        output_path,
        "Lossless optimize",
        num_threads=num_threads,
    )


def get_ghostscript_command() -> Optional[str]:
    """Get Ghostscript binary name for current platform."""
    for name in ["gs", "gswin64c", "gswin32c"]:
        if shutil.which(name):
            return name
    return None


def translate_ghostscript_error(stderr: str, return_code: int) -> str:
    """Translate Ghostscript stderr to a clear, user-friendly error message.

    Also logs the full stderr for debugging purposes.

    Args:
        stderr: Full stderr output from Ghostscript.
        return_code: Process exit code.

    Returns:
        User-friendly error message explaining what went wrong.
    """
    # Log full stderr for debugging (don't truncate!)
    logger.error(f"Ghostscript failed (exit code {return_code}). Full error:\n{stderr}")

    stderr_lower = stderr.lower()

    # Check for encryption/permission errors
    if 'invalidfileaccess' in stderr_lower or 'password' in stderr_lower:
        return "PDF is password-protected or locked. Please remove the password and try again."

    # Check for metadata/type errors
    if 'typecheck' in stderr_lower or 'rangecheck' in stderr_lower:
        return "PDF has corrupted internal data. Try re-saving it from Adobe Acrobat."

    # Check for structure errors
    if any(x in stderr_lower for x in ['undefined', 'ioerror', 'syntaxerror', 'eofread']):
        return "PDF is damaged or corrupted. Please use a different copy of the file."

    # Generic fallback with exit code
    return f"PDF processing failed (Ghostscript exit code {return_code}). The file may be corrupted."


def compress_pdf_with_ghostscript(
    input_path: Path,
    output_path: Path,
    num_threads: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Compress scanned PDF using Ghostscript at 72 DPI with JPEG encoding.

    Uses aggressive compression (72 DPI) which provides significant file size
    reduction while maintaining readable quality for legal documents.
    Converts JPEG2000 to JPEG for maximum compatibility.

    Args:
        input_path: Source PDF
        output_path: Destination for compressed PDF

    Returns:
        (success, message) tuple where message contains reduction % or error
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 72 DPI + DCTEncode: Best compression for scanned legal documents
    # Higher DPI (150, 200) often makes JPEG2000 files BIGGER
    threads = _resolve_gs_threads(num_threads)
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/screen",  # 72 DPI preset - best compression
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        # Force JPEG encoding (converts JPEG2000 to JPEG)
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        # Downsample all images to 72 DPI for maximum compression
        "-dDownsampleColorImages=true",
        f"-dColorImageDownsampleType={GS_COLOR_DOWNSAMPLE_TYPE}",
        "-dColorImageResolution=72",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageDownsampleType={GS_GRAY_DOWNSAMPLE_TYPE}",
        "-dGrayImageResolution=72",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",  # Faster for 1-bit images
        "-dMonoImageResolution=150",  # Raised for readable scanned text
        "-dMonoImageDownsampleThreshold=1.0",
        # Optimization
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        # Strip metadata that might confuse email clients
        "-dPrinted=false",
        # Speed optimizations (NO effect on compression ratio)
        f"-dNumRenderingThreads={threads}",
    ]
    if GS_FAST_WEB_VIEW:
        cmd.append("-dFastWebView=true")
    if GS_BAND_HEIGHT > 0:
        cmd.append(f"-dBandHeight={GS_BAND_HEIGHT}")
    if GS_BAND_BUFFER_SPACE_MB > 0:
        cmd.append(f"-dBandBufferSpace={GS_BAND_BUFFER_SPACE_MB * 1024 * 1024}")
    cmd.extend(
        [
            f"-sOutputFile={output_path}",
            str(input_path),
        ]
    )

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = max(600, int(file_mb * 10))  # 10 sec/MB, min 10 minutes

        logger.info(f"Compressing {input_path.name} ({file_mb:.1f}MB)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100

        logger.info(f"Result: {file_mb:.1f}MB -> {out_mb:.1f}MB ({reduction:.1f}% reduction)")

        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as e:
        return False, str(e)


def compress_ultra_aggressive(
    input_path: Path,
    output_path: Path,
    jpeg_quality: int = 50,
    num_threads: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Ultra-aggressive compression for oversized split parts.

    Uses the same 72 DPI pipeline as `compress_pdf_with_ghostscript` but lowers JPEG quality
    to squeeze parts under strict size limits. This will reduce image quality; only use when
    standard compression cannot get a part under the threshold.
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # /screen preset + lower JPEG quality for maximum size reduction.
    threads = _resolve_gs_threads(num_threads)
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/screen",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-dJPEGQ={int(jpeg_quality)}",
        # Force JPEG encoding (converts JPEG2000 to JPEG)
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        # Downsample images (72 DPI baseline)
        "-dDownsampleColorImages=true",
        f"-dColorImageDownsampleType={GS_COLOR_DOWNSAMPLE_TYPE}",
        "-dColorImageResolution=72",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageDownsampleType={GS_GRAY_DOWNSAMPLE_TYPE}",
        "-dGrayImageResolution=72",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        "-dMonoImageResolution=150",
        "-dMonoImageDownsampleThreshold=1.0",
        # Optimization
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        f"-dNumRenderingThreads={threads}",
    ]
    if GS_FAST_WEB_VIEW:
        cmd.append("-dFastWebView=true")
    if GS_BAND_HEIGHT > 0:
        cmd.append(f"-dBandHeight={GS_BAND_HEIGHT}")
    if GS_BAND_BUFFER_SPACE_MB > 0:
        cmd.append(f"-dBandBufferSpace={GS_BAND_BUFFER_SPACE_MB * 1024 * 1024}")
    cmd.extend(
        [
            f"-sOutputFile={output_path}",
            str(input_path),
        ]
    )

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = max(600, int(file_mb * 10))

        logger.info(
            f"[ULTRA] Compressing {input_path.name} ({file_mb:.1f}MB) "
            f"with JPEGQ={jpeg_quality}"
        )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100 if file_mb > 0 else 0.0

        logger.info(f"[ULTRA] Result: {file_mb:.1f}MB -> {out_mb:.1f}MB ({reduction:.1f}% reduction)")

        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as e:
        return False, str(e)


# =============================================================================
# PARALLEL COMPRESSION - Uses multiple CPU cores
# =============================================================================

def compress_parallel(
    input_path: Path,
    working_dir: Path,
    base_name: str,
    split_threshold_mb: float = 25.0,
    split_trigger_mb: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    max_workers: int = 6,
    compression_mode: str = "lossless",
    target_chunk_mb: Optional[float] = None,
    max_chunk_mb: Optional[float] = None,
) -> Dict:
    """
    Parallel compression strategy for large PDFs.

    Flow:
    1. Split input PDF into N chunks by page count (fast, no Ghostscript)
    2. Compress each chunk in parallel using ThreadPoolExecutor
    3. Merge compressed chunks back into one PDF
    4. Final split by 25MB threshold for email

    This is 3-4x faster than serial compression for large files because
    each Ghostscript process uses its own CPU core.

    Args:
        input_path: Path to input PDF.
        working_dir: Directory for temp and output files.
        base_name: Base filename for output.
        split_threshold_mb: Max size per final part (default 25MB).
        split_trigger_mb: Size that triggers splitting (defaults to split_threshold_mb).
        progress_callback: Optional callback(percent, stage, message).
        max_workers: Max parallel compression workers (default 6).
        compression_mode: "lossless" (no downsampling) or "aggressive" (/screen).
        target_chunk_mb: Override target chunk size in MB (defaults to env TARGET_CHUNK_MB).
        max_chunk_mb: Override max chunk size in MB (defaults to env MAX_CHUNK_MB).

    Returns:
        Dict with output_path(s), sizes, reduction_percent, etc.
    """
    # Import here to avoid circular imports
    from split_pdf import split_by_pages, merge_pdfs, split_for_delivery
    from utils import get_file_size_mb

    input_path = Path(input_path)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()

    def report_progress(percent: int, stage: str, message: str):
        if progress_callback:
            progress_callback(percent, stage, message)

    original_size_mb = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size

    compression_mode = (compression_mode or "lossless").lower()
    if compression_mode not in ("lossless", "aggressive"):
        logger.warning(f"[PARALLEL] Unknown compression_mode '{compression_mode}', defaulting to lossless")
        compression_mode = "lossless"

    # Tune chunk sizes for large files to reduce chunk count and merge overhead.
    effective_target_chunk_mb = TARGET_CHUNK_MB if target_chunk_mb is None else float(target_chunk_mb)
    effective_max_chunk_mb = MAX_CHUNK_MB if max_chunk_mb is None else float(max_chunk_mb)

    tuned_target = effective_target_chunk_mb
    tuned_max_chunk = effective_max_chunk_mb
    tuned_max_chunks = max(2, MAX_PARALLEL_CHUNKS)
    tuned_max_pages = MAX_PAGES_PER_CHUNK

    if original_size_mb >= 200 and compression_mode == "aggressive":
        tuned_target = max(tuned_target, 60.0)
        tuned_max_chunk = max(tuned_max_chunk, tuned_target * 1.5)
        tuned_max_chunks = min(tuned_max_chunks, 10)
        tuned_max_pages = max(tuned_max_pages, 400)

    target_chunk_mb = max(MIN_CHUNK_MB, tuned_target)
    max_chunk_mb = max(target_chunk_mb, tuned_max_chunk)
    max_parallel_chunks = max(2, tuned_max_chunks)
    max_pages_per_chunk = max(1, tuned_max_pages)
    effective_split_trigger = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb

    logger.info(
        f"[PARALLEL] Starting parallel compression for {input_path.name} "
        f"({original_size_mb:.1f}MB, mode={compression_mode})"
    )
    logger.info(
        f"[PARALLEL] Chunking target {target_chunk_mb:.1f}MB (max {max_chunk_mb:.1f}MB, cap {max_parallel_chunks}, max pages/chunk {max_pages_per_chunk})"
    )

    # Step 1: Calculate number of chunks targeting the configured size.
    # Do NOT cap by max_workers; extra chunks are processed by the thread pool in batches.
    # This keeps each Ghostscript run shorter on very large files (300MB), reducing timeouts.
    num_chunks_by_size = math.ceil(original_size_mb / target_chunk_mb)

    try:
        from PyPDF2 import PdfReader

        with open(input_path, "rb") as f:
            total_pages = len(PdfReader(f, strict=False).pages)
    except Exception:
        total_pages = None

    num_chunks_by_pages = 1
    if total_pages and max_pages_per_chunk > 0:
        num_chunks_by_pages = math.ceil(total_pages / max_pages_per_chunk)

    if original_size_mb >= 200:
        num_chunks = max(2, min(max_parallel_chunks, num_chunks_by_size))
        logger.info(f"[PARALLEL] Large file mode: size-based chunks={num_chunks}")
    else:
        num_chunks = max(2, min(max_parallel_chunks, max(num_chunks_by_size, num_chunks_by_pages)))
    if total_pages:
        logger.info(
            f"[PARALLEL] Total pages: {total_pages}, max pages/chunk: {max_pages_per_chunk}, "
            f"page-based chunks: {num_chunks_by_pages}"
        )
    report_progress(
        5,
        "splitting",
        f"Splitting into {num_chunks} chunks (~{target_chunk_mb:.0f}MB target each) for parallel compression...",
    )

    # Step 2: Split by page count (fast, no Ghostscript)
    chunk_paths = split_by_pages(input_path, working_dir, num_chunks, base_name)
    def rebalance_chunks_by_size(chunks: List[Path]) -> List[Path]:
        """Split oversized chunks again to avoid long-running Ghostscript jobs."""
        from PyPDF2 import PdfReader

        balanced: List[Path] = []
        pending: List[Path] = list(chunks)

        while pending:
            chunk_path = pending.pop(0)
            size_mb = get_file_size_mb(chunk_path)
            if size_mb <= max_chunk_mb or size_mb <= MIN_CHUNK_MB:
                balanced.append(chunk_path)
                continue

            remaining_slots = max_parallel_chunks - (len(balanced) + len(pending))
            if remaining_slots <= 0:
                balanced.append(chunk_path)
                continue

            split_count = max(2, math.ceil(size_mb / target_chunk_mb))
            try:
                with open(chunk_path, "rb") as f:
                    total_pages = len(PdfReader(f, strict=False).pages)
            except Exception:
                total_pages = None

            if not total_pages:
                logger.warning(f"[PARALLEL] Could not read pages for {chunk_path.name}; keeping chunk as-is")
                balanced.append(chunk_path)
                continue

            split_count = min(split_count, total_pages)
            split_count = min(split_count, remaining_slots + 1)
            # Avoid creating sub-chunks below our minimum size target.
            max_splits_by_size = max(2, math.floor(size_mb / MIN_CHUNK_MB))
            split_count = min(split_count, max_splits_by_size)

            if split_count <= 1:
                balanced.append(chunk_path)
                continue

            logger.info(
                f"[PARALLEL] Chunk {chunk_path.name} is {size_mb:.1f}MB; re-splitting into {split_count} parts"
            )
            sub_chunks = split_by_pages(chunk_path, working_dir, split_count, chunk_path.stem)
            chunk_path.unlink(missing_ok=True)
            pending = sub_chunks + pending

        return balanced

    chunk_paths = rebalance_chunks_by_size(chunk_paths)
    num_chunks = len(chunk_paths)
    logger.info(f"[PARALLEL] Split into {num_chunks} chunk(s)")
    split_time = time.time() - start_ts
    logger.info("[PERF] Split: %.1fs", split_time)
    try:
        chunk_sizes = [get_file_size_mb(p) for p in chunk_paths]
        logger.info(
            "[PARALLEL] Chunk stats: count=%s min=%.1fMB max=%.1fMB avg=%.1fMB target=%.1fMB max=%.1fMB",
            num_chunks,
            min(chunk_sizes) if chunk_sizes else 0,
            max(chunk_sizes) if chunk_sizes else 0,
            sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            target_chunk_mb,
            max_chunk_mb,
        )
    except Exception:
        pass

    # Step 3: Compress each chunk in parallel
    report_progress(15, "compressing", f"Compressing {num_chunks} chunks in parallel...")

    compress_fn = compress_pdf_lossless if compression_mode == "lossless" else compress_pdf_with_ghostscript

    def compress_single_chunk(chunk_path: Path) -> Tuple[Path, bool, str]:
        """Compress a single chunk and return result."""
        unique_id = str(uuid.uuid4())[:8]
        compressed_path = working_dir / f"{chunk_path.stem}_{unique_id}_compressed.pdf"

        # Skip GS if chunk is below minimum size; likely not worth recompressing.
        try:
            if get_file_size_mb(chunk_path) <= MIN_CHUNK_MB:
                return chunk_path, True, "Skipped compression (below min size)"
        except Exception:
            pass

        success, message = compress_fn(chunk_path, compressed_path, num_threads=gs_threads)

        # If Ghostscript "compression" makes this chunk larger, keep the original chunk.
        # This is common for vector/text-heavy PDFs where re-encoding can bloat output.
        try:
            if success and compressed_path.exists():
                original_bytes = chunk_path.stat().st_size
                compressed_bytes = compressed_path.stat().st_size
                if compressed_bytes >= original_bytes:
                    compressed_path.unlink(missing_ok=True)
                    original_mb = original_bytes / (1024 * 1024)
                    compressed_mb = compressed_bytes / (1024 * 1024)
                    return (
                        chunk_path,
                        False,
                        f"Compression increased size ({original_mb:.1f}MB -> {compressed_mb:.1f}MB), using original",
                    )
        except Exception:
            # If stats fail for any reason, fall back to the normal success path below.
            pass

        return compressed_path, success, message

    compressed_chunks = []
    failed_chunks = []

    worker_count = min(max_workers, len(chunk_paths)) if chunk_paths else max_workers
    effective_cpu = get_effective_cpu_count()
    per_worker_cap = max(1, effective_cpu // max(worker_count, 1))

    env_threads = os.environ.get("GS_NUM_RENDERING_THREADS")
    if env_threads:
        try:
            desired_threads = max(1, int(env_threads))
        except ValueError:
            desired_threads = per_worker_cap
        if desired_threads > per_worker_cap:
            logger.info(
                "[PARALLEL] GS_NUM_RENDERING_THREADS=%s capped to %s based on cpu/workers",
                desired_threads,
                per_worker_cap,
            )
            gs_threads = per_worker_cap
        else:
            gs_threads = desired_threads
    else:
        gs_threads = max(1, min(4, per_worker_cap))

    logger.info(
        f"[PARALLEL] Workers=%s GS threads/worker=%s (cpu=%s, max_workers=%s)",
        worker_count,
        gs_threads,
        effective_cpu,
        max_workers,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Submit all compression jobs
        futures = {
            executor.submit(compress_single_chunk, chunk): (i, chunk)
            for i, chunk in enumerate(chunk_paths)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            chunk_idx, original_chunk = futures[future]
            try:
                compressed_path, success, message = future.result()
                if success and compressed_path.exists():
                    compressed_chunks.append((chunk_idx, compressed_path))
                    logger.info(f"[PARALLEL] Chunk {chunk_idx + 1} compressed successfully")
                else:
                    # Compression failed - use original chunk
                    logger.warning(f"[PARALLEL] Chunk {chunk_idx + 1} compression failed: {message}")
                    failed_chunks.append((chunk_idx, original_chunk))
            except Exception as e:
                logger.error(f"[PARALLEL] Chunk {chunk_idx + 1} error: {e}")
                failed_chunks.append((chunk_idx, original_chunk))

            # Update progress
            done = len(compressed_chunks) + len(failed_chunks)
            pct = 15 + int(55 * done / num_chunks)
            report_progress(pct, "compressing", f"Compressed {done}/{num_chunks} chunks...")

    # Sort chunks by original order and combine with failed chunks
    all_chunks = compressed_chunks + failed_chunks
    all_chunks.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in all_chunks]

    logger.info(f"[PARALLEL] Compression complete: {len(compressed_chunks)} succeeded, {len(failed_chunks)} used original")
    compress_time = time.time() - start_ts - split_time
    logger.info("[PERF] Compress: %.1fs", compress_time)

    # Step 4: If all chunks are already under split threshold, skip merge and return them.
    # Otherwise, split each compressed chunk directly to <= threshold and return parts in order.
    all_under_threshold = all(get_file_size_mb(p) <= split_threshold_mb for p in ordered_paths)

    split_start = time.time()
    if all_under_threshold:
        final_parts = []
        for idx, chunk_path in enumerate(ordered_paths, start=1):
            dest = working_dir / f"{chunk_path.stem}_part{idx}.pdf"
            dest.unlink(missing_ok=True)
            chunk_path.rename(dest)
            final_parts.append(dest)
        logger.info(f"[PARALLEL] All chunks <= {split_threshold_mb}MB; skipping merge and returning chunks")
    else:
        report_progress(75, "splitting", f"Splitting compressed chunks into parts <= {split_threshold_mb}MB")
        final_parts: List[Path] = []
        for idx, chunk_path in enumerate(ordered_paths):
            chunk_size = get_file_size_mb(chunk_path)
            if chunk_size <= split_threshold_mb:
                dest = working_dir / f"{chunk_path.stem}_part1.pdf"
                dest.unlink(missing_ok=True)
                chunk_path.rename(dest)
                final_parts.append(dest)
                continue

            # Split oversized chunk by size
            chunk_parts = split_for_delivery(
                chunk_path,
                working_dir,
                f"{chunk_path.stem}",
                split_threshold_mb,
                progress_callback=progress_callback,
                skip_optimization_under_threshold=True,
            )
            final_parts.extend(chunk_parts)

        logger.info(
            "[PARALLEL] Split compressed chunks into %s part(s) without global merge (threshold=%.1fMB, chunks=%s)",
            len(final_parts),
            split_threshold_mb,
            len(ordered_paths),
        )

    # Cleanup original chunk files
    for chunk in chunk_paths:
        chunk.unlink(missing_ok=True)
    for _, compressed in compressed_chunks:
        compressed.unlink(missing_ok=True)

    split_total = time.time() - split_start
    logger.info("[PERF] Direct split time: %.1fs", split_total)
    report_progress(100, "complete", f"Complete: {len(final_parts)} parts")

    # Verify parts exist and sizes
    verified_parts: List[Path] = []
    for i, part in enumerate(final_parts):
        if not part.exists() or part.stat().st_size == 0:
            raise SplitError(f"Part {i+1} missing or empty after split: {part}")
        verified_parts.append(part)
        logger.info(f"[PARALLEL] Part {i+1} verified: {part.name} ({part.stat().st_size / (1024*1024):.1f}MB)")

    # Page count from original (if available)
    page_count = total_pages
    if page_count is None:
        from PyPDF2 import PdfReader
        try:
            with open(input_path, 'rb') as f:
                page_count = len(PdfReader(f, strict=False).pages)
        except Exception:
            page_count = None

    # Final size and reduction
    combined_bytes = sum(p.stat().st_size for p in verified_parts)
    combined_mb = combined_bytes / (1024 * 1024)
    reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100

    return {
        "output_path": str(verified_parts[0]),
        "output_paths": [str(p) for p in verified_parts],
        "original_size_mb": round(original_size_mb, 2),
        "compressed_size_mb": round(combined_mb, 2),
        "reduction_percent": round(reduction_percent, 1),
        "compression_method": "ghostscript_parallel",
        "compression_mode": compression_mode,
        "was_split": len(verified_parts) > 1,
        "total_parts": len(verified_parts),
        "success": True,
        "page_count": page_count,
        "part_sizes": [p.stat().st_size for p in verified_parts],
        "parallel_chunks": num_chunks
    }
