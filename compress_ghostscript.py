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
from utils import get_effective_cpu_count, env_bool, env_int, env_choice

logger = logging.getLogger(__name__)

TARGET_CHUNK_MB = float(os.environ.get("TARGET_CHUNK_MB", "30"))
MAX_CHUNK_MB = float(os.environ.get("MAX_CHUNK_MB", "50"))
MAX_PARALLEL_CHUNKS = int(os.environ.get("MAX_PARALLEL_CHUNKS", "16"))
MAX_PAGES_PER_CHUNK = int(os.environ.get("MAX_PAGES_PER_CHUNK", "200"))
# Minimum chunk size to avoid spawning tiny Ghostscript jobs that add overhead.
# Keep hardcoded (no env tuning) for predictable behavior.
MIN_CHUNK_MB = 5.0
# Large-file tuning defaults (used only when chunk overrides are not set).
LARGE_FILE_TUNE_MIN_MB = float(os.environ.get("LARGE_FILE_TUNE_MIN_MB", "200"))
LARGE_FILE_TARGET_CHUNK_MB = float(os.environ.get("LARGE_FILE_TARGET_CHUNK_MB", "50"))
LARGE_FILE_MAX_CHUNK_MB = float(os.environ.get("LARGE_FILE_MAX_CHUNK_MB", "75"))
LARGE_FILE_MAX_PARALLEL_CHUNKS = int(os.environ.get("LARGE_FILE_MAX_PARALLEL_CHUNKS", "12"))
# Guardrail: only trigger merge fallback when output is meaningfully larger than input.
PARALLEL_BLOAT_PCT = 0.08
PARALLEL_MERGE_ON_BLOAT = env_bool("PARALLEL_MERGE_ON_BLOAT", True)
# Log-only signal when chunking inflates total size (PyPDF2 duplicates resources).
PARALLEL_SPLIT_INFLATION_PCT = 0.02
# If split inflation is high, force dedupe on split parts (targeted slow path).
# Lower default threshold to catch modest PyPDF2 bloat before it cascades.
PARALLEL_DEDUP_SPLIT_INFLATION_PCT = float(os.environ.get("PARALLEL_DEDUP_SPLIT_INFLATION_PCT", "0.08"))


REBALANCE_SPLIT_ENABLED = env_bool("REBALANCE_SPLIT_ENABLED", True)
REBALANCE_MIN_PARTS_OVER_MIN = env_int("REBALANCE_MIN_PARTS_OVER_MIN", 2)
REBALANCE_SPLIT_INFLATION_PCT = float(os.environ.get("REBALANCE_SPLIT_INFLATION_PCT", "40"))
REBALANCE_MAX_PART_PCT = float(os.environ.get("REBALANCE_MAX_PART_PCT", "0.85"))

GS_FAST_WEB_VIEW = env_bool("GS_FAST_WEB_VIEW", True)
GS_COLOR_DOWNSAMPLE_TYPE = env_choice(
    "GS_COLOR_DOWNSAMPLE_TYPE",
    "/Bicubic",
    ("/Subsample", "/Average", "/Bicubic"),
)
GS_GRAY_DOWNSAMPLE_TYPE = env_choice(
    "GS_GRAY_DOWNSAMPLE_TYPE",
    "/Bicubic",
    ("/Subsample", "/Average", "/Bicubic"),
)
GS_BAND_HEIGHT = env_int("GS_BAND_HEIGHT", 100)
GS_BAND_BUFFER_SPACE_MB = env_int("GS_BAND_BUFFER_SPACE_MB", 500)
GS_COLOR_IMAGE_RESOLUTION = env_int("GS_COLOR_IMAGE_RESOLUTION", 72)
GS_GRAY_IMAGE_RESOLUTION = env_int("GS_GRAY_IMAGE_RESOLUTION", 72)
GS_MONO_IMAGE_RESOLUTION = env_int("GS_MONO_IMAGE_RESOLUTION", 300)
PARALLEL_SERIAL_FALLBACK = env_bool("PARALLEL_SERIAL_FALLBACK", True)
# Fixed SLA/quality guardrails (non-configurable)
CHUNK_TIME_BUDGET_SEC = 60
CHUNK_TIME_BUDGET_MIN_SEC = 40
CHUNK_TIME_BUDGET_MAX_SEC = 90
PROBE_TIME_BUDGET_SEC = 45          # seconds
PROBE_INFLATION_ABORT_PCT = 0.20    # 20% growth triggers bailout
PROBE_SAMPLE_PAGES = 3
PARALLEL_JOB_SLA_SEC = 300          # hard cap for parallel job
SLA_MAX_PARALLEL_CHUNKS = max(2, env_int("SLA_MAX_PARALLEL_CHUNKS", 8))
SLA_MAX_PARALLEL_CHUNKS_LARGE = max(2, env_int("SLA_MAX_PARALLEL_CHUNKS_LARGE", 12))
HARD_MAX_PARALLEL_CHUNKS = max(2, env_int("HARD_MAX_PARALLEL_CHUNKS", 24))
PARALLEL_SPLIT_DELIVER_CHUNKS = env_bool("PARALLEL_SPLIT_DELIVER_CHUNKS", True)


def _chunk_timeout_seconds(file_mb: float) -> int:
    """
    Compute a sane per-chunk timeout so Ghostscript jobs don't run indefinitely.
    Scales with size but clamps to a reasonable window to avoid hanging workers.
    """
    dynamic = int(max(1.0, file_mb) * 8)  # ~8s/MB baseline
    budget = max(CHUNK_TIME_BUDGET_MIN_SEC, min(CHUNK_TIME_BUDGET_SEC, CHUNK_TIME_BUDGET_MAX_SEC))
    return min(max(dynamic, budget), CHUNK_TIME_BUDGET_MAX_SEC)


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
    timeout_override: Optional[int] = None,
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
        timeout_override=timeout_override,
    )


def _run_lossless_ghostscript(
    input_path: Path,
    output_path: Path,
    label: str,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
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
        timeout = timeout_override or max(300, int(file_mb * 5))

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
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """Lossless PDF optimization using Ghostscript (no downsampling)."""
    return _run_lossless_ghostscript(
        input_path,
        output_path,
        "Lossless optimize",
        num_threads=num_threads,
        timeout_override=timeout_override,
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
    timeout_override: Optional[int] = None,
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
        f"-dColorImageResolution={GS_COLOR_IMAGE_RESOLUTION}",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageDownsampleType={GS_GRAY_DOWNSAMPLE_TYPE}",
        f"-dGrayImageResolution={GS_GRAY_IMAGE_RESOLUTION}",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",  # Faster for 1-bit images
        f"-dMonoImageResolution={GS_MONO_IMAGE_RESOLUTION}",  # Raised for readable scanned text
        "-dMonoImageDownsampleThreshold=1.0",
        # Optimization
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        # Strip metadata for cleaner output
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
        timeout = timeout_override or _chunk_timeout_seconds(file_mb)

        logger.info(
            "[DIAG] GS aggressive cmd: dpi=%s/%s/%s downsample=%s/%s "
            "threads=%s timeout=%ss fast_web=%s band_height=%s band_buffer_mb=%s",
            GS_COLOR_IMAGE_RESOLUTION, GS_GRAY_IMAGE_RESOLUTION, GS_MONO_IMAGE_RESOLUTION,
            GS_COLOR_DOWNSAMPLE_TYPE, GS_GRAY_DOWNSAMPLE_TYPE,
            threads, timeout,
            GS_FAST_WEB_VIEW, GS_BAND_HEIGHT, GS_BAND_BUFFER_SPACE_MB,
        )
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
    timeout_override: Optional[int] = None,
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
        f"-dColorImageResolution={GS_COLOR_IMAGE_RESOLUTION}",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageDownsampleType={GS_GRAY_DOWNSAMPLE_TYPE}",
        f"-dGrayImageResolution={GS_GRAY_IMAGE_RESOLUTION}",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        f"-dMonoImageResolution={GS_MONO_IMAGE_RESOLUTION}",
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
        timeout = timeout_override or _chunk_timeout_seconds(file_mb)

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


def compress_ultra_fallback(
    input_path: Path,
    output_path: Path,
    jpeg_quality: int = 55,
    color_res: int = 120,
    gray_res: int = 150,
    mono_res: int = 300,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Ultra fallback for stubborn files: lower DPI + JPEGQ to force size reduction.
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    threads = _resolve_gs_threads(num_threads)
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/screen",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-dJPEGQ={int(max(10, min(jpeg_quality, 95)))}",
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        "-dDownsampleColorImages=true",
        f"-dColorImageDownsampleType={GS_COLOR_DOWNSAMPLE_TYPE}",
        f"-dColorImageResolution={max(72, color_res)}",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageDownsampleType={GS_GRAY_DOWNSAMPLE_TYPE}",
        f"-dGrayImageResolution={max(72, gray_res)}",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        f"-dMonoImageResolution={max(150, mono_res)}",
        "-dMonoImageDownsampleThreshold=1.0",
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
        timeout = timeout_override or _chunk_timeout_seconds(file_mb)

        logger.info(
            "[ULTRA_FALLBACK] Compressing %s (%.1fMB) JPEGQ=%s DPI=%s/%s/%s",
            input_path.name,
            file_mb,
            jpeg_quality,
            color_res,
            gray_res,
            mono_res,
        )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100 if file_mb > 0 else 0.0

        logger.info(
            "[ULTRA_FALLBACK] Result: %.1fMB -> %.1fMB (%.1f%% reduction)",
            file_mb,
            out_mb,
            reduction,
        )

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
    split_threshold_mb: Optional[float] = None,
    split_trigger_mb: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    max_workers: int = 6,
    compression_mode: str = "lossless",
    target_chunk_mb: Optional[float] = None,
    max_chunk_mb: Optional[float] = None,
    input_page_count: Optional[int] = None,
    allow_lossy: bool = False,
) -> Dict:
    """
    Parallel compression strategy for large PDFs.

    Flow:
    1. Split input PDF into N chunks by page count (fast, no Ghostscript)
    2. Compress each chunk in parallel using ThreadPoolExecutor
    3. Merge compressed chunks back into one PDF
    4. Final split by threshold (if requested)

    This is 3-4x faster than serial compression for large files because
    each Ghostscript process uses its own CPU core.

    Args:
        input_path: Path to input PDF.
        working_dir: Directory for temp and output files.
        base_name: Base filename for output.
        split_threshold_mb: Max size per final part (None disables splitting).
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
    from split_pdf import split_by_pages, merge_pdfs, split_for_delivery, split_by_size, split_pdf
    from utils import get_file_size_mb

    input_path = Path(input_path)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()
    sla_deadline = start_ts + PARALLEL_JOB_SLA_SEC
    worst_inflation_pct = 0.0
    strategy_outcome = {
        "mode": "aggressive" if compression_mode == "aggressive" else "lossless",
        "forced_by": None,  # page_count | size | timeout | manual
        "probe_status": None,  # success | fail | bailout
        "merge_path": None,    # gs | pypdf2 | skipped
        "timeouts": {
            "chunk_timeout": 0,
            "probe_timeout": 0,
            "merge_timeout": 0,
            "job_sla_hit": False,
        },
        "quality_overrides": {
            "dedupe_forced": False,
            "dpi_floor": True,
        },
        "max_chunk_inflation_pct": 0.0,
        "max_merge_inflation_pct": 0.0,
    }

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
    if compression_mode == "aggressive" and split_threshold_mb is None:
        sla_cap = SLA_MAX_PARALLEL_CHUNKS
        if original_size_mb >= LARGE_FILE_TUNE_MIN_MB:
            sla_cap = max(sla_cap, SLA_MAX_PARALLEL_CHUNKS_LARGE)
        tuned_max_chunks = max(2, min(tuned_max_chunks, sla_cap))
    tuned_max_pages = MAX_PAGES_PER_CHUNK

    has_chunk_env = "TARGET_CHUNK_MB" in os.environ or "MAX_CHUNK_MB" in os.environ
    explicit_chunk_override = target_chunk_mb is not None or max_chunk_mb is not None
    if original_size_mb >= LARGE_FILE_TUNE_MIN_MB and compression_mode == "aggressive":
        if not has_chunk_env and not explicit_chunk_override:
            tuned_target = max(tuned_target, LARGE_FILE_TARGET_CHUNK_MB)
            tuned_max_chunk = max(tuned_max_chunk, max(LARGE_FILE_MAX_CHUNK_MB, tuned_target * 1.5))
            tuned_max_chunks = min(tuned_max_chunks, LARGE_FILE_MAX_PARALLEL_CHUNKS)
            tuned_max_pages = max(tuned_max_pages, 400)
            logger.info(
                "[PARALLEL] Large file tuning (workers=%s): target %.1fMB, max %.1fMB, max pages/chunk %s",
                max_workers,
                tuned_target,
                tuned_max_chunk,
                tuned_max_pages,
            )
        else:
            logger.info(
                "[PARALLEL] Large file tuning skipped; explicit chunk override present",
            )

    target_chunk_mb = max(MIN_CHUNK_MB, tuned_target)
    max_chunk_mb = max(target_chunk_mb, tuned_max_chunk)
    max_parallel_chunks = max(2, tuned_max_chunks)
    max_pages_per_chunk = max(1, tuned_max_pages)
    hard_max_parallel_chunks = max(max_parallel_chunks, HARD_MAX_PARALLEL_CHUNKS)
    logger.info(
        f"[PARALLEL] Starting parallel compression for {input_path.name} "
        f"({original_size_mb:.1f}MB, mode={compression_mode})"
    )
    logger.info(
        f"[PARALLEL] Chunking target {target_chunk_mb:.1f}MB (max {max_chunk_mb:.1f}MB, cap {max_parallel_chunks}, max pages/chunk {max_pages_per_chunk})"
    )
    logger.info(
        "[DIAG] === PARALLEL JOB START === file=%s size_mb=%.2f mode=%s "
        "allow_lossy=%s workers=%s target_chunk=%.1f max_chunk=%.1f",
        input_path.name, original_size_mb, compression_mode,
        allow_lossy, max_workers, target_chunk_mb, max_chunk_mb,
    )

    # Step 1: Calculate number of chunks targeting the configured size.
    # Do NOT cap by max_workers; extra chunks are processed by the thread pool in batches.
    # This keeps each Ghostscript run shorter on very large files (300MB), reducing timeouts.
    num_chunks_by_size = math.ceil(original_size_mb / target_chunk_mb)

    total_pages = input_page_count
    if total_pages is None:
        try:
            from PyPDF2 import PdfReader

            with open(input_path, "rb") as f:
                total_pages = len(PdfReader(f, strict=False).pages)
        except Exception:
            total_pages = None

    num_chunks_by_pages = 1
    if total_pages and max_pages_per_chunk > 0:
        num_chunks_by_pages = math.ceil(total_pages / max_pages_per_chunk)
        # When no split is requested, cap page-driven chunking to the parallel limit
        # but do NOT collapse it to just a few parts; very high page counts need
        # smaller chunks to keep Ghostscript runs within timeouts.
        if split_threshold_mb is None:
            num_chunks_by_pages = max(1, min(num_chunks_by_pages, max_parallel_chunks))

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

            soft_remaining = max_parallel_chunks - (len(balanced) + len(pending))
            hard_remaining = hard_max_parallel_chunks - (len(balanced) + len(pending))
            if hard_remaining <= 0:
                balanced.append(chunk_path)
                continue

            oversized = size_mb > max_chunk_mb and size_mb > MIN_CHUNK_MB
            remaining_slots = hard_remaining if oversized else soft_remaining
            if remaining_slots <= 0:
                balanced.append(chunk_path)
                continue

            if oversized and soft_remaining <= 0:
                logger.info(
                    "[PARALLEL] Soft cap %s hit; allowing extra split up to hard cap %s for %s (%.1fMB)",
                    max_parallel_chunks,
                    hard_max_parallel_chunks,
                    chunk_path.name,
                    size_mb,
                )

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
    split_inflation = False
    split_inflation_ratio = 1.0
    force_split_dedup = False
    chunk_total_mb = 0.0
    try:
        chunk_sizes = [get_file_size_mb(p) for p in chunk_paths]
        chunk_total_mb = sum(chunk_sizes)
        logger.info(
            "[PARALLEL] Chunk stats: count=%s min=%.1fMB max=%.1fMB avg=%.1fMB target=%.1fMB max=%.1fMB",
            num_chunks,
            min(chunk_sizes) if chunk_sizes else 0,
            max(chunk_sizes) if chunk_sizes else 0,
            sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            target_chunk_mb,
            max_chunk_mb,
        )
        if original_size_mb > 0:
            inflation_ratio = chunk_total_mb / original_size_mb
            split_inflation_ratio = inflation_ratio
            if inflation_ratio > (1 + PARALLEL_SPLIT_INFLATION_PCT):
                split_inflation = True
                inflation_pct = (inflation_ratio - 1) * 100
                logger.warning(
                    "[PARALLEL] Split inflation detected: %.1fMB -> %.1fMB chunks (+%.1f%%)",
                    original_size_mb,
                    chunk_total_mb,
                    inflation_pct,
                )
    except Exception:
        pass
    if split_inflation_ratio > (1 + PARALLEL_DEDUP_SPLIT_INFLATION_PCT):
        force_split_dedup = True
        strategy_outcome["quality_overrides"]["dedupe_forced"] = True
        logger.warning(
            "[PARALLEL] High split inflation %.1f%% (threshold %.1f%%); forcing dedupe on split parts (no skip)",
            (split_inflation_ratio - 1) * 100,
            PARALLEL_DEDUP_SPLIT_INFLATION_PCT * 100,
        )

    # Step 3: Compress each chunk in parallel

    compress_fn = compress_pdf_lossless if compression_mode == "lossless" else compress_pdf_with_ghostscript
    if compression_mode != "lossless" and not force_split_dedup:
        logger.info(
            "[PARALLEL] Aggressive mode: skipping pre-dedupe on chunks unless split inflation forces it"
        )

    def compress_single_chunk(
        chunk_path: Path,
        force_dedup: bool = False,
        timeout_override: Optional[int] = None,
    ) -> Tuple[Path, bool, str, float]:
        """Compress a single chunk and return result."""
        unique_id = str(uuid.uuid4())[:8]
        compressed_path = working_dir / f"{chunk_path.stem}_{unique_id}_compressed.pdf"
        chunk_mb = get_file_size_mb(chunk_path)
        chunk_timeout = timeout_override or _chunk_timeout_seconds(chunk_mb)

        should_dedupe = compression_mode == "lossless" or force_dedup
        if should_dedupe:
            # Attempt dedupe first (fast, lossless), then fall through to full compression.
            dedup_path = working_dir / f"{chunk_path.stem}_{unique_id}_dedup.pdf"
            ok, msg = optimize_split_part(
                chunk_path,
                dedup_path,
                num_threads=gs_threads,
                timeout_override=chunk_timeout,
            )
            if ok and dedup_path.exists():
                try:
                    dedup_mb = get_file_size_mb(dedup_path)
                    orig_mb = get_file_size_mb(chunk_path)
                    if dedup_mb <= orig_mb:
                        return dedup_path, True, f"DEDUP: {orig_mb:.1f}MB -> {dedup_mb:.1f}MB", chunk_mb
                    dedup_path.unlink(missing_ok=True)
                except Exception:
                    dedup_path.unlink(missing_ok=True)

        success, message = compress_fn(
            chunk_path,
            compressed_path,
            num_threads=gs_threads,
            timeout_override=chunk_timeout,
        )

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
                        chunk_mb,
                    )
        except Exception:
            # If stats fail for any reason, fall back to the normal success path below.
            pass

        return compressed_path, success, message, chunk_mb

    # Default threads; may be adjusted after load evaluation.
    gs_threads = max(1, min(4, get_effective_cpu_count()))

    compressed_chunks = []
    failed_chunks = []
    skipped_chunks = 0

    # ------------------------------------------------------------------
    # Probe: run the smallest chunk first to measure throughput, then
    # dynamically retune remaining chunk sizes to hit a time budget.
    # ------------------------------------------------------------------
    probe_pps = None
    probe_mbps = None
    probe_elapsed = None
    probe_reinserted = False
    time_budget = max(
        CHUNK_TIME_BUDGET_MIN_SEC,
        min(CHUNK_TIME_BUDGET_SEC, CHUNK_TIME_BUDGET_MAX_SEC),
    )

    def _read_pages(path: Path) -> Optional[int]:
        try:
            from PyPDF2 import PdfReader

            with open(path, "rb") as f:
                return len(PdfReader(f, strict=False).pages)
        except Exception:
            return None

    if chunk_paths and compression_mode != "lossless":
        # Pick the smallest chunk by size for the probe to keep risk low.
        chunk_sizes = [get_file_size_mb(p) for p in chunk_paths]
        probe_idx = chunk_sizes.index(min(chunk_sizes))
        probe_order = probe_idx
        probe_chunk = chunk_paths.pop(probe_idx)
        logger.info(
            "[PARALLEL_PROBE] Running probe on %s (%.1fMB)",
            probe_chunk.name,
            chunk_sizes[probe_idx],
        )
        probe_start = time.time()
        # For the probe we want to avoid the expensive /default dedupe pass in
        # aggressive mode because it frequently inflates the chunk and trips
        # the bailout logic. In lossless mode we still allow dedupe.
        probe_force_dedup = force_split_dedup if compression_mode == "lossless" else False
        probe_result = compress_single_chunk(
            probe_chunk,
            probe_force_dedup,
            timeout_override=PROBE_TIME_BUDGET_SEC,
        )
        probe_elapsed = time.time() - probe_start
        try:
            probe_path, success, message, probe_mb = probe_result
        except Exception as exc:  # unpacking safety
            logger.error("[PARALLEL_PROBE] failed: %s", exc)
            success = False
            message = str(exc)
            probe_path = probe_chunk
            probe_mb = get_file_size_mb(probe_chunk)

        if success and probe_path.exists():
            compressed_chunks.append((probe_order, probe_path))
            probe_pages = _read_pages(probe_chunk)
            if probe_elapsed > 0:
                probe_mbps = probe_mb / probe_elapsed
                if probe_pages:
                    probe_pps = probe_pages / probe_elapsed
            logger.info(
                "[PARALLEL_PROBE] success size=%.1fMB pages=%s elapsed=%.1fs mbps=%.2f pps=%s msg=%s",
                probe_mb,
                probe_pages if probe_pages is not None else "unknown",
                probe_elapsed,
                probe_mbps or -1,
                (f"{probe_pps:.2f}" if probe_pps else "n/a"),
                message,
            )
            strategy_outcome["probe_status"] = "success"
        else:
            logger.warning("[PARALLEL_PROBE] failed: %s", message)
            failed_chunks.append((probe_order, probe_chunk))
            strategy_outcome["probe_status"] = "fail"

        # Early exit if probe inflates or is slow; avoid spending minutes on bad strategy.
        probe_out_mb = get_file_size_mb(probe_path) if probe_path.exists() else probe_mb
        probe_inflated = success and probe_out_mb > probe_mb * (1 + PROBE_INFLATION_ABORT_PCT)
        probe_slow = probe_elapsed is not None and probe_elapsed > PROBE_TIME_BUDGET_SEC
        if probe_inflated:
            bailout_reason = []
            if probe_inflated:
                bailout_reason.append(
                    f"inflated {probe_mb:.1f}->{probe_out_mb:.1f}MB (+{(probe_out_mb/probe_mb-1)*100:.1f}%)"
                )
            if probe_slow:
                bailout_reason.append(f"slow {probe_elapsed:.1f}s>{PROBE_TIME_BUDGET_SEC}s")
            logger.warning(
                "[PARALLEL_PROBE] Bailout triggered (%s); returning original to preserve SLA/quality",
                ", ".join(bailout_reason),
            )
            merged_path = working_dir / f"{base_name}_compressed.pdf"
            # Always attempt a quick lossless full-file pass before giving up.
            lossless_path = working_dir / f"{base_name}_lossless_probe.pdf"
            lossless_used = False
            ok_lossless = False
            msg_lossless = ""
            # Cap the lossless attempt to stay within the 1–5 minute SLA.
            lossless_timeout = min(240, max(90, int(original_size_mb * 1.2)))
            try:
                ok_lossless, msg_lossless = compress_pdf_lossless(
                    input_path,
                    lossless_path,
                    timeout_override=lossless_timeout,
                )
            except Exception as exc:
                msg_lossless = str(exc)

            if ok_lossless and lossless_path.exists():
                lossless_mb = get_file_size_mb(lossless_path)
                if lossless_mb < original_size_mb:
                    lossless_used = True
                    logger.info(
                        "[PARALLEL_PROBE] Lossless fallback succeeded: %.1fMB -> %.1fMB (timeout=%ss)",
                        original_size_mb,
                        lossless_mb,
                        lossless_timeout,
                    )
                    original_bytes = lossless_path.stat().st_size
                    original_size_mb = lossless_mb
                    merged_path.unlink(missing_ok=True)
                    shutil.copy2(lossless_path, merged_path)
                lossless_path.unlink(missing_ok=True)
            else:
                if not ok_lossless and "timeout" in (msg_lossless or "").lower():
                    strategy_outcome["timeouts"]["probe_timeout"] += 1

            # Cleanup temp chunks
            for p in chunk_paths:
                p.unlink(missing_ok=True)
            if "probe_path" in locals() and probe_path.exists() and probe_path != probe_chunk:
                probe_path.unlink(missing_ok=True)
            if probe_chunk.exists():
                probe_chunk.unlink(missing_ok=True)
            # Materialize output path
            merged_path = working_dir / f"{base_name}_compressed.pdf"
            merged_path.unlink(missing_ok=True)
            shutil.copy2(input_path, merged_path)
            strategy_outcome["probe_status"] = "bailout"
            # If lossless fallback produced a smaller file, return it; otherwise return original.
            compressed_mb = get_file_size_mb(merged_path) if merged_path.exists() else original_size_mb
            reduction_pct = 0.0
            if lossless_used and compressed_mb < original_size_mb and original_size_mb > 0:
                reduction_pct = ((original_size_mb - compressed_mb) / original_size_mb) * 100
            return {
                "output_path": str(merged_path),
                "output_paths": [str(merged_path)],
                "original_size_mb": round(original_size_mb, 2),
                "compressed_size_mb": round(compressed_mb, 2),
                "reduction_percent": round(reduction_pct, 1),
                "compression_method": "lossless" if lossless_used else "none",
                "compression_mode": compression_mode,
                "quality_mode": "lossless",
                "was_split": False,
                "total_parts": 1,
                "success": True,
                "page_count": total_pages,
                "part_sizes": [original_bytes],
                "bloat_detected": not lossless_used,
                "bloat_action": "lossless_fallback" if lossless_used else "probe_bailout",
                "probe_bailout": True,
                "probe_bailout_reason": ", ".join(bailout_reason),
                "lossless_fallback_used": lossless_used,
                "note": "Probe showed inflation; applied lossless fallback before returning",
            }
        elif probe_slow and not probe_inflated and not success:
            logger.warning(
                "[PARALLEL_PROBE] Probe timed out (%.1fs > %ss) but no inflation; "
                "continuing with chunk compression (re-inserting probe chunk)",
                probe_elapsed,
                PROBE_TIME_BUDGET_SEC,
            )
            strategy_outcome["probe_status"] = "timeout_continue"
            strategy_outcome["timeouts"]["probe_timeout"] += 1

            failed_chunks = [
                (idx, path) for idx, path in failed_chunks
                if path != probe_chunk
            ]

            insert_pos = min(probe_idx, len(chunk_paths))
            chunk_paths.insert(insert_pos, probe_chunk)

            if probe_path != probe_chunk and probe_path.exists():
                probe_path.unlink(missing_ok=True)

            probe_reinserted = True

        # Retune remaining chunks based on probe throughput.
        if (probe_pps or probe_mbps) and chunk_paths:
            adjusted: List[Path] = []
            for path in chunk_paths:
                size_mb = get_file_size_mb(path)
                pages = _read_pages(path)
                ests = []
                if probe_mbps:
                    ests.append(size_mb / probe_mbps)
                if probe_pps and pages:
                    ests.append(pages / probe_pps)
                est_time = max(ests) if ests else 0
                if est_time > time_budget and pages and pages > 1:
                    split_count = max(2, math.ceil(est_time / time_budget))
                    split_count = min(split_count, pages)
                    logger.info(
                        "[PARALLEL_PROBE] Resplitting %s (%.1fMB, ~%.1fs est) into %s parts",
                        path.name,
                        size_mb,
                        est_time,
                        split_count,
                    )
                    parts = split_by_pages(path, working_dir, split_count, path.stem)
                    path.unlink(missing_ok=True)
                    adjusted.extend(parts)
                else:
                    adjusted.append(path)
            chunk_paths = adjusted

    # Build execution plan with stable order keys (page order).
    tasks: List[Tuple[int, Path]] = []
    plan_list: List[Optional[Path]] = list(chunk_paths)
    if "probe_chunk" in locals() and not probe_reinserted:
        insert_at = min(probe_idx, len(plan_list))
        plan_list.insert(insert_at, None)  # placeholder for probe

    order_idx = 0
    for item in plan_list:
        if item is None:
            if "probe_path" in locals() and success and probe_path.exists():
                compressed_chunks.append((order_idx, probe_path))
            else:
                failed_chunks.append((order_idx, probe_chunk))
        else:
            tasks.append((order_idx, item))
        order_idx += 1

    num_chunks = order_idx
    logger.info("[PARALLEL] Post-probe chunk count: %s (tasks=%s, probe=%s)", num_chunks, len(tasks), "yes" if "probe_chunk" in locals() else "no")

    worker_count = max(1, min(max_workers, len(tasks))) if tasks else 1
    effective_cpu = get_effective_cpu_count()
    per_worker_cap = max(1, effective_cpu // max(worker_count, 1))

    try:
        load1 = os.getloadavg()[0]
    except Exception:
        load1 = None

    if load1 is not None and effective_cpu > 0:
        if load1 > effective_cpu * 0.8 and worker_count > 1:
            new_workers = max(1, int(effective_cpu * 0.6))
            if new_workers < worker_count:
                logger.info(
                    "[PARALLEL] High load %.2f (cpu=%s); reducing workers %s->%s",
                    load1,
                    effective_cpu,
                    worker_count,
                    new_workers,
                )
                worker_count = max(1, new_workers)
                per_worker_cap = max(1, effective_cpu // worker_count)
        elif load1 < effective_cpu * 0.5 and worker_count < max_workers:
            logger.info(
                "[PARALLEL] Load healthy %.2f (cpu=%s); keeping workers=%s",
                load1,
                effective_cpu,
                worker_count,
            )

    report_progress(15, "compressing", f"Compressing {num_chunks} chunks in parallel...")

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
            executor.submit(compress_single_chunk, chunk, force_split_dedup, None): (idx, chunk)
            for idx, chunk in tasks
        }

        # Collect results as they complete
        for future in as_completed(futures):
            chunk_idx, original_chunk = futures[future]
            chunk_mb_val: Optional[float] = None
            try:
                compressed_path, success, message, chunk_mb = future.result()
                chunk_mb_val = chunk_mb
            except Exception as e:
                logger.error(f"[PARALLEL] Chunk {chunk_idx + 1} error: {e}")
                failed_chunks.append((chunk_idx, original_chunk))
            else:
                skipped = isinstance(message, str) and message.startswith("SKIPPED:")
                if success and compressed_path.exists():
                    compressed_chunks.append((chunk_idx, compressed_path))
                    if skipped:
                        skipped_chunks += 1
                        skip_reason = message.split(":", 1)[1].strip() if ":" in message else message
                        logger.info(
                            "[PARALLEL_CHUNK] idx=%s size=%.1fMB status=skipped reason=%s",
                            chunk_idx + 1,
                            chunk_mb,
                            skip_reason,
                        )
                    else:
                        if chunk_mb_val:
                            # Track worst-case chunk inflation (if any)
                            comp_mb = get_file_size_mb(compressed_path)
                            if comp_mb > chunk_mb_val:
                                inflation_pct = ((comp_mb - chunk_mb_val) / chunk_mb_val) * 100
                                strategy_outcome["max_chunk_inflation_pct"] = max(
                                    strategy_outcome["max_chunk_inflation_pct"],
                                    inflation_pct,
                                )
                        logger.info(
                            "[PARALLEL_CHUNK] idx=%s size=%.1fMB status=compressed msg=%s",
                            chunk_idx + 1,
                            chunk_mb,
                            message,
                        )
                else:
                    logger.warning(
                        "[PARALLEL_CHUNK] idx=%s size=%.1fMB status=failed msg=%s",
                        chunk_idx + 1,
                        chunk_mb_val if chunk_mb_val is not None else -1,
                        message,
                    )
                    failed_chunks.append((chunk_idx, original_chunk))

            # Update progress
            done = len(compressed_chunks) + len(failed_chunks)
            pct = 15 + int(55 * done / num_chunks)
            report_progress(pct, "compressing", f"Compressed {done}/{num_chunks} chunks...")

    # Sort chunks by original order and combine with failed chunks
    all_chunks = compressed_chunks + failed_chunks
    all_chunks.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in all_chunks]

    compressed_ok = max(0, len(compressed_chunks) - skipped_chunks)
    logger.info(
        "[PARALLEL] Compression complete: %s compressed, %s skipped, %s used original",
        compressed_ok,
        skipped_chunks,
        len(failed_chunks),
    )
    compress_time = time.time() - start_ts - split_time
    logger.info("[PERF] Compress: %.1fs", compress_time)
    sla_exceeded = time.time() > sla_deadline
    if sla_exceeded:
        logger.warning(
            "[PARALLEL] SLA %.0fs exceeded (elapsed %.1fs); minimizing post-processing",
            PARALLEL_JOB_SLA_SEC,
            time.time() - start_ts,
        )

    # Step 4: Merge compressed chunks for consistent results, then split if required.
    split_enabled = split_threshold_mb is not None and split_threshold_mb > 0
    effective_split_trigger_mb = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb

    split_start = time.time()
    merged_path = working_dir / f"{base_name}_compressed.pdf"
    merged_path.unlink(missing_ok=True)

    direct_delivery = False
    merge_used = "skipped"
    merged_mb = 0.0
    final_parts: List[Path] = []

    def _build_direct_parts(paths: List[Path]) -> List[Path]:
        parts: List[Path] = []
        if split_threshold_mb is None or split_threshold_mb <= 0:
            return parts
        for idx, path in enumerate(paths, start=1):
            size_mb = get_file_size_mb(path)
            if size_mb <= split_threshold_mb:
                parts.append(path)
                continue
            logger.info(
                "[PARALLEL] Chunk %s is %.1fMB > %.1fMB; splitting for delivery",
                path.name,
                size_mb,
                split_threshold_mb,
            )
            chunk_base = f"{base_name}_direct_{idx}"
            chunk_parts = split_for_delivery(
                path,
                working_dir,
                chunk_base,
                split_threshold_mb,
                progress_callback=progress_callback,
                skip_optimization_under_threshold=not force_split_dedup,
            )
            if path not in chunk_parts:
                path.unlink(missing_ok=True)
            parts.extend(chunk_parts)
        return parts

    direct_total_mb = None
    if split_enabled:
        try:
            direct_total_mb = sum(get_file_size_mb(p) for p in ordered_paths)
        except Exception:
            direct_total_mb = None

    if (
        split_enabled
        and PARALLEL_SPLIT_DELIVER_CHUNKS
        and (effective_split_trigger_mb is None or (direct_total_mb and direct_total_mb > effective_split_trigger_mb))
    ):
        direct_delivery = True
        report_progress(75, "splitting", "Preparing split parts...")
        final_parts = _build_direct_parts(ordered_paths)
        merge_used = "direct"
        merged_mb = sum(p.stat().st_size for p in final_parts) / (1024 * 1024) if final_parts else 0.0
        logger.info("[PARALLEL] Split delivery: using %s chunk-derived part(s)", len(final_parts))
    else:
        report_progress(75, "merging", "Merging compressed chunks...")
        merge_used = "gs"
        if len(ordered_paths) == 1:
            ordered_paths[0].rename(merged_path)
            merge_used = "skipped"
        else:
            try:
                merge_pdfs(ordered_paths, merged_path)
                merge_used = "gs"
            except Exception as exc:
                merge_used = "pypdf2"
                strategy_outcome["timeouts"]["merge_timeout"] += 1 if isinstance(exc, TimeoutError) else 0
                raise

        if not merged_path.exists() or merged_path.stat().st_size == 0:
            raise SplitError(f"Merged output missing or empty: {merged_path.name}")

        final_parts = [merged_path]
        merged_mb = get_file_size_mb(merged_path)

        if split_enabled and effective_split_trigger_mb is not None:
            logger.info(
                "[PARALLEL] Split check: merged=%.1fMB trigger=%.1fMB threshold=%.1fMB",
                merged_mb,
                effective_split_trigger_mb,
                split_threshold_mb,
            )
            if merged_mb > effective_split_trigger_mb:
                report_progress(80, "splitting", "Splitting merged output...")
                final_parts = split_for_delivery(
                    merged_path,
                    working_dir,
                    f"{base_name}_merged",
                    split_threshold_mb,
                    progress_callback=progress_callback,
                    skip_optimization_under_threshold=not force_split_dedup,
                )
                merged_path.unlink(missing_ok=True)
                logger.info("[PARALLEL] Split merged output into %s part(s)", len(final_parts))
            else:
                logger.info(
                    "[PARALLEL] Split not required; merged output kept as a single file",
                )
        else:
            logger.info(
                "[PARALLEL] Split disabled; merged %s chunk(s) into %s",
                len(ordered_paths),
                merged_path.name,
            )

    # Cleanup original chunk files
    final_part_set = {p.resolve() for p in final_parts}
    source_chunks: List[Path] = [c for _, c in tasks]
    if "probe_chunk" in locals() and not probe_reinserted:
        source_chunks.append(probe_chunk)
    for chunk in source_chunks:
        if chunk.resolve() in final_part_set:
            continue
        chunk.unlink(missing_ok=True)
    for _, compressed in compressed_chunks:
        if compressed.resolve() in final_part_set:
            continue
        compressed.unlink(missing_ok=True)

    split_total = time.time() - split_start
    logger.info("[PERF] Direct split time: %.1fs", split_total)
    report_progress(100, "finalizing", f"Finalizing {len(final_parts)} parts...")

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

    merge_fallback_used = False
    merge_fallback_time = 0.0
    if split_enabled and PARALLEL_MERGE_ON_BLOAT and original_size_mb > 0 and not sla_exceeded and not direct_delivery:
        bloat_ratio = (combined_mb / original_size_mb) - 1
        if bloat_ratio > PARALLEL_BLOAT_PCT:
            bloat_pct = bloat_ratio * 100
            logger.warning(
                "[PARALLEL] Bloat detected: %.1fMB -> %.1fMB (+%.1f%%). Running merge fallback.",
                original_size_mb,
                combined_mb,
                bloat_pct,
            )
            merged_path = working_dir / f"{input_path.stem}_merged_dedup.pdf"
            try:
                report_progress(95, "finalizing", "Deduplicating output (merge fallback)...")
                merge_start = time.time()
                merge_pdfs(verified_parts, merged_path)
                if merged_path.exists():
                    if force_split_dedup:
                        logger.info("[PARALLEL] Merge fallback: dedupe optimization on split parts (forced)")
                    else:
                        logger.info("[PARALLEL] Merge fallback: fast split (skip per-part optimization)")
                    split_policy = "optimize_parts" if force_split_dedup else "skip_opt"
                    logger.info(
                        "[PARALLEL] Merge fallback split policy: %s (split_inflation=%.1f%%, threshold=%.1f%%)",
                        split_policy,
                        (split_inflation_ratio - 1) * 100,
                        PARALLEL_DEDUP_SPLIT_INFLATION_PCT * 100,
                    )
                    logger.info("[PARALLEL] Merge fallback: size-based split (no ultra)")
                    fallback_parts = split_by_size(
                        merged_path,
                        working_dir,
                        f"{input_path.stem}_merged",
                        split_threshold_mb,
                        skip_optimization_under_threshold=not force_split_dedup,
                    )
                    for part in verified_parts:
                        part.unlink(missing_ok=True)
                    merged_path.unlink(missing_ok=True)
                    verified_parts = fallback_parts
                    combined_bytes = sum(p.stat().st_size for p in verified_parts)
                    combined_mb = combined_bytes / (1024 * 1024)
                    reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100
                    merge_fallback_used = True
                    merge_fallback_time = time.time() - merge_start
                    strategy_outcome["merge_path"] = "gs_fallback_split"
                    logger.info(
                        "[PARALLEL] Merge fallback complete: parts=%s, reduction=%.1f%%",
                        len(verified_parts),
                        reduction_percent,
                    )
                else:
                    logger.warning("[PARALLEL] Merge fallback failed: merged file missing")
            except Exception as exc:
                logger.warning("[PARALLEL] Merge fallback failed: %s", exc)

    rebalance_attempted = False
    rebalance_applied = False
    rebalance_reason = None
    rebalance_parts_before = len(verified_parts)
    rebalance_parts_after = None
    rebalance_size_before_mb = round(combined_mb, 2)
    rebalance_size_after_mb = None
    rebalance_time = 0.0

    if (
        split_enabled
        and REBALANCE_SPLIT_ENABLED
        and not sla_exceeded
        and split_threshold_mb > 0
        and rebalance_parts_before > 1
        and not direct_delivery
    ):
        part_sizes_mb = [p.stat().st_size / (1024 * 1024) for p in verified_parts]
        max_part_mb = max(part_sizes_mb) if part_sizes_mb else 0.0
        min_parts = max(1, math.ceil(combined_mb / split_threshold_mb))
        parts_over_min = rebalance_parts_before - min_parts
        split_inflation_pct = round((split_inflation_ratio - 1) * 100, 1)
        high_inflation = split_inflation_pct >= REBALANCE_SPLIT_INFLATION_PCT
        slack_ok = max_part_mb <= split_threshold_mb * REBALANCE_MAX_PART_PCT

        candidate = (
            parts_over_min >= REBALANCE_MIN_PARTS_OVER_MIN
            and (high_inflation or force_split_dedup)
            and not merge_fallback_used
            and slack_ok
        )

        if candidate:
            rebalance_attempted = True
            rebalance_reason = (
                f"over_min={parts_over_min} inflation={split_inflation_pct:.1f}% "
                f"dedupe_parts={'yes' if force_split_dedup else 'no'} max_part={max_part_mb:.1f}MB"
            )
            rebalance_tag = str(uuid.uuid4())[:8]
            rebalance_base = f"{input_path.stem}_rebalance_{rebalance_tag}"
            merged_path = working_dir / f"{rebalance_base}.pdf"
            rebalance_start = time.time()
            try:
                merge_pdfs(verified_parts, merged_path)
                if merged_path.exists():
                    rebalance_parts = split_pdf(
                        merged_path,
                        working_dir,
                        rebalance_base,
                        split_threshold_mb,
                        skip_optimization_under_threshold=True,
                    )
                    rebalance_parts_after = len(rebalance_parts)
                    rebalance_size_after_mb = sum(
                        p.stat().st_size for p in rebalance_parts
                    ) / (1024 * 1024)
                    if rebalance_parts_after < rebalance_parts_before:
                        for part in verified_parts:
                            part.unlink(missing_ok=True)
                        verified_parts = rebalance_parts
                        combined_bytes = sum(p.stat().st_size for p in verified_parts)
                        combined_mb = combined_bytes / (1024 * 1024)
                        reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100
                        rebalance_applied = True
                    else:
                        for part in rebalance_parts:
                            part.unlink(missing_ok=True)
                else:
                    logger.warning("[REBALSPLIT] merged file missing; skipping")
            except Exception as exc:
                logger.warning("[REBALSPLIT] failed: %s", exc)
                for part in working_dir.glob(f"{rebalance_base}_part*.pdf"):
                    part.unlink(missing_ok=True)
            finally:
                merged_path.unlink(missing_ok=True)
                rebalance_time = time.time() - rebalance_start
                rebalance_parts_after = rebalance_parts_after or rebalance_parts_before
                size_after = rebalance_size_after_mb or rebalance_size_before_mb
                logger.info(
                    "[REBALSPLIT] triggered=yes kept=%s reason=%s parts_before=%s parts_after=%s "
                    "size_before=%.1fMB size_after=%.1fMB time=%.1fs",
                    "yes" if rebalance_applied else "no",
                    rebalance_reason,
                    rebalance_parts_before,
                    rebalance_parts_after,
                    rebalance_size_before_mb,
                    size_after,
                    rebalance_time,
                )
        elif (
            parts_over_min >= REBALANCE_MIN_PARTS_OVER_MIN
            and (high_inflation or force_split_dedup)
            and not merge_fallback_used
        ):
            rebalance_reason = (
                f"over_min={parts_over_min} inflation={split_inflation_pct:.1f}% "
                f"dedupe_parts={'yes' if force_split_dedup else 'no'} max_part={max_part_mb:.1f}MB "
                f"slack_ok={'yes' if slack_ok else 'no'}"
            )
            logger.info(
                "[REBALSPLIT] triggered=no reason=%s parts_before=%s size_before=%.1fMB",
                rebalance_reason,
                rebalance_parts_before,
                rebalance_size_before_mb,
            )

    bloat_detected = False
    bloat_action = None
    bloat_pct = 0.0
    if original_size_mb > 0 and combined_mb >= original_size_mb:
        bloat_detected = True
        bloat_pct = round(((combined_mb / original_size_mb) - 1) * 100, 1)
        if split_enabled and direct_delivery:
            logger.warning(
                "[PARALLEL] Output larger than input (%.1fMB -> %.1fMB, +%.1f%%); keeping split parts",
                original_size_mb,
                combined_mb,
                bloat_pct,
            )
            bloat_action = "direct_parts"
        elif split_enabled:
            logger.warning(
                "[PARALLEL] Output larger than input (%.1fMB -> %.1fMB, +%.1f%%); splitting original instead",
                original_size_mb,
                combined_mb,
                bloat_pct,
            )
            for part in verified_parts:
                part.unlink(missing_ok=True)
            try:
                verified_parts = split_for_delivery(
                    input_path,
                    working_dir,
                    f"{input_path.stem}_original",
                    split_threshold_mb,
                    progress_callback=progress_callback,
                    prefer_binary=True,
                    # Always optimize parts when recovering from bloat; PyPDF2 raw
                    # splits duplicate resources and can easily exceed input size.
                    skip_optimization_under_threshold=False,
                )
                combined_bytes = sum(p.stat().st_size for p in verified_parts)
                combined_mb = combined_bytes / (1024 * 1024)
                reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100
                bloat_action = "split_original"
                bloat_pct = round(((combined_mb / original_size_mb) - 1) * 100, 1)
            except Exception as exc:
                logger.warning(
                    "[PARALLEL] Fallback split of original failed: %s; returning original instead",
                    exc,
                )
                # Fail safe: return the original, not a bloated output.
                merged_path = verified_parts[0] if verified_parts else (working_dir / f"{input_path.stem}_compressed.pdf")
                if merged_path.resolve() != input_path.resolve():
                    merged_path.unlink(missing_ok=True)
                    shutil.copy2(input_path, merged_path)
                verified_parts = [merged_path]
                combined_mb = original_size_mb
                reduction_percent = 0.0
                bloat_action = "return_original"
                bloat_pct = 0.0
        else:
            logger.warning(
                "[PARALLEL] Output larger than input (%.1fMB -> %.1fMB, +%.1f%%); returning original",
                original_size_mb,
                combined_mb,
                bloat_pct,
            )
            lossless_used = False
            lossless_path = working_dir / f"{input_path.stem}_lossless_fallback.pdf"
            try:
                lossless_timeout = min(240, max(90, int(original_size_mb * 1.2)))
                ok_lossless, msg_lossless = compress_pdf_lossless(
                    input_path,
                    lossless_path,
                    timeout_override=lossless_timeout,
                )
                if ok_lossless and lossless_path.exists():
                    lossless_mb = get_file_size_mb(lossless_path)
                    if lossless_mb < combined_mb:
                        for part in verified_parts:
                            part.unlink(missing_ok=True)
                        verified_parts = [lossless_path]
                        combined_bytes = lossless_path.stat().st_size
                        combined_mb = combined_bytes / (1024 * 1024)
                        reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100
                        bloat_action = "lossless_fallback"
                        bloat_detected = False
                        bloat_pct = round(((combined_mb / original_size_mb) - 1) * 100, 1) if original_size_mb > 0 else 0.0
                        lossless_used = True
                        logger.info(
                            "[PARALLEL] Lossless fallback after bloat succeeded: %.1fMB -> %.1fMB (timeout=%ss)",
                            original_size_mb,
                            combined_mb,
                            lossless_timeout,
                        )
                    else:
                        lossless_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("[PARALLEL] Lossless fallback after bloat failed: %s", exc)
                lossless_path.unlink(missing_ok=True)

            # If lossless didn't help and lossy is allowed, try ultra fallback.
            ultra_used = False
            if not lossless_used and allow_lossy:
                ultra_path = working_dir / f"{input_path.stem}_ultra_parallel.pdf"
                ultra_timeout = min(240, max(90, int(original_size_mb * 1.2)))
                try:
                    logger.info(
                        "[PARALLEL] Trying ultra fallback after bloat (%.1fMB, timeout=%ss)",
                        original_size_mb,
                        ultra_timeout,
                    )
                    ok_ultra, msg_ultra = compress_ultra_fallback(
                        input_path,
                        ultra_path,
                        timeout_override=ultra_timeout,
                    )
                    if ok_ultra and ultra_path.exists():
                        ultra_mb = get_file_size_mb(ultra_path)
                        if ultra_mb < original_size_mb:
                            for part in verified_parts:
                                part.unlink(missing_ok=True)
                            verified_parts = [ultra_path]
                            combined_bytes = ultra_path.stat().st_size
                            combined_mb = combined_bytes / (1024 * 1024)
                            reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100
                            bloat_action = "ultra_fallback"
                            bloat_detected = False
                            bloat_pct = 0.0
                            ultra_used = True
                            logger.info(
                                "[PARALLEL] Ultra fallback succeeded: %.1fMB -> %.1fMB (%.1f%%)",
                                original_size_mb,
                                combined_mb,
                                reduction_percent,
                            )
                        else:
                            logger.warning(
                                "[PARALLEL] Ultra fallback also inflated (%.1fMB -> %.1fMB); returning original",
                                original_size_mb,
                                ultra_mb,
                            )
                            ultra_path.unlink(missing_ok=True)
                    else:
                        logger.warning("[PARALLEL] Ultra fallback failed: %s", msg_ultra)
                        ultra_path.unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("[PARALLEL] Ultra fallback error: %s", exc)
                    ultra_path.unlink(missing_ok=True)

            if not lossless_used and not ultra_used:
                merged_path = verified_parts[0] if verified_parts else (working_dir / f"{input_path.stem}_compressed.pdf")
                if merged_path.resolve() != input_path.resolve():
                    merged_path.unlink(missing_ok=True)
                    shutil.copy2(input_path, merged_path)
                verified_parts = [merged_path]
                combined_mb = original_size_mb
                reduction_percent = 0.0
                bloat_action = "return_original"
                bloat_pct = 0.0

    strategy_outcome["merge_path"] = strategy_outcome["merge_path"] or merge_used
    strategy_outcome["timeouts"]["job_sla_hit"] = sla_exceeded
    strategy_outcome["max_merge_inflation_pct"] = max(
        strategy_outcome["max_merge_inflation_pct"],
        max(0.0, ((merged_mb / original_size_mb) - 1) * 100) if original_size_mb > 0 else 0.0,
    )

    logger.info(
        "[PARALLEL] Summary: %.1fMB -> %.1fMB (%.1f%%), parts=%s, compressed=%s, skipped=%s, "
        "used_original=%s, split_inflation=%s (%.1f%%), dedupe_parts=%s, merge_fallback=%s, "
        "bloat_detected=%s, bloat_action=%s, bloat_pct=%.1f%%, sla_exceeded=%s, "
        "strategy=%s",
        original_size_mb,
        combined_mb,
        reduction_percent,
        len(verified_parts),
        compressed_ok,
        skipped_chunks,
        len(failed_chunks),
        "yes" if split_inflation else "no",
        (split_inflation_ratio - 1) * 100,
        "yes" if force_split_dedup else "no",
        "yes" if merge_fallback_used else "no",
        "yes" if bloat_detected else "no",
        bloat_action or "none",
        bloat_pct,
        "yes" if sla_exceeded else "no",
        strategy_outcome,
    )
    total_time = time.time() - start_ts
    logger.info(
        "[PARALLEL_METRICS] total=%.1fs split=%.1fs compress=%.1fs split_parts=%.1fs merge_fallback=%.1fs sla=%s",
        total_time,
        split_time,
        compress_time,
        split_total,
        merge_fallback_time,
        "hit" if sla_exceeded else "ok",
    )

    return {
        "output_path": str(verified_parts[0]),
        "output_paths": [str(p) for p in verified_parts],
        "original_size_mb": round(original_size_mb, 2),
        "compressed_size_mb": round(combined_mb, 2),
        "reduction_percent": round(reduction_percent, 1),
        "compression_method": "ghostscript_parallel",
        "compression_mode": compression_mode,
        "quality_mode": "aggressive_72dpi" if compression_mode == "aggressive" else "lossless",
        "was_split": len(verified_parts) > 1,
        "total_parts": len(verified_parts),
        "success": True,
        "page_count": page_count,
        "part_sizes": [p.stat().st_size for p in verified_parts],
        "parallel_chunks": num_chunks,
        "split_inflation": split_inflation,
        "split_inflation_pct": round((split_inflation_ratio - 1) * 100, 1),
        "dedupe_parts": force_split_dedup,
        "merge_fallback": merge_fallback_used,
        "merge_fallback_time": round(merge_fallback_time, 2),
        "rebalance_attempted": rebalance_attempted,
        "rebalance_applied": rebalance_applied,
        "rebalance_reason": rebalance_reason,
        "rebalance_parts_before": rebalance_parts_before,
        "rebalance_parts_after": rebalance_parts_after,
        "rebalance_size_before_mb": rebalance_size_before_mb,
        "rebalance_size_after_mb": None if rebalance_size_after_mb is None else round(rebalance_size_after_mb, 2),
        "rebalance_time": round(rebalance_time, 2),
        "bloat_detected": bloat_detected,
        "bloat_pct": bloat_pct,
        "bloat_action": bloat_action,
        "sla_exceeded": sla_exceeded,
        "probe_bailout": False,
        "probe_bailout_reason": None,
    }


# =============================================================================
# MICRO PROBE (small sample compression to predict risk)
# =============================================================================

def run_micro_probe(
    input_path: Path,
    compression_mode: str,
    max_pages: int = PROBE_SAMPLE_PAGES,
) -> Dict:
    """
    Compress a small sample (first few pages) to predict inflation/throughput.

    Returns a dict with success, delta_pct, elapsed, in_mb, out_mb, pages, message.
    """
    from PyPDF2 import PdfReader, PdfWriter

    t0 = time.time()
    sample_path = input_path.parent / f"{input_path.stem}_probe.pdf"
    sample_out = input_path.parent / f"{input_path.stem}_probe_out.pdf"
    result = {
        "success": False,
        "delta_pct": 0.0,
        "elapsed": 0.0,
        "in_mb": 0.0,
        "out_mb": 0.0,
        "pages": 0,
        "message": "",
    }
    try:
        with open(input_path, "rb") as f:
            reader = PdfReader(f, strict=False)
            pages = min(max_pages, len(reader.pages))
            writer = PdfWriter()
            for i in range(pages):
                writer.add_page(reader.pages[i])
            with open(sample_path, "wb") as out_f:
                writer.write(out_f)
        sample_bytes = sample_path.stat().st_size
        result["in_mb"] = sample_bytes / (1024 * 1024)
        result["pages"] = pages

        compress_fn = compress_pdf_lossless if compression_mode == "lossless" else compress_pdf_with_ghostscript
        success, message = compress_fn(
            sample_path,
            sample_out,
            timeout_override=PROBE_TIME_BUDGET_SEC,
        )
        result["elapsed"] = time.time() - t0
        result["message"] = message
        if success and sample_out.exists():
            out_bytes = sample_out.stat().st_size
            result["out_mb"] = out_bytes / (1024 * 1024)
            if result["in_mb"] > 0:
                result["delta_pct"] = ((result["out_mb"] - result["in_mb"]) / result["in_mb"]) * 100
            result["success"] = True
        else:
            result["success"] = False
    except Exception as exc:
        result["message"] = str(exc)
    finally:
        sample_path.unlink(missing_ok=True)
        sample_out.unlink(missing_ok=True)
        result["elapsed"] = time.time() - t0
    return result
