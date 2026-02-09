"""Ghostscript PDF compression helpers and parallel pipeline."""

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
from utils import env_bool, env_choice, env_float, env_int, get_effective_cpu_count

logger = logging.getLogger(__name__)

TARGET_CHUNK_MB = float(os.environ.get("TARGET_CHUNK_MB", "30"))
MAX_CHUNK_MB = float(os.environ.get("MAX_CHUNK_MB", "50"))
MAX_PARALLEL_CHUNKS = int(os.environ.get("MAX_PARALLEL_CHUNKS", "64"))
MAX_PAGES_PER_CHUNK = int(os.environ.get("MAX_PAGES_PER_CHUNK", "600"))
MIN_CHUNK_MB = 5.0

LARGE_FILE_TUNE_MIN_MB = float(os.environ.get("LARGE_FILE_TUNE_MIN_MB", "200"))
LARGE_FILE_TARGET_CHUNK_MB = float(os.environ.get("LARGE_FILE_TARGET_CHUNK_MB", "50"))
LARGE_FILE_MAX_CHUNK_MB = float(os.environ.get("LARGE_FILE_MAX_CHUNK_MB", "75"))
LARGE_FILE_MAX_PARALLEL_CHUNKS = int(os.environ.get("LARGE_FILE_MAX_PARALLEL_CHUNKS", "12"))

PARALLEL_SPLIT_INFLATION_PCT = 0.02

SLA_MAX_PARALLEL_CHUNKS = max(2, env_int("SLA_MAX_PARALLEL_CHUNKS", 8))
SLA_MAX_PARALLEL_CHUNKS_LARGE = max(2, env_int("SLA_MAX_PARALLEL_CHUNKS_LARGE", 12))
HARD_MAX_PARALLEL_CHUNKS = max(2, env_int("HARD_MAX_PARALLEL_CHUNKS", 96))

GS_FAST_WEB_VIEW = env_bool("GS_FAST_WEB_VIEW", False)
GS_COLOR_DOWNSAMPLE_TYPE = env_choice(
    "GS_COLOR_DOWNSAMPLE_TYPE",
    "/Subsample",
    ("/Subsample", "/Average", "/Bicubic"),
)
GS_GRAY_DOWNSAMPLE_TYPE = env_choice(
    "GS_GRAY_DOWNSAMPLE_TYPE",
    "/Subsample",
    ("/Subsample", "/Average", "/Bicubic"),
)
GS_BAND_HEIGHT = env_int("GS_BAND_HEIGHT", 100)
GS_BAND_BUFFER_SPACE_MB = env_int("GS_BAND_BUFFER_SPACE_MB", 500)
GS_COLOR_IMAGE_RESOLUTION = env_int("GS_COLOR_IMAGE_RESOLUTION", 72)
GS_GRAY_IMAGE_RESOLUTION = env_int("GS_GRAY_IMAGE_RESOLUTION", 72)
GS_MONO_IMAGE_RESOLUTION = env_int("GS_MONO_IMAGE_RESOLUTION", 200)
DEFAULT_GS_THREADS_PER_WORKER = 1

CHUNK_TIME_BUDGET_MIN_SEC = 60
CHUNK_TIME_BUDGET_MAX_SEC = 300
CHUNK_TIME_BUDGET_MAX_SEC_LARGE = 600
PROBE_TIME_BUDGET_SEC = 45

PARALLEL_JOB_SLA_SEC = int(os.environ.get("PARALLEL_JOB_SLA_SEC", "300"))
PARALLEL_JOB_SLA_LARGE_MIN_MB = float(os.environ.get("PARALLEL_JOB_SLA_LARGE_MIN_MB", "200"))
PARALLEL_JOB_SLA_LARGE_SEC_PER_MB = float(os.environ.get("PARALLEL_JOB_SLA_LARGE_SEC_PER_MB", "2.5"))
PARALLEL_JOB_SLA_MAX_SEC = int(os.environ.get("PARALLEL_JOB_SLA_MAX_SEC", "1800"))

def _chunk_timeout_seconds(
    file_mb: float,
    large_file: bool = False,
    page_count: Optional[int] = None,
    sec_per_page: Optional[float] = None,
) -> int:
    """Compute per-chunk timeout from size and page complexity."""
    cap = CHUNK_TIME_BUDGET_MAX_SEC_LARGE if large_file else CHUNK_TIME_BUDGET_MAX_SEC
    mb_estimate = int(max(1.0, file_mb) * 8)
    page_estimate = 0
    if page_count and sec_per_page:
        page_estimate = int(max(1.0, page_count * sec_per_page) * 2.5)

    timeout = max(CHUNK_TIME_BUDGET_MIN_SEC, mb_estimate, page_estimate)
    return min(timeout, cap)


def _resolve_gs_threads(num_threads: Optional[int]) -> int:
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

    return max(1, min(4, get_effective_cpu_count()))


def _resolve_parallel_sla(file_mb: float, processing_profile: Optional[Dict[str, object]] = None) -> int:
    """Resolve total parallel job SLA in seconds."""
    if processing_profile and "recommended_sla_sec" in processing_profile:
        try:
            raw = int(processing_profile["recommended_sla_sec"])
        except (TypeError, ValueError):
            raw = PARALLEL_JOB_SLA_SEC
    elif file_mb >= PARALLEL_JOB_SLA_LARGE_MIN_MB:
        raw = max(PARALLEL_JOB_SLA_SEC, int(file_mb * PARALLEL_JOB_SLA_LARGE_SEC_PER_MB))
    else:
        raw = PARALLEL_JOB_SLA_SEC

    return min(max(PARALLEL_JOB_SLA_SEC, raw), PARALLEL_JOB_SLA_MAX_SEC)


def optimize_split_part(
    input_path: Path,
    output_path: Path,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """Lossless optimization for split parts."""
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
    """Run Ghostscript with lossless settings."""
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
        "-dPDFSETTINGS=/default",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        f"-dNumRenderingThreads={threads}",
        "-dDownsampleColorImages=false",
        "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dPassThroughJPEGImages=true",
        "-dPassThroughJPXImages=true",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        f"-sOutputFile={output_path}",
        str(input_path),
    ]

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = timeout_override or max(120, int(file_mb * 5))

        logger.info("%s %s (%.1fMB)", label, input_path.name, file_mb)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100 if file_mb > 0 else 0.0
        logger.info("Optimized: %.1fMB -> %.1fMB (%.1f%% reduction)", file_mb, out_mb, reduction)
        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as exc:
        return False, str(exc)


def compress_pdf_lossless(
    input_path: Path,
    output_path: Path,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """Lossless PDF optimization using Ghostscript."""
    return _run_lossless_ghostscript(
        input_path,
        output_path,
        "Lossless optimize",
        num_threads=num_threads,
        timeout_override=timeout_override,
    )


def get_ghostscript_command() -> Optional[str]:
    """Get Ghostscript executable name for current platform."""
    for name in ["gs", "gswin64c", "gswin32c"]:
        if shutil.which(name):
            return name
    return None


def translate_ghostscript_error(stderr: str, return_code: int) -> str:
    """Translate Ghostscript stderr to user-friendly errors."""
    logger.error("Ghostscript failed (exit code %s). Full error:\n%s", return_code, stderr)

    stderr_lower = stderr.lower()

    if "invalidfileaccess" in stderr_lower or "password" in stderr_lower:
        return "PDF is password-protected or locked. Please remove the password and try again."

    if "typecheck" in stderr_lower or "rangecheck" in stderr_lower:
        return "PDF has corrupted internal data. Try re-saving it from Adobe Acrobat."

    if any(x in stderr_lower for x in ["undefined", "ioerror", "syntaxerror", "eofread"]):
        return "PDF is damaged or corrupted. Please use a different copy of the file."

    return f"PDF processing failed (Ghostscript exit code {return_code}). The file may be corrupted."


def compress_aggressive(
    input_path: Path,
    output_path: Path,
    color_dpi: int = 72,
    gray_dpi: int = 72,
    mono_dpi: int = 200,
    jpeg_quality: int = 50,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """Aggressive lossy compression with configurable DPI/JPEG quality."""
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
        "-dPassThroughJPEGImages=false",
        "-dPassThroughJPXImages=false",
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        "-dDownsampleColorImages=true",
        f"-dColorImageDownsampleType={GS_COLOR_DOWNSAMPLE_TYPE}",
        f"-dColorImageResolution={max(36, int(color_dpi))}",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageDownsampleType={GS_GRAY_DOWNSAMPLE_TYPE}",
        f"-dGrayImageResolution={max(36, int(gray_dpi))}",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        f"-dMonoImageResolution={max(72, int(mono_dpi))}",
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

    cmd.extend([f"-sOutputFile={output_path}", str(input_path)])

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = timeout_override or _chunk_timeout_seconds(file_mb)

        logger.info(
            "[DIAG] GS aggressive cmd: dpi=%s/%s/%s downsample=%s/%s "
            "threads=%s timeout=%ss fast_web=%s band_height=%s band_buffer_mb=%s",
            color_dpi,
            gray_dpi,
            mono_dpi,
            GS_COLOR_DOWNSAMPLE_TYPE,
            GS_GRAY_DOWNSAMPLE_TYPE,
            threads,
            timeout,
            GS_FAST_WEB_VIEW,
            GS_BAND_HEIGHT,
            GS_BAND_BUFFER_SPACE_MB,
        )
        logger.info("Compressing %s (%.1fMB)", input_path.name, file_mb)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100 if file_mb > 0 else 0.0
        logger.info("Result: %.1fMB -> %.1fMB (%.1f%% reduction)", file_mb, out_mb, reduction)
        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as exc:
        return False, str(exc)


def compress_pdf_with_ghostscript(
    input_path: Path,
    output_path: Path,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """Backward-compatible wrapper for aggressive compression."""
    return compress_aggressive(
        input_path,
        output_path,
        color_dpi=GS_COLOR_IMAGE_RESOLUTION,
        gray_dpi=GS_GRAY_IMAGE_RESOLUTION,
        mono_dpi=GS_MONO_IMAGE_RESOLUTION,
        jpeg_quality=env_int("GS_JPEGQ", 50),
        num_threads=num_threads,
        timeout_override=timeout_override,
    )


def compress_ultra_aggressive(
    input_path: Path,
    output_path: Path,
    jpeg_quality: int = 50,
    num_threads: Optional[int] = None,
    timeout_override: Optional[int] = None,
) -> Tuple[bool, str]:
    """Backward-compatible wrapper for aggressive compression with custom quality."""
    return compress_aggressive(
        input_path,
        output_path,
        color_dpi=GS_COLOR_IMAGE_RESOLUTION,
        gray_dpi=GS_GRAY_IMAGE_RESOLUTION,
        mono_dpi=GS_MONO_IMAGE_RESOLUTION,
        jpeg_quality=jpeg_quality,
        num_threads=num_threads,
        timeout_override=timeout_override,
    )


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
    """Backward-compatible wrapper for aggressive compression with custom settings."""
    return compress_aggressive(
        input_path,
        output_path,
        color_dpi=color_res,
        gray_dpi=gray_res,
        mono_dpi=mono_res,
        jpeg_quality=jpeg_quality,
        num_threads=num_threads,
        timeout_override=timeout_override,
    )


def _read_pages(path: Path) -> Optional[int]:
    try:
        from PyPDF2 import PdfReader

        with open(path, "rb") as handle:
            return len(PdfReader(handle, strict=False).pages)
    except Exception:
        return None


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
    micro_probe_delta: Optional[float] = None,
    micro_probe_success: bool = False,
    processing_profile: Optional[Dict[str, object]] = None,
) -> Dict:
    """Linear parallel compression path with no fallback cascade."""
    del micro_probe_delta, micro_probe_success

    from split_pdf import merge_pdfs, split_by_pages, split_for_delivery
    from utils import get_file_size_mb

    input_path = Path(input_path)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    start_ts = time.time()

    def report_progress(percent: int, stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(percent, stage, message)

    original_size_mb = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size

    profile = processing_profile or {}
    profile_tier = str(profile.get("tier", "fast"))
    try:
        profile_spp = float(profile.get("sec_per_page", 0.25))
    except (TypeError, ValueError):
        profile_spp = 0.25
    try:
        profile_max_pages = int(profile.get("max_pages_per_chunk", MAX_PAGES_PER_CHUNK))
    except (TypeError, ValueError):
        profile_max_pages = MAX_PAGES_PER_CHUNK
    try:
        profile_chunk_timeout = int(profile.get("chunk_timeout_sec", 0))
    except (TypeError, ValueError):
        profile_chunk_timeout = 0

    compression_mode = (compression_mode or "aggressive").lower()
    if compression_mode not in ("lossless", "aggressive"):
        compression_mode = "aggressive"

    effective_target_chunk = TARGET_CHUNK_MB if target_chunk_mb is None else float(target_chunk_mb)
    effective_max_chunk = MAX_CHUNK_MB if max_chunk_mb is None else float(max_chunk_mb)

    if original_size_mb >= LARGE_FILE_TUNE_MIN_MB and target_chunk_mb is None:
        effective_target_chunk = max(effective_target_chunk, LARGE_FILE_TARGET_CHUNK_MB)
        effective_max_chunk = max(effective_max_chunk, LARGE_FILE_MAX_CHUNK_MB)

    effective_target_chunk = max(MIN_CHUNK_MB, effective_target_chunk)
    effective_max_chunk = max(effective_target_chunk, effective_max_chunk)

    soft_parallel_chunks = max(2, MAX_PARALLEL_CHUNKS)
    hard_parallel_chunks = max(soft_parallel_chunks, HARD_MAX_PARALLEL_CHUNKS)

    total_pages = input_page_count
    if total_pages is None:
        total_pages = _read_pages(input_path)

    max_pages_per_chunk = max(1, min(profile_max_pages, MAX_PAGES_PER_CHUNK if profile_max_pages <= 0 else profile_max_pages))
    if total_pages:
        max_pages_per_chunk = max(1, min(max_pages_per_chunk, total_pages))

    num_chunks_by_size = max(1, math.ceil(original_size_mb / effective_target_chunk))
    num_chunks_by_pages = 1
    if total_pages:
        num_chunks_by_pages = max(1, math.ceil(total_pages / max_pages_per_chunk))

    requested_chunks = max(num_chunks_by_size, num_chunks_by_pages)
    if requested_chunks > soft_parallel_chunks:
        logger.info(
            "[PARALLEL] Raising chunk count beyond MAX_PARALLEL_CHUNKS=%s to satisfy page budget (requested=%s, hard_cap=%s)",
            soft_parallel_chunks,
            requested_chunks,
            hard_parallel_chunks,
        )
    num_chunks = max(2, min(requested_chunks, hard_parallel_chunks))

    job_sla_sec = _resolve_parallel_sla(original_size_mb, processing_profile=processing_profile)
    sla_deadline = start_ts + job_sla_sec

    logger.info(
        "[PARALLEL] Start file=%s size=%.1fMB mode=%s tier=%s spp=%.3f pages_per_chunk=%s workers=%s target=%.1fMB max=%.1fMB chunks=%s sla=%ss",
        input_path.name,
        original_size_mb,
        compression_mode,
        profile_tier,
        profile_spp,
        max_pages_per_chunk,
        max_workers,
        effective_target_chunk,
        effective_max_chunk,
        num_chunks,
        job_sla_sec,
    )

    report_progress(5, "splitting", f"Splitting into {num_chunks} chunks...")
    split_started = time.time()
    chunk_paths = split_by_pages(input_path, working_dir, num_chunks, base_name)
    split_time = time.time() - split_started

    chunk_sizes = [get_file_size_mb(path) for path in chunk_paths]
    chunk_total_mb = sum(chunk_sizes)
    split_inflation_ratio = (chunk_total_mb / original_size_mb) if original_size_mb > 0 else 1.0
    split_inflation = split_inflation_ratio > (1 + PARALLEL_SPLIT_INFLATION_PCT)

    logger.info(
        "[PARALLEL] Split complete chunks=%s split_time=%.1fs chunk_total=%.1fMB inflation=%.1f%%",
        len(chunk_paths),
        split_time,
        chunk_total_mb,
        (split_inflation_ratio - 1) * 100,
    )

    compress_fn = compress_pdf_lossless if compression_mode == "lossless" else compress_pdf_with_ghostscript

    worker_count = max(1, min(max_workers, len(chunk_paths)))
    threads_per_worker = DEFAULT_GS_THREADS_PER_WORKER

    env_threads = os.environ.get("GS_NUM_RENDERING_THREADS")
    if env_threads:
        try:
            threads_per_worker = max(1, int(env_threads))
        except ValueError:
            pass

    report_progress(20, "compressing", f"Compressing {len(chunk_paths)} chunks...")

    temp_outputs: List[Path] = []
    results: Dict[int, Path] = {}
    failed_chunks = 0

    def compress_single_chunk(idx: int, chunk_path: Path) -> Tuple[int, Path, bool]:
        chunk_mb = get_file_size_mb(chunk_path)
        chunk_pages = _read_pages(chunk_path)
        timeout = profile_chunk_timeout or _chunk_timeout_seconds(
            chunk_mb,
            large_file=(original_size_mb >= LARGE_FILE_TUNE_MIN_MB),
            page_count=chunk_pages,
            sec_per_page=profile_spp,
        )

        out_path = working_dir / f"{chunk_path.stem}_{uuid.uuid4().hex[:8]}_compressed.pdf"
        ok, msg = compress_fn(
            chunk_path,
            out_path,
            num_threads=threads_per_worker,
            timeout_override=timeout,
        )

        if not ok or not out_path.exists():
            out_path.unlink(missing_ok=True)
            logger.warning("[PARALLEL_CHUNK] idx=%s status=failed timeout=%ss msg=%s", idx + 1, timeout, msg)
            return idx, chunk_path, True

        if out_path.stat().st_size >= chunk_path.stat().st_size:
            out_path.unlink(missing_ok=True)
            logger.info("[PARALLEL_CHUNK] idx=%s status=kept_original reason=no_reduction", idx + 1)
            return idx, chunk_path, False

        logger.info("[PARALLEL_CHUNK] idx=%s status=compressed timeout=%ss", idx + 1, timeout)
        return idx, out_path, False

    compress_started = time.time()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(compress_single_chunk, idx, path): idx
            for idx, path in enumerate(chunk_paths)
        }
        completed = 0
        for future in as_completed(futures):
            idx, chosen_path, failed = future.result()
            results[idx] = chosen_path
            if chosen_path != chunk_paths[idx]:
                temp_outputs.append(chosen_path)
            if failed:
                failed_chunks += 1

            completed += 1
            pct = 20 + int(45 * completed / max(1, len(chunk_paths)))
            report_progress(pct, "compressing", f"Compressed {completed}/{len(chunk_paths)} chunks")

    compress_time = time.time() - compress_started
    ordered_paths = [results[idx] for idx in sorted(results.keys())]

    report_progress(70, "merging", "Merging chunks...")
    merge_started = time.time()
    merged_path = working_dir / f"{base_name}_compressed.pdf"
    merged_path.unlink(missing_ok=True)

    merge_failed = False
    if len(ordered_paths) == 1:
        src = ordered_paths[0]
        if src.resolve() != merged_path.resolve():
            shutil.copy2(src, merged_path)
        else:
            merged_path = src
    else:
        try:
            merge_pdfs(ordered_paths, merged_path)
        except Exception as exc:
            merge_failed = True
            logger.warning("[PARALLEL] Merge failed, falling back to original output: %s", exc)
            merged_path.unlink(missing_ok=True)
            shutil.copy2(input_path, merged_path)

    merge_time = time.time() - merge_started

    split_enabled = split_threshold_mb is not None and split_threshold_mb > 0
    effective_trigger = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb

    final_parts: List[Path] = [merged_path]
    bloat_detected = False
    bloat_action: Optional[str] = None

    combined_mb = merged_path.stat().st_size / (1024 * 1024)

    if combined_mb >= original_size_mb:
        bloat_detected = True
        if split_enabled and effective_trigger is not None and original_size_mb > effective_trigger:
            report_progress(80, "splitting", "Splitting original for delivery...")
            final_parts = split_for_delivery(
                input_path,
                working_dir,
                f"{base_name}_original",
                split_threshold_mb,
                progress_callback=progress_callback,
                skip_optimization_under_threshold=True,
            )
            bloat_action = "split_original"
        else:
            merged_path.unlink(missing_ok=True)
            shutil.copy2(input_path, merged_path)
            final_parts = [merged_path]
            bloat_action = "return_original"
    elif split_enabled and effective_trigger is not None and combined_mb > effective_trigger:
        report_progress(80, "splitting", "Splitting compressed output...")
        final_parts = split_for_delivery(
            merged_path,
            working_dir,
            f"{base_name}_merged",
            split_threshold_mb,
            progress_callback=progress_callback,
            skip_optimization_under_threshold=True,
        )
        if len(final_parts) > 1:
            merged_path.unlink(missing_ok=True)

    final_set = {part.resolve() for part in final_parts}
    for path in chunk_paths:
        if path.resolve() not in final_set:
            path.unlink(missing_ok=True)
    for path in temp_outputs:
        if path.resolve() not in final_set:
            path.unlink(missing_ok=True)

    verified_parts: List[Path] = []
    for part in final_parts:
        if not part.exists() or part.stat().st_size == 0:
            raise SplitError(f"Part missing or empty: {part}")
        verified_parts.append(part)

    combined_bytes = sum(path.stat().st_size for path in verified_parts)
    combined_mb = combined_bytes / (1024 * 1024)
    reduction_percent = ((original_size_mb - combined_mb) / original_size_mb) * 100 if original_size_mb > 0 else 0.0
    bloat_pct = round(max(0.0, ((combined_mb / original_size_mb) - 1) * 100), 1) if original_size_mb > 0 else 0.0

    total_time = time.time() - start_ts
    sla_exceeded = time.time() > sla_deadline

    timings_dict = {
        "total_seconds": round(total_time, 2),
        "split_seconds": round(split_time, 2),
        "compress_seconds": round(compress_time, 2),
        "merge_seconds": round(merge_time, 2),
        "fallback_seconds": 0.0,
        "probe_seconds": 0.0,
        "sla_seconds": int(job_sla_sec),
        "sla_breached": sla_exceeded,
    }

    logger.info(
        "[PARALLEL] Summary %.1fMB -> %.1fMB (%.1f%%) parts=%s chunks=%s failed=%s split_inflation=%s (%.1f%%) bloat=%s action=%s sla=%s",
        original_size_mb,
        combined_mb,
        reduction_percent,
        len(verified_parts),
        len(chunk_paths),
        failed_chunks,
        "yes" if split_inflation else "no",
        (split_inflation_ratio - 1) * 100,
        "yes" if bloat_detected else "no",
        bloat_action or "none",
        "hit" if sla_exceeded else "ok",
    )

    return {
        "output_path": str(verified_parts[0]),
        "output_paths": [str(path) for path in verified_parts],
        "original_size_mb": round(original_size_mb, 2),
        "compressed_size_mb": round(combined_mb, 2),
        "reduction_percent": round(reduction_percent, 1),
        "compression_method": "ghostscript_parallel",
        "compression_mode": compression_mode,
        "quality_mode": "aggressive_72dpi" if compression_mode == "aggressive" else "lossless",
        "was_split": len(verified_parts) > 1,
        "total_parts": len(verified_parts),
        "success": True,
        "page_count": total_pages,
        "part_sizes": [part.stat().st_size for part in verified_parts],
        "parallel_chunks": len(chunk_paths),
        "split_inflation": split_inflation,
        "split_inflation_pct": round((split_inflation_ratio - 1) * 100, 1),
        "dedupe_parts": False,
        "merge_fallback": merge_failed,
        "merge_fallback_time": 0.0,
        "rebalance_attempted": False,
        "rebalance_applied": False,
        "rebalance_reason": None,
        "rebalance_parts_before": len(verified_parts),
        "rebalance_parts_after": len(verified_parts),
        "rebalance_size_before_mb": round(combined_mb, 2),
        "rebalance_size_after_mb": round(combined_mb, 2),
        "rebalance_time": 0.0,
        "bloat_detected": bloat_detected,
        "bloat_pct": bloat_pct,
        "bloat_action": bloat_action,
        "sla_exceeded": sla_exceeded,
        "early_terminated": False,
        "probe_bailout": False,
        "probe_bailout_reason": None,
        "timings": timings_dict,
        "processing_profile": processing_profile or {},
        "profile_tier": profile_tier,
    }


def run_micro_probe(
    input_path: Path,
    compression_mode: str,
    max_pages: int = 3,
) -> Dict:
    """Compatibility micro-probe helper (lightweight, advisory only)."""
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
        with open(input_path, "rb") as handle:
            reader = PdfReader(handle, strict=False)
            pages = min(max_pages, len(reader.pages))
            writer = PdfWriter()
            for idx in range(pages):
                writer.add_page(reader.pages[idx])
            with open(sample_path, "wb") as out_handle:
                writer.write(out_handle)

        in_mb = sample_path.stat().st_size / (1024 * 1024)
        result["in_mb"] = in_mb
        result["pages"] = pages

        if compression_mode == "lossless":
            ok, msg = compress_pdf_lossless(sample_path, sample_out, timeout_override=PROBE_TIME_BUDGET_SEC)
        else:
            ok, msg = compress_pdf_with_ghostscript(sample_path, sample_out, timeout_override=PROBE_TIME_BUDGET_SEC)

        result["message"] = msg
        result["elapsed"] = time.time() - t0

        if ok and sample_out.exists():
            out_mb = sample_out.stat().st_size / (1024 * 1024)
            result["out_mb"] = out_mb
            if in_mb > 0:
                result["delta_pct"] = ((out_mb - in_mb) / in_mb) * 100
            result["success"] = True
    except Exception as exc:
        result["message"] = str(exc)
    finally:
        sample_path.unlink(missing_ok=True)
        sample_out.unlink(missing_ok=True)
        result["elapsed"] = time.time() - t0

    return result
