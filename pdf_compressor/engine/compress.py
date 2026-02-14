"""PDF compression routing using simple serial/parallel paths."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from PyPDF2 import PdfReader

import pdf_compressor.engine.ghostscript as compress_ghostscript
import pdf_compressor.engine.split as split_pdf
from pdf_compressor.core.exceptions import (
    EncryptionError,
    MetadataCorruptionError,
    ProcessingTimeoutError,
    SplitError,
    StructureError,
)
from pdf_compressor.engine.pdf_diagnostics import (
    detect_already_compressed,
    detect_scanned_document,
    estimate_processing_profile,
    fingerprint_pdf,
)
from pdf_compressor.core.settings import resolve_parallel_compute_plan as resolve_parallel_compute_plan_settings
from pdf_compressor.core.utils import env_bool, env_float, get_effective_cpu_count, get_file_size_mb

logger = logging.getLogger(__name__)

SERIAL_THRESHOLD_MB = env_float("SERIAL_THRESHOLD_MB", 50.0)
PARALLEL_THRESHOLD_MB = SERIAL_THRESHOLD_MB
MIN_COMPRESSION_SIZE_MB = env_float("MIN_COMPRESSION_SIZE_MB", 1.0)
DEFAULT_TARGET_CHUNK_MB = env_float("TARGET_CHUNK_MB", 30.0)
DEFAULT_MAX_CHUNK_MB = env_float("MAX_CHUNK_MB", 50.0)

COMPRESSION_MODE = os.environ.get("COMPRESSION_MODE", "aggressive").strip().lower()
ALLOW_LOSSY_COMPRESSION = env_bool("ALLOW_LOSSY_COMPRESSION", True)
PDF_PRECHECK_ENABLED = env_bool("PDF_PRECHECK_ENABLED", True)


def _resolve_parallel_workers(effective_cpu: Optional[int] = None) -> Tuple[int, Optional[int]]:
    """Resolve parallel worker count with CPU/thread safety."""
    plan = resolve_parallel_compute_plan(effective_cpu=effective_cpu)
    env_workers = plan.get("env_parallel_workers")
    return int(plan["effective_parallel_workers"]), int(env_workers) if isinstance(env_workers, int) else None


def resolve_parallel_compute_plan(effective_cpu: Optional[int] = None) -> Dict[str, Any]:
    """Compatibility wrapper around centralized settings planner."""
    return resolve_parallel_compute_plan_settings(effective_cpu=effective_cpu)


def resolve_compression_mode() -> str:
    """Resolve compression mode with a reliability-first policy."""
    mode = COMPRESSION_MODE
    if mode not in ("aggressive", "lossless", "adaptive"):
        mode = "aggressive"

    if mode == "adaptive":
        mode = "aggressive" if ALLOW_LOSSY_COMPRESSION else "lossless"

    if mode == "aggressive" and not ALLOW_LOSSY_COMPRESSION:
        mode = "lossless"

    return mode


def _map_compression_error(input_name: str, message: str) -> Exception:
    msg_lower = (message or "").lower()
    if "timeout" in msg_lower:
        return ProcessingTimeoutError.for_file(input_name)
    if "password" in msg_lower or "encrypt" in msg_lower or "locked" in msg_lower:
        return EncryptionError.for_file(input_name)
    if "metadata" in msg_lower or "internal data" in msg_lower:
        return MetadataCorruptionError.for_file(input_name)
    return StructureError.for_file(input_name, message)


def compress_pdf(
    input_path: str,
    working_dir: Optional[Path] = None,
    split_threshold_mb: Optional[float] = None,
    split_trigger_mb: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
) -> Dict:
    """Compress a PDF using a simple serial/parallel routing model."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    def report_progress(percent: int, stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(percent, stage, message)

    page_count = None
    if PDF_PRECHECK_ENABLED:
        try:
            with open(input_path, "rb") as handle:
                reader = PdfReader(handle, strict=False)
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        pass
                page_count = len(reader.pages)
        except Exception as exc:
            logger.warning("PDF pre-validation warning (continuing): %s", exc)

    working_dir = working_dir or input_path.parent
    original_size = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size

    compression_mode = resolve_compression_mode()
    effective_split_trigger = split_trigger_mb if split_trigger_mb is not None else split_threshold_mb
    split_requested = split_threshold_mb is not None and split_threshold_mb > 0

    bytes_per_page = (original_bytes / max(1, page_count)) if page_count else None

    already_compressed, already_reason = detect_already_compressed(input_path)
    scanned, scan_confidence = detect_scanned_document(input_path)

    fingerprint = fingerprint_pdf(input_path)
    processing_profile = estimate_processing_profile(fingerprint, original_size)

    logger.info(
        "[DIAG] === JOB START === file=%s size_mb=%.2f pages=%s mode=%s split=%s threshold=%s profile=%s",
        input_path.name,
        original_size,
        page_count,
        compression_mode,
        split_requested,
        split_threshold_mb,
        processing_profile,
    )

    quality_mode = "aggressive_72dpi" if compression_mode == "aggressive" else "lossless"

    probe_result = None
    probe_bad = False
    probe_action = None
    PROBE_SKIP_THRESHOLD_PCT = -5.0

    def _augment(response: Dict) -> Dict:
        response.setdefault("quality_mode", quality_mode)
        response.setdefault(
            "analysis",
            {
                "page_count": page_count,
                "bytes_per_page": bytes_per_page,
                "probe": probe_result,
                "probe_bad": probe_bad,
                "probe_action": probe_action,
                "composition": {
                    "already_compressed": already_compressed,
                    "already_reason": already_reason,
                    "scanned": scanned,
                    "scan_confidence": scan_confidence,
                },
                "processing_profile": processing_profile,
            },
        )
        return response

    if original_size < MIN_COMPRESSION_SIZE_MB:
        output_paths = [input_path]
        if split_requested and effective_split_trigger and original_size > effective_split_trigger:
            output_paths = split_pdf.split_for_delivery(
                input_path,
                working_dir,
                input_path.stem,
                threshold_mb=split_threshold_mb,
                progress_callback=progress_callback,
                skip_optimization_under_threshold=True,
            )
        return _augment(
            {
                "output_path": str(output_paths[0]),
                "output_paths": [str(path) for path in output_paths],
                "original_size_mb": round(original_size, 2),
                "compressed_size_mb": round(original_size, 2),
                "reduction_percent": 0.0,
                "compression_method": "none",
                "compression_mode": compression_mode,
                "was_split": len(output_paths) > 1,
                "total_parts": len(output_paths),
                "success": True,
                "note": "File under minimum compression threshold",
                "page_count": page_count,
                "part_sizes": [path.stat().st_size for path in output_paths],
                "probe_bailout": False,
                "probe_bailout_reason": None,
            }
        )

    def run_parallel(reason: str) -> Dict:
        logger.info("[PARALLEL] Routing reason=%s", reason)
        effective_cpu = get_effective_cpu_count()
        plan = resolve_parallel_compute_plan(effective_cpu=effective_cpu)
        max_workers, env_workers = _resolve_parallel_workers(effective_cpu)
        logger.info(
            "[PARALLEL] Worker cap max_workers=%s cpu=%s env=%s async=%s gs_threads=%s total_budget=%s per_job_budget=%s enforce=%s capped=%s",
            max_workers,
            effective_cpu,
            env_workers if env_workers is not None else "unset",
            plan.get("async_workers"),
            plan.get("gs_threads_per_worker"),
            plan.get("total_parallel_budget"),
            plan.get("per_job_parallel_budget"),
            "yes" if plan.get("parallel_budget_enforce") else "no",
            "yes" if plan.get("capped_by_budget") else "no",
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
            target_chunk_mb=DEFAULT_TARGET_CHUNK_MB,
            max_chunk_mb=DEFAULT_MAX_CHUNK_MB,
            input_page_count=page_count,
            processing_profile=processing_profile,
        )

    use_parallel = original_size > SERIAL_THRESHOLD_MB

    # Fast incompressibility gate: probe 3 pages before committing to parallel
    if use_parallel:
        try:
            probe_result = compress_ghostscript.run_micro_probe(input_path, compression_mode)
            if probe_result.get("success"):
                delta = probe_result.get("delta_pct", 0.0)
                logger.info(
                    "[PROBE] file=%s delta_pct=%.1f threshold=%.1f pages=%s elapsed=%.1fs",
                    input_path.name,
                    delta,
                    PROBE_SKIP_THRESHOLD_PCT,
                    probe_result.get("pages"),
                    probe_result.get("elapsed", 0),
                )
                if delta >= PROBE_SKIP_THRESHOLD_PCT:
                    probe_bad = True
                    probe_action = "continue_parallel"
                    logger.warning(
                        "[PROBE] Incompressible sample (delta %.1f%% >= %.1f%%) — advisory only, continuing to parallel",
                        delta,
                        PROBE_SKIP_THRESHOLD_PCT,
                    )
            else:
                logger.warning("[PROBE] Probe failed (%s); continuing to parallel", probe_result.get("message"))
        except Exception as exc:
            logger.warning("[PROBE] Probe error (%s); continuing to parallel", exc)

    if use_parallel:
        return _augment(run_parallel("size"))

    report_progress(15, "compressing", "Compressing PDF...")

    output_path = working_dir / f"{input_path.stem}_compressed.pdf"
    serial_timeout = int(max(120, processing_profile.get("chunk_timeout_sec", 120)))

    if compression_mode == "lossless":
        primary_fn = compress_ghostscript.compress_pdf_lossless
    else:
        primary_fn = compress_ghostscript.compress_pdf_with_ghostscript

    success, message = primary_fn(input_path, output_path, timeout_override=serial_timeout)

    if not success:
        if "timeout" in (message or "").lower():
            logger.warning("[compress] Serial timed out; switching to parallel")
            return _augment(run_parallel("timeout"))
        raise _map_compression_error(input_path.name, message)

    if not output_path.exists():
        raise StructureError.for_file(input_path.name, "Compression output not created")

    compressed_size = get_file_size_mb(output_path)

    if compressed_size >= original_size:
        logger.info("[compress] Primary mode gave no reduction; trying alternate mode once")
        output_path.unlink(missing_ok=True)

        alt_ok = False
        alt_message = ""
        if compression_mode == "aggressive":
            alt_ok, alt_message = compress_ghostscript.compress_pdf_lossless(
                input_path,
                output_path,
                timeout_override=serial_timeout,
            )
            if alt_ok:
                quality_mode = "lossless"
                compression_mode = "lossless"
        elif ALLOW_LOSSY_COMPRESSION:
            alt_ok, alt_message = compress_ghostscript.compress_pdf_with_ghostscript(
                input_path,
                output_path,
                timeout_override=serial_timeout,
            )
            if alt_ok:
                quality_mode = "aggressive_72dpi"
                compression_mode = "aggressive"

        if not alt_ok or not output_path.exists():
            logger.warning("[compress] Alternate mode failed (%s); returning original", alt_message)
            shutil.copy2(input_path, output_path)
            compressed_size = original_size
        else:
            compressed_size = get_file_size_mb(output_path)
            if compressed_size >= original_size:
                logger.info("[compress] Alternate mode also no reduction; returning original")
                output_path.unlink(missing_ok=True)
                shutil.copy2(input_path, output_path)
                compressed_size = original_size

    reduction = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0.0

    output_paths = [output_path]
    if split_requested and effective_split_trigger and compressed_size > effective_split_trigger:
        report_progress(70, "splitting", "Splitting output...")
        output_paths = split_pdf.split_for_delivery(
            output_path,
            working_dir,
            input_path.stem,
            threshold_mb=split_threshold_mb,
            progress_callback=progress_callback,
            skip_optimization_under_threshold=True,
        )

    report_progress(95, "finalizing", "Finalizing...")

    method = "lossless" if compression_mode == "lossless" else "ghostscript"
    if compressed_size >= original_size:
        method = "none"
    combined_mb = sum(path.stat().st_size for path in output_paths) / (1024 * 1024)
    final_reduction = ((original_size - combined_mb) / original_size) * 100 if original_size > 0 else 0.0

    return _augment(
        {
            "output_path": str(output_paths[0]),
            "output_paths": [str(path) for path in output_paths],
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(combined_mb, 2),
            "reduction_percent": round(final_reduction, 1),
            "compression_method": method,
            "compression_mode": compression_mode,
            "was_split": len(output_paths) > 1,
            "total_parts": len(output_paths),
            "success": True,
            "page_count": page_count,
            "part_sizes": [path.stat().st_size for path in output_paths],
            "split_inflation": False,
            "split_inflation_pct": 0.0,
            "dedupe_parts": False,
            "merge_fallback": False,
            "merge_fallback_time": 0.0,
            "rebalance_attempted": False,
            "rebalance_applied": False,
            "bloat_detected": compressed_size >= original_size,
            "bloat_pct": 0.0,
            "bloat_action": "return_original" if compressed_size >= original_size else "none",
            "sla_exceeded": False,
            "probe_bailout": False,
            "probe_bailout_reason": None,
            "processing_profile": processing_profile,
            "profile_tier": processing_profile.get("tier"),
        }
    )
