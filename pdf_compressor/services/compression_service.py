"""Flask API for PDF compression with async processing and optional splitting."""

import base64
import contextlib
import hashlib
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote, urlparse, parse_qs, parse_qsl, urlencode

import requests

from flask import jsonify, request, render_template, send_file, has_request_context
from werkzeug.exceptions import RequestEntityTooLarge, HTTPException, NotFound
from werkzeug.utils import secure_filename

from pdf_compressor.engine.compress import (
    compress_pdf,
    PARALLEL_THRESHOLD_MB,
)
from pdf_compressor.engine.ghostscript import get_ghostscript_command
from pdf_compressor.engine.pdf_diagnostics import diagnose_for_job, get_quality_warnings, fingerprint_pdf
from pdf_compressor.core.exceptions import (
    PDFCompressionError,
    EncryptionError,
    StructureError,
    MetadataCorruptionError,
    SplitError,
    DownloadError,
    ProcessingTimeoutError,
)
import pdf_compressor.workers.job_queue as job_queue
from pdf_compressor.core.settings import resolve_parallel_compute_plan
import pdf_compressor.core.utils as utils
from pdf_compressor.core.utils import MAX_DOWNLOAD_SIZE

# Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Constants
MAX_CONTENT_LENGTH = MAX_DOWNLOAD_SIZE
HARD_ASYNC_LIMIT_MB = 600.0  # Absolute ceiling for async jobs
PDF_FINGERPRINT_ENABLED = True
PDF_FINGERPRINT_MAX_PAGES = 5


def resolve_upload_folder() -> Path:
    """Resolve the upload folder, preferring persistent storage on Azure."""
    env_override = os.environ.get("UPLOAD_FOLDER")
    if env_override:
        override_path = Path(env_override)
        # Ignore unsafe Azure overrides that point at ephemeral /root.
        if (os.environ.get("WEBSITE_INSTANCE_ID") or os.environ.get("WEBSITE_SITE_NAME")) and str(override_path).startswith("/root"):
            logger.warning("UPLOAD_FOLDER=%s is not persistent on Azure; falling back to /home", override_path)
        else:
            return override_path

    # Prefer Azure's persistent /home, even if HOME is misconfigured.
    for base in (Path("/home/site/wwwroot"), Path("/home")):
        try:
            if base.exists() and os.access(base, os.W_OK):
                return base / "uploads"
        except OSError:
            continue

    # Azure App Service: /home is persistent and shared across instances.
    if os.environ.get("WEBSITE_INSTANCE_ID") or os.environ.get("WEBSITE_SITE_NAME"):
        home = Path(os.environ.get("HOME", "/home"))
        site_root = home / "site" / "wwwroot"
        if site_root.exists():
            return site_root / "uploads"
        return home / "uploads"

    # Keep local fallback consistent with the previous root-level app.py behavior.
    return Path(__file__).resolve().parents[2] / "uploads"


UPLOAD_FOLDER = resolve_upload_folder()
FILE_RETENTION_SECONDS = int(os.environ.get('FILE_RETENTION_SECONDS', '86400'))  # 24 hours
MIN_FILE_RETENTION_SECONDS = int(os.environ.get('MIN_FILE_RETENTION_SECONDS', '3600'))  # 1 hour minimum
EFFECTIVE_FILE_RETENTION_SECONDS = max(FILE_RETENTION_SECONDS, MIN_FILE_RETENTION_SECONDS)
BASE_SPLIT_THRESHOLD_MB = float(os.environ.get('SPLIT_THRESHOLD_MB', '25'))
SPLIT_THRESHOLD_MB = BASE_SPLIT_THRESHOLD_MB
# Split only when output exceeds this size, but keep parts under SPLIT_THRESHOLD_MB.
_split_trigger = float(os.environ.get("SPLIT_TRIGGER_MB", str(SPLIT_THRESHOLD_MB)))
SPLIT_TRIGGER_MB = max(SPLIT_THRESHOLD_MB, _split_trigger)
# Async jobs can be large, but still need a hard ceiling to protect the instance.
ASYNC_MAX_MB = min(float(os.environ.get("ASYNC_MAX_MB", "450")), HARD_ASYNC_LIMIT_MB)
API_TOKEN = os.environ.get('API_TOKEN')
# Public base URL for download links (e.g., https://yourapp.azurewebsites.net)
BASE_URL = os.environ.get('BASE_URL', '').rstrip('/')
_EFFECTIVE_CPU = utils.get_effective_cpu_count()
ASYNC_WORKERS = int(resolve_parallel_compute_plan(_EFFECTIVE_CPU)["async_workers"])
MAX_ACTIVE_COMPRESSIONS = max(1, min(ASYNC_WORKERS, _EFFECTIVE_CPU))
SMALL_JOB_THRESHOLD_MB = 20.0
SMALL_JOB_RESERVED_SLOTS = 1 if MAX_ACTIVE_COMPRESSIONS > 1 else 0
GENERAL_COMPRESSION_SLOTS = max(1, MAX_ACTIVE_COMPRESSIONS - SMALL_JOB_RESERVED_SLOTS)
COMPRESSION_SEMAPHORE = threading.Semaphore(GENERAL_COMPRESSION_SLOTS)
SMALL_JOB_SEMAPHORE = threading.Semaphore(SMALL_JOB_RESERVED_SLOTS) if SMALL_JOB_RESERVED_SLOTS else None
DEDUPE_INFLIGHT_ENABLED = utils.env_bool("DEDUPE_INFLIGHT_ENABLED", True)
DEDUPE_URL_ENABLED = utils.env_bool("DEDUPE_URL_ENABLED", True)
DEDUPE_MIN_MB = max(PARALLEL_THRESHOLD_MB, utils.env_float("DEDUPE_MIN_MB", PARALLEL_THRESHOLD_MB))

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
logger.info(
    "Upload folder resolved to: %s (retention=%ss, min=%ss)",
    UPLOAD_FOLDER.resolve(),
    FILE_RETENTION_SECONDS,
    MIN_FILE_RETENTION_SECONDS,
)


def _format_env_value(name: str, effective: Any, default_label: str = "default") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return f"{effective} ({default_label})"
    return f"{effective} (env:{raw})"


def _format_const_value(value: Any) -> str:
    return f"{value} (const)"


def _truncate(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


_BOX_LABEL_WIDTH = 30
_BOX_VALUE_WIDTH = 34
_BOX_INNER_WIDTH = _BOX_LABEL_WIDTH + _BOX_VALUE_WIDTH + 5


def _banner(title: str) -> list[str]:
    title_text = f"[ {title} ]"
    border = "+" + "=" * _BOX_INNER_WIDTH + "+"
    return [border, f"|{title_text:^{_BOX_INNER_WIDTH}}|", border]


def _box(title: str, rows: list[tuple[str, str]]) -> list[str]:
    title_text = f" {title} "
    title_border = "+" + "=" * _BOX_INNER_WIDTH + "+"
    row_border = (
        "+"
        + "-" * (_BOX_LABEL_WIDTH + 2)
        + "+"
        + "-" * (_BOX_VALUE_WIDTH + 2)
        + "+"
    )
    lines = [title_border, f"|{title_text:^{_BOX_INNER_WIDTH}}|", title_border]
    lines.append(row_border)
    for label, value in rows:
        safe_label = _truncate(label, _BOX_LABEL_WIDTH)
        safe_value = _truncate(value, _BOX_VALUE_WIDTH)
        lines.append(f"| {safe_label:<{_BOX_LABEL_WIDTH}} | {safe_value:<{_BOX_VALUE_WIDTH}} |")
    lines.append(row_border)
    return lines


def _format_split_label(split_requested: bool, split_threshold_mb: float | None, origin: str) -> str:
    if not split_requested or split_threshold_mb is None:
        return "off"
    suffix = origin or "request"
    return f"{split_threshold_mb:.1f}MB ({suffix})"


def _log_effective_config() -> None:
    try:
        import pdf_compressor.engine.ghostscript as gs_cfg
        import pdf_compressor.engine.split as split_cfg
    except Exception as exc:
        logger.warning("Config snapshot skipped: %s", exc)
        return

    async_raw = os.environ.get("ASYNC_WORKERS")
    async_value = ASYNC_WORKERS
    async_display = f"{async_value} (env:{async_raw})" if async_raw else f"{async_value} (auto)"

    plan = resolve_parallel_compute_plan(_EFFECTIVE_CPU)
    parallel_display = (
        f"{plan['effective_parallel_workers']} "
        f"(requested:{plan['requested_parallel_workers']})"
    )
    gs_threads_display = str(plan["gs_threads_per_worker"])

    rows_concurrency = [
        ("ASYNC_WORKERS", async_display),
        ("MAX_ACTIVE_COMPRESSIONS", str(MAX_ACTIVE_COMPRESSIONS)),
        ("PARALLEL_MAX_WORKERS", parallel_display),
        ("GS_NUM_RENDERING_THREADS", gs_threads_display),
        ("PARALLEL_TOTAL_BUDGET", str(plan["total_parallel_budget"])),
        ("PARALLEL_PER_JOB_BUDGET", str(plan["per_job_parallel_budget"])),
        ("PARALLEL_BUDGET_ENFORCE", _format_env_value("PARALLEL_BUDGET_ENFORCE", plan["parallel_budget_enforce"])),
        ("PARALLEL_CAPPED_BY_BUDGET", "yes" if plan["capped_by_budget"] else "no"),
        ("PARALLEL_CAP_REASONS", ",".join(plan.get("capped_reasons", [])) or "none"),
    ]

    rows_chunking = [
        ("TARGET_CHUNK_MB", _format_env_value("TARGET_CHUNK_MB", gs_cfg.TARGET_CHUNK_MB)),
        ("MAX_CHUNK_MB", _format_env_value("MAX_CHUNK_MB", gs_cfg.MAX_CHUNK_MB)),
        ("MAX_PARALLEL_CHUNKS", _format_env_value("MAX_PARALLEL_CHUNKS", gs_cfg.MAX_PARALLEL_CHUNKS)),
        ("MAX_PAGES_PER_CHUNK", _format_env_value("MAX_PAGES_PER_CHUNK", gs_cfg.MAX_PAGES_PER_CHUNK)),
        ("SLA_MAX_PARALLEL_CHUNKS", _format_env_value("SLA_MAX_PARALLEL_CHUNKS", gs_cfg.SLA_MAX_PARALLEL_CHUNKS)),
        ("SLA_MAX_PARALLEL_CHUNKS_LARGE", _format_env_value("SLA_MAX_PARALLEL_CHUNKS_LARGE", gs_cfg.SLA_MAX_PARALLEL_CHUNKS_LARGE)),
        ("HARD_MAX_PARALLEL_CHUNKS", _format_env_value("HARD_MAX_PARALLEL_CHUNKS", gs_cfg.HARD_MAX_PARALLEL_CHUNKS)),
    ]

    rows_split = [
        ("SPLIT_THRESHOLD_MB", _format_env_value("SPLIT_THRESHOLD_MB", SPLIT_THRESHOLD_MB)),
        ("SPLIT_TRIGGER_MB", _format_env_value("SPLIT_TRIGGER_MB", SPLIT_TRIGGER_MB)),
        ("SPLIT_MINIMIZE_PARTS", _format_env_value("SPLIT_MINIMIZE_PARTS", split_cfg.SPLIT_MINIMIZE_PARTS)),
        ("SPLIT_ENABLE_BINARY_FALLBACK", _format_env_value("SPLIT_ENABLE_BINARY_FALLBACK", split_cfg.SPLIT_ENABLE_BINARY_FALLBACK)),
        ("SPLIT_ADAPTIVE_MAX_ATTEMPTS", _format_env_value("SPLIT_ADAPTIVE_MAX_ATTEMPTS", split_cfg.SPLIT_ADAPTIVE_MAX_ATTEMPTS)),
        ("MERGE_TIMEOUT_SEC", _format_env_value("MERGE_TIMEOUT_SEC", split_cfg.MERGE_TIMEOUT_SEC)),
        ("MERGE_FALLBACK_TIMEOUT_SEC", _format_env_value("MERGE_FALLBACK_TIMEOUT_SEC", split_cfg.MERGE_FALLBACK_TIMEOUT_SEC)),
    ]

    rows_quality = [
        ("COMPRESSION_MODE", _format_env_value("COMPRESSION_MODE", os.environ.get("COMPRESSION_MODE", "aggressive"))),
        ("ALLOW_LOSSY_COMPRESSION", _format_env_value("ALLOW_LOSSY_COMPRESSION", utils.env_bool("ALLOW_LOSSY_COMPRESSION", True))),
        ("GS_COLOR_IMAGE_RESOLUTION", _format_env_value("GS_COLOR_IMAGE_RESOLUTION", utils.env_int("GS_COLOR_IMAGE_RESOLUTION", 72))),
        ("GS_GRAY_IMAGE_RESOLUTION", _format_env_value("GS_GRAY_IMAGE_RESOLUTION", utils.env_int("GS_GRAY_IMAGE_RESOLUTION", 72))),
        ("GS_MONO_IMAGE_RESOLUTION", _format_env_value("GS_MONO_IMAGE_RESOLUTION", gs_cfg.GS_MONO_IMAGE_RESOLUTION)),
        ("GS_COLOR_DOWNSAMPLE_TYPE", _format_env_value("GS_COLOR_DOWNSAMPLE_TYPE", gs_cfg.GS_COLOR_DOWNSAMPLE_TYPE)),
        ("GS_GRAY_DOWNSAMPLE_TYPE", _format_env_value("GS_GRAY_DOWNSAMPLE_TYPE", gs_cfg.GS_GRAY_DOWNSAMPLE_TYPE)),
        ("PARALLEL_JOB_SLA_LARGE_MIN_MB", _format_env_value("PARALLEL_JOB_SLA_LARGE_MIN_MB", gs_cfg.PARALLEL_JOB_SLA_LARGE_MIN_MB)),
        ("PARALLEL_JOB_SLA_LARGE_SEC_PER_MB", _format_env_value("PARALLEL_JOB_SLA_LARGE_SEC_PER_MB", gs_cfg.PARALLEL_JOB_SLA_LARGE_SEC_PER_MB)),
        ("PARALLEL_JOB_SLA_MAX_SEC", _format_env_value("PARALLEL_JOB_SLA_MAX_SEC", gs_cfg.PARALLEL_JOB_SLA_MAX_SEC)),
        ("PDF_FINGERPRINT_ENABLED", _format_const_value(PDF_FINGERPRINT_ENABLED)),
        ("PDF_FINGERPRINT_MAX_PAGES", _format_const_value(PDF_FINGERPRINT_MAX_PAGES)),
    ]

    lines = _banner("CONFIG SNAPSHOT (startup)")
    lines += _box("Concurrency", rows_concurrency)
    lines += _box("Chunking", rows_chunking)
    lines += _box("Split and Merge", rows_split)
    lines += _box("Quality", rows_quality)
    logger.info("\n%s", "\n".join(lines))


_log_effective_config()


def configure_app(app) -> None:
    """Apply Flask app config values required by this service layer."""
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def register_error_handlers(app) -> None:
    """Register HTTP and framework error handlers."""
    app.register_error_handler(RequestEntityTooLarge, handle_large_file)
    app.register_error_handler(HTTPException, handle_http_exception)
    app.register_error_handler(Exception, handle_error)


def require_auth(f):
    """Decorator to require Bearer token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)  # No token configured = open access

        auth_header = request.headers.get('Authorization')

        # Debug logging to diagnose auth issues
        logger.debug(f"Auth check on {request.path}: header={'present' if auth_header else 'missing'}, content_type={request.content_type}")

        if not auth_header:
            logger.warning(f"Missing Authorization header on {request.path} (content_type: {request.content_type})")
            return jsonify({"success": False, "error": "Missing Authorization header"}), 401

        if not auth_header.startswith('Bearer '):
            logger.warning(f"Invalid Authorization format on {request.path}: {auth_header[:30]}...")
            return jsonify({"success": False, "error": "Authorization must use Bearer token format"}), 401

        if auth_header[7:] != API_TOKEN:
            logger.warning(f"Invalid token on {request.path}")
            return jsonify({"success": False, "error": "Invalid token"}), 403

        return f(*args, **kwargs)
    return decorated


def build_download_url(path_str: str) -> str:
    """
    Build a public download URL from a local path or filename.

    Keeps backward compatibility: if BASE_URL is not set, returns a relative /download path.
    """
    name = Path(path_str).name
    rel = f"/download/{name}"
    base = BASE_URL
    if not base and has_request_context():
        base = request.url_root.rstrip('/')

    # Normalize to HTTPS to avoid mixed-content when the page is served over TLS
    if base and base.startswith("http://"):
        base = "https://" + base[len("http://"):]

    return f"{base}{rel}" if base else rel


def _normalize_display_base(name_hint: str | None) -> str:
    if not name_hint:
        return "document"
    raw = name_hint.strip()
    if not raw:
        return "document"
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        candidate = parsed.path or ""
        filename = Path(candidate).name
        if not filename or not Path(filename).suffix:
            return "document"
    else:
        candidate = parsed.path if parsed.path else raw
        filename = Path(candidate).name
    stem = Path(filename).stem if filename else ""
    safe_stem = secure_filename(stem)
    return safe_stem or "document"


def _build_display_filename(base_name: str, part_idx: int, total_parts: int) -> str:
    if total_parts > 1:
        return f"{base_name}_compressed_part{part_idx}.pdf"
    return f"{base_name}_compressed.pdf"


def _build_download_links(output_paths: list[Path], name_hint: str | None) -> list[str]:
    if not output_paths:
        return []
    base_name = _normalize_display_base(name_hint)
    total_parts = len(output_paths)
    links: list[str] = []
    for idx, path in enumerate(output_paths, start=1):
        display_name = _build_display_filename(base_name, idx, total_parts)
        display_name = secure_filename(display_name)
        if not display_name:
            display_name = (
                f"document_compressed_part{idx}.pdf"
                if total_parts > 1
                else "document_compressed.pdf"
            )
        if not display_name.lower().endswith(".pdf"):
            display_name = f"{display_name}.pdf"
        links.append(f"/download/{path.name}?name={quote(display_name)}")
    return links


def _log_job_header(
    job_id: str,
    source: str,
    input_path: Path,
    name_hint: str | None,
    split_threshold_mb: float | None = None,
    split_trigger_mb: float | None = None,
) -> None:
    try:
        size_mb = input_path.stat().st_size / (1024 * 1024)
        size_label = f"{size_mb:.1f}MB"
    except OSError:
        size_label = "unknown"

    name_label = _normalize_display_base(name_hint)

    threshold_mb = SPLIT_THRESHOLD_MB if split_threshold_mb is None else split_threshold_mb
    trigger_mb = SPLIT_TRIGGER_MB if split_trigger_mb is None else split_trigger_mb
    split_label = "off" if split_threshold_mb is None else f"{threshold_mb:.1f}MB"
    trigger_label = "off" if split_trigger_mb is None else f"{trigger_mb:.1f}MB"

    logger.info(
        "[JOB] %s | source=%s | name=%s | file=%s | size=%s | split=%s | trigger=%s",
        job_id,
        source,
        name_label,
        input_path.name,
        size_label,
        split_label,
        trigger_label,
    )


def _log_job_snapshot(
    job_id: str,
    source: str,
    input_path: Path,
    name_hint: str | None,
    split_threshold_mb: float | None,
    split_trigger_mb: float | None,
    split_origin: str,
) -> None:
    try:
        import pdf_compressor.engine.ghostscript as gs_cfg
        import pdf_compressor.engine.split as split_cfg
    except Exception as exc:
        logger.warning("[%s] Job snapshot skipped: %s", job_id, exc)
        return

    try:
        size_mb = input_path.stat().st_size / (1024 * 1024)
        size_label = f"{size_mb:.1f}MB"
    except OSError:
        size_label = "unknown"

    name_label = _normalize_display_base(name_hint)
    split_requested = split_threshold_mb is not None
    split_label = _format_split_label(split_requested, split_threshold_mb, split_origin)
    trigger_label = "off"
    if split_trigger_mb is not None:
        trigger_label = f"{split_trigger_mb:.1f}MB (effective)"

    plan = resolve_parallel_compute_plan(_EFFECTIVE_CPU)
    parallel_workers = (
        f"{plan['effective_parallel_workers']} "
        f"(requested:{plan['requested_parallel_workers']})"
    )
    gs_threads = str(plan["gs_threads_per_worker"])

    rows_job = [
        ("JOB_ID", job_id),
        ("SOURCE", source),
        ("INPUT_FILE", input_path.name),
        ("INPUT_SIZE", size_label),
        ("NAME_HINT", name_label),
        ("ASYNC_MAX_MB", f"{ASYNC_MAX_MB:.0f}MB"),
    ]

    rows_split = [
        ("SPLIT_REQUESTED", "yes" if split_requested else "no"),
        ("SPLIT_THRESHOLD_MB", split_label),
        ("SPLIT_TRIGGER_MB", trigger_label),
        ("SPLIT_MINIMIZE_PARTS", _format_const_value(split_cfg.SPLIT_MINIMIZE_PARTS)),
        ("MERGE_TIMEOUT_SEC", _format_env_value("MERGE_TIMEOUT_SEC", split_cfg.MERGE_TIMEOUT_SEC)),
        ("MERGE_FALLBACK_TIMEOUT", _format_env_value("MERGE_FALLBACK_TIMEOUT_SEC", split_cfg.MERGE_FALLBACK_TIMEOUT_SEC)),
    ]

    rows_parallel = [
        ("PARALLEL_MAX_WORKERS", parallel_workers),
        ("GS_NUM_RENDERING_THREADS", gs_threads),
        ("PARALLEL_TOTAL_BUDGET", str(plan["total_parallel_budget"])),
        ("PARALLEL_PER_JOB_BUDGET", str(plan["per_job_parallel_budget"])),
        ("PARALLEL_BUDGET_ENFORCE", "yes" if plan["parallel_budget_enforce"] else "no"),
        ("PARALLEL_CAPPED_BY_BUDGET", "yes" if plan["capped_by_budget"] else "no"),
        ("PARALLEL_CAP_REASONS", ",".join(plan.get("capped_reasons", [])) or "none"),
        ("TARGET_CHUNK_MB", _format_env_value("TARGET_CHUNK_MB", f"{gs_cfg.TARGET_CHUNK_MB:.1f}MB")),
        ("MAX_CHUNK_MB", _format_env_value("MAX_CHUNK_MB", f"{gs_cfg.MAX_CHUNK_MB:.1f}MB")),
        ("MAX_PARALLEL_CHUNKS", _format_env_value("MAX_PARALLEL_CHUNKS", gs_cfg.MAX_PARALLEL_CHUNKS)),
        ("MAX_PAGES_PER_CHUNK", _format_env_value("MAX_PAGES_PER_CHUNK", gs_cfg.MAX_PAGES_PER_CHUNK)),
        ("HARD_MAX_PARALLEL_CHUNKS", _format_env_value("HARD_MAX_PARALLEL_CHUNKS", gs_cfg.HARD_MAX_PARALLEL_CHUNKS)),
        ("LARGE_FILE_TUNE_MIN_MB", _format_env_value("LARGE_FILE_TUNE_MIN_MB", f"{gs_cfg.LARGE_FILE_TUNE_MIN_MB:.1f}MB")),
        ("LARGE_FILE_TARGET_MB", _format_env_value("LARGE_FILE_TARGET_CHUNK_MB", f"{gs_cfg.LARGE_FILE_TARGET_CHUNK_MB:.1f}MB")),
        ("LARGE_FILE_MAX_MB", _format_env_value("LARGE_FILE_MAX_CHUNK_MB", f"{gs_cfg.LARGE_FILE_MAX_CHUNK_MB:.1f}MB")),
    ]

    rows_quality = [
        ("COMPRESSION_MODE", _format_env_value("COMPRESSION_MODE", os.environ.get("COMPRESSION_MODE", "aggressive"))),
        ("ALLOW_LOSSY", _format_env_value("ALLOW_LOSSY_COMPRESSION", utils.env_bool("ALLOW_LOSSY_COMPRESSION", True))),
        ("CHUNK_TIME_MAX_SEC", str(gs_cfg.CHUNK_TIME_BUDGET_MAX_SEC_LARGE)),
        ("PARALLEL_JOB_SLA_SEC", str(gs_cfg.PARALLEL_JOB_SLA_SEC)),
        ("PARALLEL_JOB_SLA_LARGE_MIN_MB", _format_env_value("PARALLEL_JOB_SLA_LARGE_MIN_MB", gs_cfg.PARALLEL_JOB_SLA_LARGE_MIN_MB)),
        ("PARALLEL_JOB_SLA_LARGE_SEC_PER_MB", _format_env_value("PARALLEL_JOB_SLA_LARGE_SEC_PER_MB", gs_cfg.PARALLEL_JOB_SLA_LARGE_SEC_PER_MB)),
        ("PARALLEL_JOB_SLA_MAX_SEC", _format_env_value("PARALLEL_JOB_SLA_MAX_SEC", gs_cfg.PARALLEL_JOB_SLA_MAX_SEC)),
        ("PDF_FINGERPRINT_ENABLED", _format_const_value(PDF_FINGERPRINT_ENABLED)),
        ("PDF_FINGERPRINT_MAX_PAGES", _format_const_value(PDF_FINGERPRINT_MAX_PAGES)),
        ("GS_COLOR_DPI", _format_env_value("GS_COLOR_IMAGE_RESOLUTION", gs_cfg.GS_COLOR_IMAGE_RESOLUTION)),
        ("GS_GRAY_DPI", _format_env_value("GS_GRAY_IMAGE_RESOLUTION", gs_cfg.GS_GRAY_IMAGE_RESOLUTION)),
        ("GS_MONO_DPI", _format_env_value("GS_MONO_IMAGE_RESOLUTION", gs_cfg.GS_MONO_IMAGE_RESOLUTION)),
        ("GS_COLOR_DOWNSAMPLE", _format_env_value("GS_COLOR_DOWNSAMPLE_TYPE", gs_cfg.GS_COLOR_DOWNSAMPLE_TYPE)),
        ("GS_GRAY_DOWNSAMPLE", _format_env_value("GS_GRAY_DOWNSAMPLE_TYPE", gs_cfg.GS_GRAY_DOWNSAMPLE_TYPE)),
    ]

    lines = _banner(f"JOB SNAPSHOT [{job_id}]")
    lines += _box("Job", rows_job)
    lines += _box("Split and Delivery", rows_split)
    lines += _box("Parallel Plan", rows_parallel)
    lines += _box("Quality and Timeouts", rows_quality)
    logger.info("\n%s", "\n".join(lines))


def _log_pdf_fingerprint(job_id: str, input_path: Path) -> None:
    if not PDF_FINGERPRINT_ENABLED:
        return

    max_pages = PDF_FINGERPRINT_MAX_PAGES
    try:
        fp = fingerprint_pdf(input_path, max_pages=max_pages)
    except Exception as exc:
        logger.warning("[%s] PDF fingerprint failed: %s", job_id, exc)
        return

    if fp.get("error"):
        logger.warning("[%s] PDF fingerprint error: %s", job_id, fp["error"])
        return

    bytes_per_page = fp.get("bytes_per_page") or 0.0
    bytes_label = f"{bytes_per_page / 1024:.1f}KB" if bytes_per_page else "n/a"
    producer = fp.get("producer") or "n/a"
    creator = fp.get("creator") or "n/a"
    already_compressed = "yes" if fp.get("already_compressed") else "no"
    already_reason = fp.get("already_reason") or "n/a"

    filters = fp.get("image_filters") or {}
    if filters:
        filters_label = " ".join(f"{key}:{value}" for key, value in filters.items())
    else:
        filters_label = "none"

    avg_dpi = fp.get("avg_image_dpi")
    avg_dpi_label = f"{avg_dpi:.1f}" if isinstance(avg_dpi, (int, float)) else "n/a"

    verdict = f"{already_compressed} ({already_reason})"
    rows_structure = [
        ("VERDICT", verdict),
        ("PAGES", str(fp.get("page_count") or 0)),
        ("SAMPLE_PAGES", str(fp.get("sample_pages") or 0)),
        ("FILE_SIZE_MB", f"{fp.get('file_size_mb', 0.0):.1f}"),
        ("BYTES_PER_PAGE", bytes_label),
        ("PRODUCER", producer),
        ("CREATOR", creator),
    ]

    rows_media = [
        ("IMAGE_COUNT", str(fp.get("image_count") or 0)),
        ("IMAGE_FILTERS", filters_label),
        ("IMAGE_MPIXELS", f"{fp.get('image_megapixels', 0.0):.2f}"),
        ("AVG_IMAGE_DPI", avg_dpi_label),
        ("TEXT_PAGES", f"{fp.get('text_pages', 0)}/{fp.get('sample_pages', 0)}"),
        ("TEXT_CHARS", str(fp.get("text_chars") or 0)),
    ]

    lines = _banner(f"PDF FINGERPRINT [{job_id}]")
    lines += _box("Structure", rows_structure)
    lines += _box("Images + Text", rows_media)
    logger.info("\n%s", "\n".join(lines))


def _log_job_result(job_id: str, result: dict, output_paths: list[Path]) -> None:
    parts = result.get("total_parts") or len(output_paths)
    sizes = []
    for path in output_paths:
        try:
            sizes.append(f"{path.stat().st_size / (1024 * 1024):.1f}")
        except OSError:
            sizes.append("?")
    size_label = ",".join(sizes) if sizes else "none"
    page_count = result.get("page_count")
    page_label = str(page_count) if page_count is not None else "unknown"
    bloat_detected = result.get("bloat_detected")
    if bloat_detected is True:
        bloat_label = "yes"
    elif bloat_detected is False:
        bloat_label = "no"
    else:
        bloat_label = "n/a"
    bloat_pct = result.get("bloat_pct")
    bloat_pct_label = f"{bloat_pct:.1f}%" if isinstance(bloat_pct, (int, float)) else "n/a"
    bloat_action = result.get("bloat_action") or "n/a"

    logger.info(
        "[JOB_RESULT] %s | mode=%s | method=%s | pages=%s | in=%.1fMB | out=%.1fMB | "
        "reduction=%.1f%% | parts=%s | part_sizes_mb=%s | bloat=%s (%s) action=%s",
        job_id,
        result.get("compression_mode"),
        result.get("compression_method"),
        page_label,
        result.get("original_size_mb", 0.0),
        result.get("compressed_size_mb", 0.0),
        result.get("reduction_percent", 0.0),
        parts,
        size_label,
        bloat_label,
        bloat_pct_label,
        bloat_action,
    )


def _log_job_flags(
    job_id: str,
    result: dict,
    output_paths: list[Path],
    split_threshold_mb: float | None = None,
) -> None:
    if not output_paths:
        return

    part_sizes_mb = []
    for path in output_paths:
        try:
            part_sizes_mb.append(path.stat().st_size / (1024 * 1024))
        except OSError:
            continue

    if not part_sizes_mb:
        return

    total_parts = len(part_sizes_mb)
    total_mb = sum(part_sizes_mb)
    max_part_mb = max(part_sizes_mb)
    min_part_mb = min(part_sizes_mb)
    avg_part_mb = total_mb / total_parts if total_parts else 0.0

    if split_threshold_mb is None:
        logger.info(
            "[JOB_FLAGS] %s | split=off | parts=%s | part_range=%.1f-%.1fMB avg=%.1fMB",
            job_id,
            total_parts,
            min_part_mb,
            max_part_mb,
            avg_part_mb,
        )
        return

    threshold_mb = SPLIT_THRESHOLD_MB if split_threshold_mb is None else split_threshold_mb
    if threshold_mb > 0:
        min_parts = max(1, math.ceil(total_mb / threshold_mb))
        max_part_pct = (max_part_mb / threshold_mb) * 100
        slack_mb = max(0.0, threshold_mb - max_part_mb)
        slack_pct = (slack_mb / threshold_mb) * 100
    else:
        min_parts = total_parts
        max_part_pct = 0.0
        slack_mb = 0.0
        slack_pct = 0.0

    parts_over_min = total_parts - min_parts
    method = result.get("compression_method")

    split_inflation_pct = _safe_float(result.get("split_inflation_pct"))
    split_inflation_label = f"{split_inflation_pct:.1f}%" if split_inflation_pct is not None else "n/a"

    def _flag_label(value: Any) -> str:
        if value is True:
            return "yes"
        if value is False:
            return "no"
        return "n/a"

    merge_fallback = result.get("merge_fallback")
    merge_label = _flag_label(merge_fallback)

    logger.info(
        "[JOB_FLAGS] %s | method=%s | parts=%s | min_parts=%s | over_min=%s | "
        "max_part=%.1fMB(%.0f%% of %.1fMB) | slack=%.1fMB(%.0f%%) | "
        "part_range=%.1f-%.1fMB avg=%.1fMB | split_inflation=%s | merge_fallback=%s",
        job_id,
        method,
        total_parts,
        min_parts,
        parts_over_min,
        max_part_mb,
        max_part_pct,
        threshold_mb,
        slack_mb,
        slack_pct,
        min_part_mb,
        max_part_mb,
        avg_part_mb,
        split_inflation_label,
        merge_label,
    )


def _log_job_parallel(job_id: str, result: dict) -> None:
    if result.get("compression_method") != "ghostscript_parallel":
        return

    split_inflation = result.get("split_inflation")
    split_label = "yes" if split_inflation else "no"
    split_pct = result.get("split_inflation_pct")
    if split_pct is None:
        split_pct = 0.0

    merge_fallback = result.get("merge_fallback")
    merge_label = "yes" if merge_fallback else "no"
    merge_time = result.get("merge_fallback_time")
    if merge_time is None:
        merge_time = 0.0

    logger.info(
        "[JOB_PARALLEL] %s | chunks=%s | split_inflation=%s (%.1f%%) | "
        "merge_fallback=%s | merge_time=%.1fs",
        job_id,
        result.get("parallel_chunks", "unknown"),
        split_label,
        split_pct,
        merge_label,
        merge_time,
    )


def _log_output_page_counts(
    output_paths: list[Path],
    label: str,
    input_pages: int | None = None,
    input_path: Path | None = None,
) -> None:
    if not LOG_PART_PAGE_COUNTS or not output_paths:
        return
    try:
        from PyPDF2 import PdfReader
    except Exception as exc:
        logger.warning("[%s] Output page count skipped (PyPDF2 unavailable): %s", label, exc)
        return

    input_size_mb = None
    page_labels_present: bool | None = None
    if input_path:
        try:
            input_size_mb = input_path.stat().st_size / (1024 * 1024)
        except OSError:
            input_size_mb = None
        try:
            with open(input_path, "rb") as f:
                reader = PdfReader(f, strict=False)
                if input_pages is None:
                    input_pages = len(reader.pages)
                try:
                    root = reader.trailer.get("/Root")
                    page_labels_present = bool(root and root.get("/PageLabels"))
                except Exception:
                    page_labels_present = None
        except Exception as exc:
            logger.warning("[%s] Input page audit failed for %s: %s", label, input_path.name, exc)

    if input_pages is not None:
        logger.info("[%s] Input pages: %s", label, input_pages)
    if input_path or input_size_mb is not None or page_labels_present is not None:
        audit_bits = []
        if input_path:
            audit_bits.append(input_path.name)
        if input_size_mb is not None:
            audit_bits.append(f"{input_size_mb:.1f}MB")
        if input_pages is not None:
            audit_bits.append(f"pages={input_pages}")
        else:
            audit_bits.append("pages=unknown")
        if page_labels_present is not None:
            audit_bits.append(f"page_labels={'yes' if page_labels_present else 'no'}")
        logger.info("[%s] Input audit: %s", label, ", ".join(audit_bits))

    total_pages = 0
    total_size_mb = 0.0
    for idx, path in enumerate(output_paths, start=1):
        try:
            with open(path, "rb") as f:
                count = len(PdfReader(f, strict=False).pages)
            size_bytes = path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            total_pages += count
            total_size_mb += size_mb
            logger.info(
                "[%s] Output pages part %s: %s = %s pages (%.1fMB)",
                label,
                idx,
                path.name,
                count,
                size_mb,
            )
        except Exception as exc:
            logger.warning("[%s] Output page count failed for %s: %s", label, path.name, exc)

    logger.info(
        "[%s] Output pages total: %s parts = %s pages (%.1fMB)",
        label,
        len(output_paths),
        total_pages,
        total_size_mb,
    )
    if input_pages is not None:
        delta = total_pages - input_pages
        if delta:
            logger.warning(
                "[%s] Page mismatch: output_total=%s input=%s delta=%+d",
                label,
                total_pages,
                input_pages,
                delta,
            )
        else:
            logger.info("[%s] Page match: output_total=%s input=%s", label, total_pages, input_pages)


def send_salesforce_callback(callback_url: str, payload: dict, job_id: str, max_retries: int = 3, timeout: int = 5) -> bool:
    """Send callback to Salesforce with basic retry/backoff. Runs in a worker thread."""
    headers = {"Content-Type": "application/json"}
    try:
        utils.validate_external_url(callback_url)
    except DownloadError as exc:
        logger.error("[%s] Callback URL blocked: %s", job_id, exc)
        return False

    for attempt in range(max_retries):
        try:
            resp = requests.post(callback_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            logger.info(f"[{job_id}] Callback succeeded: {resp.status_code}")
            return True
        except requests.exceptions.Timeout:
            logger.warning(f"[{job_id}] Callback timeout (attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"[{job_id}] Callback failed (attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            backoff = 2 ** attempt
            time.sleep(backoff)

    logger.error(f"[{job_id}] Callback failed after {max_retries} attempts")
    return False


def spawn_callback(callback_url: str, payload: dict, job_id: str) -> None:
    """Spawn a daemon thread to send the Salesforce callback without blocking workers."""
    thread = threading.Thread(
        target=send_salesforce_callback,
        args=(callback_url, payload, job_id),
        daemon=True,
        name=f"callback-{job_id}"
    )
    thread.start()


LOG_PART_PAGE_COUNTS = utils.env_bool("LOG_PART_PAGE_COUNTS", True)
ENABLE_QUALITY_WARNINGS = utils.env_bool("ENABLE_QUALITY_WARNINGS", True)

DASHBOARD_FILE_MAP = [
    {
        "step": 1,
        "file": "pdf_compressor/services/compression_service.py",
        "summary": "Entry + finalize: API, request parsing, queue, callbacks, download naming",
        "functions": [
            "compress_async",
            "process_compression_job",
            "_build_download_links",
            "send_salesforce_callback",
        ],
        "variables": [
            "ASYNC_MAX_MB",
            "SPLIT_THRESHOLD_MB",
        ],
    },
    {
        "step": 2,
        "file": "pdf_compressor/workers/job_queue.py",
        "summary": "Async worker + job state (async only)",
        "functions": ["create_job", "enqueue", "update_job", "get_stats"],
        "variables": [],
    },
    {
        "step": 3,
        "file": "pdf_compressor/core/utils.py",
        "summary": "Safe download + CPU detection",
        "functions": ["download_pdf", "get_effective_cpu_count"],
        "variables": [],
    },
    {
        "step": 4,
        "file": "pdf_compressor/engine/pdf_diagnostics.py",
        "summary": "Diagnostics + quality warnings",
        "functions": ["get_quality_warnings", "detect_scanned_document"],
        "variables": [],
    },
    {
        "step": 5,
        "file": "pdf_compressor/engine/compress.py",
        "summary": "Compression flow + serial/parallel selection",
        "functions": ["compress_pdf", "_resolve_parallel_workers"],
        "variables": ["PARALLEL_THRESHOLD_MB"],
    },
    {
        "step": 6,
        "file": "pdf_compressor/engine/ghostscript.py",
        "summary": "Ghostscript command + parallel chunk compression",
        "functions": [
            "compress_parallel",
            "compress_pdf_with_ghostscript",
            "_resolve_gs_threads",
        ],
        "variables": [
            "TARGET_CHUNK_MB",
            "MAX_CHUNK_MB",
            "MAX_PAGES_PER_CHUNK",
        ],
    },
    {
        "step": 7,
        "file": "pdf_compressor/engine/split.py",
        "summary": "Split logic (size-based + binary search)",
        "functions": ["split_pdf", "split_by_size", "split_for_delivery"],
        "variables": [],
    },
    {
        "step": 8,
        "file": "pdf_compressor/core/exceptions.py",
        "summary": "Typed errors used across the flow",
        "functions": [
            "PDFCompressionError",
            "DownloadError",
            "SplitError",
            "EncryptionError",
        ],
        "variables": [],
    },
]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@contextlib.contextmanager
def _compression_slot(job_id: str | None = None, input_size_mb: float | None = None):
    label = job_id or "job"
    is_small = input_size_mb is not None and input_size_mb < SMALL_JOB_THRESHOLD_MB
    start = time.time()
    semaphore = COMPRESSION_SEMAPHORE

    if is_small and SMALL_JOB_SEMAPHORE:
        acquired_small = SMALL_JOB_SEMAPHORE.acquire(blocking=False)
        if acquired_small:
            semaphore = SMALL_JOB_SEMAPHORE
        else:
            COMPRESSION_SEMAPHORE.acquire()
            semaphore = COMPRESSION_SEMAPHORE
    else:
        COMPRESSION_SEMAPHORE.acquire()
        semaphore = COMPRESSION_SEMAPHORE

    waited = time.time() - start
    if waited >= 1:
        slot_label = "small" if semaphore is SMALL_JOB_SEMAPHORE else "general"
        logger.info("[%s] Waited %.1fs for %s compression slot", label, waited, slot_label)
    try:
        yield
    finally:
        semaphore.release()


def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _resolve_split_threshold_mb(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        cleaned = raw_value.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in ("off", "none", "false", "no"):
            return None
        if lowered.endswith("mb"):
            cleaned = cleaned[:-2].strip()
        raw_value = cleaned
    value = _safe_float(raw_value)
    if value is None:
        return None
    if value <= 0:
        return None
    # Keep split thresholds in a practical delivery range.
    return min(50.0, max(5.0, value))


def _normalize_url_for_dedupe(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        return ""
    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            scheme = parsed.scheme.lower()
            host = (parsed.hostname or "").lower()
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            netloc = f"{host}:{parsed.port}" if parsed.port else host
            base = f"{scheme}://{netloc}{parsed.path or '/'}"

            query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
            if not query_pairs:
                return base

            canonical_query = urlencode(sorted(query_pairs), doseq=True)
            query_sig = hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()[:16]
            return f"{base}?qsig={query_sig}"
    except Exception:
        pass
    return value


def _build_inflight_dedupe_key(task_data: Dict[str, Any]) -> str | None:
    """Build a stable key for deduping equivalent in-flight compression jobs."""
    if not DEDUPE_INFLIGHT_ENABLED:
        return None

    split_threshold = _resolve_split_threshold_mb(task_data.get("split_threshold_mb") or task_data.get("splitSizeMB"))
    split_label = f"{split_threshold:.1f}" if split_threshold is not None else "off"
    pages_label = str(task_data.get("pages") or "").strip()
    password = str(task_data.get("password") or "")
    password_sig = hashlib.sha256(password.encode("utf-8")).hexdigest()[:12] if password else "none"
    matter_label = str(task_data.get("matter_id") or "").strip()
    callback_url = str(task_data.get("callback_url") or task_data.get("callbackUrl") or "").strip()
    callback_sig = hashlib.sha256(callback_url.encode("utf-8")).hexdigest()[:12] if callback_url else "none"
    mode = (os.environ.get("COMPRESSION_MODE", "aggressive") or "aggressive").strip().lower()

    source_sig: str | None = None
    pdf_bytes = task_data.get("pdf_bytes")
    if isinstance(pdf_bytes, (bytes, bytearray)):
        size_mb = len(pdf_bytes) / (1024 * 1024)
        if size_mb < DEDUPE_MIN_MB:
            return None
        source_sig = f"bytes:{hashlib.sha256(bytes(pdf_bytes)).hexdigest()}"
    elif isinstance(task_data.get("download_url"), str):
        if not DEDUPE_URL_ENABLED:
            return None
        normalized = _normalize_url_for_dedupe(task_data["download_url"])
        if normalized:
            source_sig = f"url:{normalized}"

    if not source_sig:
        return None

    return "|".join([
        "v2",
        source_sig,
        f"mode:{mode}",
        f"split:{split_label}",
        f"pages:{pages_label}",
        f"password:{password_sig}",
        f"matter:{matter_label}",
        f"callback:{callback_sig}",
    ])


def _extract_display_names(download_links: list[str]) -> list[str]:
    names = []
    for link in download_links:
        try:
            parsed = urlparse(link)
            name = parse_qs(parsed.query).get("name", [None])[0]
            if name:
                names.append(name)
            else:
                path_name = Path(parsed.path).name
                names.append(path_name or link)
        except Exception:
            names.append(link)
    return names


def _format_recent_jobs(recent_jobs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    formatted = []
    for job in recent_jobs:
        result = job.get("result") or {}
        progress = job.get("progress") or {}
        status = job.get("status") or "unknown"
        status_class = "ok" if status == "completed" else "warn" if status == "failed" else "neutral"

        original_mb = _safe_float(result.get("original_mb") or result.get("original_size_mb"))
        compressed_mb = _safe_float(result.get("compressed_mb") or result.get("compressed_size_mb"))
        reduction_pct = _safe_float(result.get("reduction_percent"))
        total_parts = result.get("total_parts")
        if total_parts is None and result.get("download_links"):
            total_parts = len(result["download_links"])
        page_count = result.get("page_count")
        compression_mode = result.get("compression_mode")

        size_display = None
        if original_mb is not None:
            if compressed_mb is not None:
                size_display = f"{original_mb:.1f}MB -> {compressed_mb:.1f}MB"
            else:
                size_display = f"{original_mb:.1f}MB"
            if reduction_pct is not None:
                size_display = f"{size_display} ({reduction_pct:.1f}%)"

        parts_bits = []
        if total_parts:
            parts_bits.append(f"parts: {total_parts}")
        if page_count:
            parts_bits.append(f"pages: {page_count}")
        if compression_mode:
            parts_bits.append(f"mode: {compression_mode}")
        parts_display = " | ".join(parts_bits) if parts_bits else None

        timings = result.get("processing_time") or {}
        timing_bits = []
        download_s = _safe_float(timings.get("download_seconds"))
        compress_s = _safe_float(timings.get("compression_seconds"))
        total_s = _safe_float(timings.get("total_seconds"))
        if download_s is not None:
            timing_bits.append(f"download {download_s:.1f}s")
        if compress_s is not None:
            timing_bits.append(f"compress {compress_s:.1f}s")
        if total_s is not None:
            timing_bits.append(f"total {total_s:.1f}s")
        timing_display = " | ".join(timing_bits) if timing_bits else None

        duration_display = "n/a"
        if status == "completed":
            total_s = _safe_float((result.get("processing_time") or {}).get("total_seconds"))
            if total_s is not None:
                duration_display = f"{total_s:.1f}s" if total_s < 90 else f"{total_s / 60:.1f}m"
        elif status == "processing":
            duration_display = "in progress"

        progress_percent = progress.get("percent")
        progress_stage = progress.get("stage")
        progress_message = progress.get("message")
        progress_bits = []
        if progress_stage:
            progress_bits.append(str(progress_stage))
        if progress_percent is not None:
            progress_bits.append(f"{progress_percent}%")
        if progress_message:
            progress_bits.append(str(progress_message))
        progress_display = " | ".join(progress_bits) if progress_bits else None

        file_names_full = _extract_display_names(result.get("download_links") or [])
        file_names = file_names_full[:6]
        file_names_more = max(0, len(file_names_full) - len(file_names))

        formatted.append({
            "job_id": job.get("job_id"),
            "status": status,
            "status_class": status_class,
            "duration_display": duration_display,
            "progress_display": progress_display,
            "size_display": size_display,
            "parts_display": parts_display,
            "timing_display": timing_display,
            "error": job.get("error"),
            "file_names": file_names,
            "file_names_full": file_names_full,
            "file_names_count": len(file_names_full),
            "file_names_more": file_names_more,
        })
    return formatted


def build_health_snapshot() -> Dict[str, Any]:
    """Build a lightweight snapshot for health endpoints and the debug dashboard."""
    import pdf_compressor.engine.ghostscript as gs_cfg
    import pdf_compressor.engine.split as split_cfg

    gs_cmd = get_ghostscript_command()
    try:
        pdf_count = len(list(UPLOAD_FOLDER.glob("*.pdf")))
    except Exception:
        pdf_count = -1

    queue_stats = job_queue.get_stats()
    effective_cpu = utils.get_effective_cpu_count()
    compute_plan = resolve_parallel_compute_plan(effective_cpu)
    recent_jobs = []
    try:
        recent_limit = utils.env_int("DASHBOARD_RECENT_JOBS", 8)
        recent_jobs = _format_recent_jobs(job_queue.get_recent_jobs(recent_limit))
    except Exception:
        recent_jobs = []

    return {
        "status": "healthy" if gs_cmd else "degraded",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "file_map": DASHBOARD_FILE_MAP,
        "recent_jobs": recent_jobs,
        "ghostscript": {
            "available": gs_cmd is not None,
            "command": gs_cmd or "missing",
        },
        "instance_id": os.environ.get("WEBSITE_INSTANCE_ID", "local"),
        "cpu_effective": effective_cpu,
        "storage": {
            "upload_folder": str(UPLOAD_FOLDER.resolve()),
            "pdf_count": pdf_count,
            "retention_seconds": FILE_RETENTION_SECONDS,
            "min_retention_seconds": MIN_FILE_RETENTION_SECONDS,
            "effective_retention_seconds": EFFECTIVE_FILE_RETENTION_SECONDS,
        },
        "queue": queue_stats,
        "limits": {
            "async_max_mb": ASYNC_MAX_MB,
        },
        "compression": {
            "mode": os.environ.get("COMPRESSION_MODE", "aggressive"),
            "allow_lossy": utils.env_bool("ALLOW_LOSSY_COMPRESSION", True),
            "pdf_precheck_enabled": utils.env_bool("PDF_PRECHECK_ENABLED", True),
            "quality_warnings_enabled": utils.env_bool("ENABLE_QUALITY_WARNINGS", True),
            "gs_fast_web_view": utils.env_bool("GS_FAST_WEB_VIEW", False),
            "gs_band_height": utils.env_int("GS_BAND_HEIGHT", 100),
            "gs_band_buffer_space_mb": utils.env_int("GS_BAND_BUFFER_SPACE_MB", 500),
            "gs_color_downsample_type": gs_cfg.GS_COLOR_DOWNSAMPLE_TYPE,
            "gs_gray_downsample_type": gs_cfg.GS_GRAY_DOWNSAMPLE_TYPE,
            "gs_color_image_resolution": gs_cfg.GS_COLOR_IMAGE_RESOLUTION,
            "gs_gray_image_resolution": gs_cfg.GS_GRAY_IMAGE_RESOLUTION,
            "gs_mono_image_resolution": gs_cfg.GS_MONO_IMAGE_RESOLUTION,
        },
        "split": {
            "threshold_mb": SPLIT_THRESHOLD_MB,
            "trigger_mb": SPLIT_TRIGGER_MB,
            "base_threshold_mb": BASE_SPLIT_THRESHOLD_MB,
            "safety_buffer_mb": utils.env_float("SPLIT_SAFETY_BUFFER_MB", 0.0),
            "minimize_parts": split_cfg.SPLIT_MINIMIZE_PARTS,
            "enable_binary_fallback": split_cfg.SPLIT_ENABLE_BINARY_FALLBACK,
            "adaptive_max_attempts": split_cfg.SPLIT_ADAPTIVE_MAX_ATTEMPTS,
        },
        "parallel": {
            "threshold_mb": PARALLEL_THRESHOLD_MB,
            "requested_parallel_workers": compute_plan["requested_parallel_workers"],
            "configured_parallel_workers": compute_plan["configured_parallel_workers"],
            "effective_parallel_workers": compute_plan["effective_parallel_workers"],
            "parallel_max_workers": compute_plan["configured_parallel_workers"],  # Backward compatibility
            "async_workers": compute_plan["async_workers"],
            "max_active_compressions": MAX_ACTIVE_COMPRESSIONS,
            "total_parallel_budget": compute_plan["total_parallel_budget"],
            "per_job_parallel_budget": compute_plan["per_job_parallel_budget"],
            "parallel_budget_enforce": compute_plan["parallel_budget_enforce"],
            "capped_by_budget": compute_plan["capped_by_budget"],
            "capped_reasons": compute_plan.get("capped_reasons", []),
            "cap_chain": compute_plan.get("cap_chain", {}),
            "max_parallel_chunks": utils.env_int("MAX_PARALLEL_CHUNKS", 64),
            "target_chunk_mb": utils.env_float("TARGET_CHUNK_MB", 30.0),
            "max_chunk_mb": utils.env_float("MAX_CHUNK_MB", 50.0),
            "max_pages_per_chunk": utils.env_int("MAX_PAGES_PER_CHUNK", 600),
            "gs_num_rendering_threads": compute_plan["gs_threads_per_worker"],
        },
    }


def create_error_response(error: Exception, status_code: int = 500):
    """Create standardized error response with backward compatibility.

    Returns both 'error' (for old clients) and 'error_type'/'error_message' (for new clients).
    """
    if isinstance(error, PDFCompressionError):
        return jsonify({
            "success": False,
            "error": error.message,  # Backward compatibility
            "error_type": error.error_type,
            "error_message": error.message,
        }), status_code

    return jsonify({
        "success": False,
        "error": str(error),
        "error_type": "UnknownError",
        "error_message": str(error),
    }), status_code


def get_error_status_code(error: Exception) -> int:
    """Map exception type to appropriate HTTP status code."""
    if isinstance(error, DownloadError):
        # DownloadError carries its own status code for clarity
        return getattr(error, "status_code", 400)
    if isinstance(error, ProcessingTimeoutError):
        return 504
    if isinstance(error, (EncryptionError, StructureError, MetadataCorruptionError, SplitError)):
        return 422  # Unprocessable Entity - valid request but can't process the PDF
    if isinstance(error, FileNotFoundError):
        return 404
    if isinstance(error, ValueError):
        return 400
    return 500


def get_performance_notes(timings: dict, pdf_info: dict, warnings: list) -> list:
    """Generate explanations for slow processing.

    Called after compression to explain to users WHY processing took time.
    Only returns notes when processing exceeds expected thresholds.

    Args:
        timings: Dict with download_seconds, compression_seconds
        pdf_info: Dict with page_count, original_mb
        warnings: Quality warnings from get_quality_warnings()

    Returns:
        List of performance note dicts with stage, seconds, and reason.
    """
    notes = []

    # Slow compression (>30 seconds)
    compress_seconds = timings.get("compression_seconds", 0)
    if compress_seconds > 30:
        pages = pdf_info.get("page_count") or 0
        size_mb = pdf_info.get("original_mb") or 0

        # Check if warnings indicate scanned document
        is_scanned = any(w.get("type") == "scanned_document" for w in warnings)

        if pages > 100:
            reason = f"PDF has {pages} pages - large documents require more processing"
        elif size_mb > 50:
            reason = f"PDF is {size_mb:.0f}MB - large files take longer to compress"
        elif is_scanned:
            reason = "Scanned document with images - these compress slower than text PDFs"
        else:
            reason = "Complex PDF structure required extended processing"

        notes.append({
            "stage": "compression",
            "seconds": compress_seconds,
            "reason": reason
        })

    # Slow download (>10 seconds)
    download_seconds = timings.get("download_seconds", 0)
    if download_seconds > 10:
        notes.append({
            "stage": "download",
            "seconds": download_seconds,
            "reason": "Source server responded slowly - not within our control"
        })

    return notes


# File tracking for cleanup
file_lock = threading.Lock()
tracked_files = {}


def track_file(path: Path):
    """Track file for cleanup."""
    with file_lock:
        tracked_files[str(path)] = time.time()


def cleanup_daemon():
    """Background cleanup - removes old files."""
    while True:
        time.sleep(60)
        cutoff = time.time() - EFFECTIVE_FILE_RETENTION_SECONDS
        with file_lock:
            tracked_snapshot = dict(tracked_files)
        expired = {p for p, t in tracked_snapshot.items() if t < cutoff}

        # Include untracked PDFs (e.g., after restart) based on mtime.
        try:
            for pdf_path in UPLOAD_FOLDER.glob("*.pdf"):
                pdf_str = str(pdf_path)
                if pdf_str in tracked_snapshot:
                    continue
                try:
                    mtime = pdf_path.stat().st_mtime
                except OSError:
                    continue
                if mtime < cutoff:
                    expired.add(pdf_str)
                else:
                    with file_lock:
                        tracked_files[pdf_str] = mtime
        except Exception as e:
            logger.error(f"Cleanup scan error: {e}")

        for path in sorted(expired):
            try:
                Path(path).unlink(missing_ok=True)
                with file_lock:
                    tracked_files.pop(path, None)
                logger.info(f"Cleaned up: {Path(path).name}")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# Start cleanup daemon


def process_compression_job(job_id: str, task_data: Dict[str, Any]) -> None:
    """Process a compression job in the background.

    Args:
        job_id: The job identifier.
        task_data: Dict containing either 'pdf_bytes' or 'download_url'.
    """
    start_time = time.time()

    # Progress callback to update job status in real-time
    def progress_callback(percent: int, stage: str, message: str):
        job_queue.update_job(job_id, "processing", progress={
            "percent": percent,
            "stage": stage,
            "message": message
        })

    try:
        name_hint: str | None = None
        if isinstance(task_data.get("name"), str):
            name_hint = task_data["name"]
        elif isinstance(task_data.get("download_url"), str):
            name_hint = task_data["download_url"]

        split_override_raw = task_data.get("split_threshold_mb") or task_data.get("splitSizeMB")
        split_override = _resolve_split_threshold_mb(split_override_raw)
        split_requested = _is_truthy(task_data.get("split")) or split_override is not None
        split_origin = "off"
        if split_requested:
            split_origin = "request" if split_override is not None else "default"
        if split_requested and split_override is None:
            split_override = BASE_SPLIT_THRESHOLD_MB
        split_threshold_mb = split_override if split_requested else None
        split_trigger_mb = split_threshold_mb if split_threshold_mb is not None else None

        input_path: Path
        source_label = "unknown"

        progress_callback(5, "uploading", "Receiving file...")

        # Get PDF bytes - either from task_data, a pre-downloaded file, or download from URL
        if 'input_path' in task_data:
            input_path = Path(task_data['input_path'])
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            source_label = "preloaded"
        else:
            input_path = UPLOAD_FOLDER / f"{job_id}_input.pdf"
            if 'pdf_bytes' in task_data:
                pdf_bytes = task_data['pdf_bytes']
                input_path.write_bytes(pdf_bytes)
                source_label = "upload"
            elif 'download_url' in task_data:
                job_queue.download_pdf(
                    task_data['download_url'],
                    input_path,
                    max_download_size_bytes=int(ASYNC_MAX_MB * 1024 * 1024),
                )
                source_label = "download"
            else:
                raise ValueError("No PDF data provided")

        track_file(input_path)

        # Optional: subset pages and/or decrypt using provided password
        page_spec = task_data.get("pages")
        password = task_data.get("password")
        if page_spec or password:
            from PyPDF2 import PdfReader, PdfWriter
            logger.info(f"[{job_id}] Applying page/password options (pages={bool(page_spec)}, password={'yes' if password else 'no'})")
            if page_spec:
                spec_preview = str(page_spec).strip()
                if len(spec_preview) > 120:
                    spec_preview = f"{spec_preview[:117]}..."
                logger.info("[%s] Page selection spec: %s", job_id, spec_preview)

            def _parse_pages(spec: str, total: int) -> list[int]:
                pages: list[int] = []
                for part in spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        start_s, end_s = part.split("-", 1)
                        try:
                            start = int(start_s)
                            end = int(end_s)
                        except ValueError:
                            continue
                        if start <= 0:
                            start = 1
                        if end <= 0:
                            continue
                        for p in range(start, end + 1):
                            if 1 <= p <= total:
                                pages.append(p)
                    else:
                        try:
                            p = int(part)
                            if 1 <= p <= total:
                                pages.append(p)
                        except ValueError:
                            continue
                # Deduplicate while preserving order
                seen = set()
                ordered = []
                for p in pages:
                    if p not in seen:
                        seen.add(p)
                        ordered.append(p)
                return ordered

            subset_path = UPLOAD_FOLDER / f"{job_id}_subset.pdf"
            with open(input_path, "rb") as f:
                reader = PdfReader(f, strict=False)
                if reader.is_encrypted:
                    try:
                        reader.decrypt(password or "")
                    except Exception:
                        # Keep behavior consistent with legacy: let GS try later
                        pass
                total_pages = len(reader.pages)
                selected = _parse_pages(page_spec, total_pages) if page_spec else list(range(1, total_pages + 1))
                if page_spec and not selected:
                    logger.warning(
                        "[%s] Page selection matched 0 pages; defaulting to full document",
                        job_id,
                    )
                if not selected:
                    selected = list(range(1, total_pages + 1))
                writer = PdfWriter()
                added = 0
                skipped = 0
                first_error: Exception | None = None
                first_error_page: int | None = None
                for p in selected:
                    # p is 1-based
                    try:
                        writer.add_page(reader.pages[p - 1])
                        added += 1
                    except Exception as exc:
                        skipped += 1
                        if first_error is None:
                            first_error = exc
                            first_error_page = p
                        continue
                logger.info(
                    "[%s] Page selection summary: total=%s selected=%s added=%s skipped=%s",
                    job_id,
                    total_pages,
                    len(selected),
                    added,
                    skipped,
                )
                if skipped and first_error:
                    logger.warning(
                        "[%s] Page selection skipped %s pages (first failed page=%s); first error: %s",
                        job_id,
                        skipped,
                        first_error_page,
                        first_error,
                    )
                with open(subset_path, "wb") as out_f:
                    writer.write(out_f)
            input_path = subset_path
            track_file(input_path)

        _log_job_header(
            job_id,
            source_label,
            input_path,
            name_hint,
            split_threshold_mb=split_threshold_mb,
            split_trigger_mb=split_trigger_mb,
        )
        _log_job_snapshot(
            job_id,
            source_label,
            input_path,
            name_hint,
            split_threshold_mb,
            split_trigger_mb,
            split_origin,
        )
        _log_pdf_fingerprint(job_id, input_path)

        download_time = round(time.time() - start_time, 2)

        # Get quality warnings before compression
        warnings = get_quality_warnings(input_path) if ENABLE_QUALITY_WARNINGS else []

        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        progress_callback(8, "queued", "Waiting for compression slot...")
        with _compression_slot(job_id, input_size_mb=input_size_mb):
            compress_start = time.time()
            progress_callback(10, "compressing", "Starting compression...")

            # Compress with splitting enabled
            result = compress_pdf(
                str(input_path),
                working_dir=UPLOAD_FOLDER,
                split_threshold_mb=split_threshold_mb,
                split_trigger_mb=split_trigger_mb,
                progress_callback=progress_callback
            )

            compress_time = round(time.time() - compress_start, 2)

        output_paths = [Path(p) for p in result.get("output_paths", [])]
        if not output_paths and result.get("output_path"):
            output_paths = [Path(result["output_path"])]

        expected_parts = result.get("total_parts") or len(output_paths) or 1
        if expected_parts > len(output_paths):
            recovered = sorted(UPLOAD_FOLDER.glob(f"{input_path.stem}_part*.pdf"))
            if len(recovered) >= expected_parts:
                output_paths = recovered[:expected_parts]
                logger.warning(f"[{job_id}] Recovered {len(output_paths)} split parts from disk")
            else:
                logger.warning(
                    f"[{job_id}] Expected {expected_parts} parts but only found {len(output_paths)} output paths"
                )

        if not output_paths:
            raise SplitError("Compression produced no output files")

        # Track all output files
        for path in output_paths:
            track_file(path)

        should_log_parts = split_threshold_mb is not None or len(output_paths) > 1
        if should_log_parts:
            _log_output_page_counts(output_paths, job_id, result.get("page_count"), input_path)
        _log_job_result(job_id, result, output_paths)
        _log_job_parallel(job_id, result)
        _log_job_flags(job_id, result, output_paths, split_threshold_mb=split_threshold_mb)

        progress_callback(98, "finalizing", "Preparing download links...")

        # Build download links
        download_links = _build_download_links(output_paths, name_hint)
        files = [build_download_url(link) for link in download_links]

        # Build timing info
        timings = {
            "download_seconds": download_time,
            "compression_seconds": compress_time,
            "total_seconds": round(time.time() - start_time, 2)
        }

        # Generate performance notes explaining why processing took time
        perf_notes = get_performance_notes(
            timings,
            {"page_count": result.get('page_count'), "original_mb": result['original_size_mb']},
            warnings
        )

        # Debug summary for performance/part count investigations
        threshold_label = f"{split_threshold_mb:.1f}MB" if split_threshold_mb is not None else "off"
        trigger_label = f"{split_trigger_mb:.1f}MB" if split_trigger_mb is not None else "off"
        logger.info(
            "[%s] Summary: %.1fMB -> %.1fMB (%.1f%%), parts=%s (threshold=%s trigger=%s), "
            "timings: download=%.2fs compress=%.2fs total=%.2fs",
            job_id,
            result['original_size_mb'],
            result['compressed_size_mb'],
            result['reduction_percent'],
            result.get('total_parts') or len(output_paths),
            threshold_label,
            trigger_label,
            timings.get("download_seconds", 0.0),
            timings.get("compression_seconds", 0.0),
            timings.get("total_seconds", 0.0),
        )

        matter_id = task_data.get("matter_id")
        callback_url = task_data.get("callback_url") or task_data.get("callbackUrl")
        base_url_hint = (task_data.get("base_url") or BASE_URL or "").rstrip("/")
        name_hint = task_data.get("name")

        job_queue.update_job(job_id, "completed", result={
            "was_split": result.get('was_split', False),
            "total_parts": result.get('total_parts', 1),
            "download_links": download_links,
            "files": files,
            "original_mb": result['original_size_mb'],
            "compressed_mb": result['compressed_size_mb'],
            "reduction_percent": result['reduction_percent'],
            "compression_method": result['compression_method'],
            "compression_mode": result.get("compression_mode"),
            "request_id": job_id,
            "page_count": result.get('page_count'),
            "part_sizes": result.get('part_sizes'),
            "quality_warnings": warnings,
            "processing_time": timings,
            "performance_notes": perf_notes,
            "matter_id": matter_id,
            "expiresAt": (datetime.utcnow().timestamp() + EFFECTIVE_FILE_RETENTION_SECONDS),
            "name": name_hint or None,
        })

        logger.info(f"[{job_id}] Job completed: {len(download_links)} file(s)")

        # Send callback to Salesforce if requested
        if callback_url:
            def to_absolute(link: str) -> str:
                if link.startswith("http://") or link.startswith("https://"):
                    return link
                base = base_url_hint
                if not base and has_request_context():
                    base = request.url_root.rstrip('/')
                if base and base.startswith("http://"):
                    base = "https://" + base[len("http://"):]
                if base:
                    if link.startswith("/"):
                        return f"{base}{link}"
                    return f"{base}/{link.lstrip('/')}"
                return link  # Fallback to relative if no base available

            absolute_links = [to_absolute(l) for l in download_links]
            payload = {
                "matterId": matter_id,
                "compressedLinks": absolute_links,
                "compressedLink": absolute_links[0] if absolute_links else None,
                "downloadLinks": absolute_links,
                "totalParts": len(absolute_links),
                "wasSplit": len(absolute_links) > 1,
                "partSizes": result.get("part_sizes"),
                "expiresAt": (datetime.utcnow().timestamp() + EFFECTIVE_FILE_RETENTION_SECONDS),
            }
            spawn_callback(callback_url, payload, job_id)

    except PDFCompressionError as e:
        # Our custom exceptions - log and return structured error
        logger.error(f"[{job_id}] Job failed ({e.error_type}): {e.message}")
        job_queue.update_job(job_id, "failed", error=str(e), result={
            "error_type": e.error_type,
            "error_message": e.message,
        })
        # Send error callback if requested
        matter_id = task_data.get("matter_id")
        callback_url = task_data.get("callback_url")
        if matter_id and callback_url:
            payload = {
                "matterId": matter_id,
                "compressedLinks": [],
                "error": e.message or str(e),
                "error_type": e.error_type,
                "error_message": e.message or str(e),
            }
            spawn_callback(callback_url, payload, job_id)
    except Exception as e:
        # Unexpected errors - log full stack trace
        logger.exception(f"[{job_id}] Job failed with unexpected error: {e}")
        job_queue.update_job(job_id, "failed", error=str(e), result={
            "error_type": "UnknownError",
            "error_message": str(e),
        })
        matter_id = task_data.get("matter_id")
        callback_url = task_data.get("callback_url")
        if matter_id and callback_url:
            payload = {
                "matterId": matter_id,
                "compressedLinks": [],
                "error": str(e),
                "error_type": "UnknownError",
                "error_message": str(e),
            }
            spawn_callback(callback_url, payload, job_id)



# Error handlers
def handle_large_file(e):
    max_mb = int(MAX_CONTENT_LENGTH / (1024 * 1024))
    message = f"File too large (max {max_mb}MB)"
    return jsonify({
        "success": False,
        "error": message,
        "error_type": "FileTooLarge",
        "error_message": message,
    }), 413


def handle_http_exception(e):
    if isinstance(e, NotFound):
        logger.info("404 %s %s", request.method, request.path)
    else:
        logger.warning("HTTP %s on %s %s: %s", e.code, request.method, request.path, e.description)
    return create_error_response(e, e.code or 400)


def handle_error(e):
    logger.exception("Unhandled error")
    return create_error_response(e, get_error_status_code(e))


# Routes
def dashboard():
    """Serve the debug dashboard."""
    snapshot = build_health_snapshot()
    return render_template('dashboard.html', snapshot=snapshot)


def health():
    """Health check endpoint with instance info for debugging."""
    snapshot = build_health_snapshot()
    return jsonify(snapshot)


@require_auth
def compress_async():
    """
    Compress a PDF file asynchronously.

    Accepts:
    - application/json with 'file_download_link' (URL to download PDF)
    - application/json with 'file_content_base64' (base64-encoded PDF)
    - multipart/form-data with 'pdf' or 'file' field

    Returns job_id for polling status. Poll /status/<job_id> for results.
    """
    task_data: Dict[str, Any] = {}
    matter_id = None
    callback_url = None

    def _request_error(message: str, status: int = 400, error_type: str = "InvalidRequest"):
        return jsonify({
            "success": False,
            "error": message,
            "error_type": error_type,
            "error_message": message,
        }), status

    max_upload_bytes = int(ASYNC_MAX_MB * 1024 * 1024)

    # Parse input
    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _request_error("Invalid JSON body")
        if not data:
            return _request_error("Empty request body")
        split_override_raw = data.get("split_threshold_mb")
        if split_override_raw is None:
            split_override_raw = data.get("splitSizeMB")
        split_override = _resolve_split_threshold_mb(split_override_raw)
        if split_override is not None:
            task_data["split_threshold_mb"] = split_override
            task_data["split"] = True

        # Option 1: URL to download PDF
        download_url = data.get('file_download_link') or data.get('url') or data.get('file_url')
        if download_url:
            if not download_url or not isinstance(download_url, str):
                return _request_error("Invalid file_download_link")
            task_data['download_url'] = download_url
            logger.info("Compress request with URL: %s", utils.redact_url_for_log(download_url, max_len=120))

        # Option 2: Base64-encoded PDF
        elif 'file_content_base64' in data:
            try:
                pdf_bytes = base64.b64decode(data['file_content_base64'])
            except Exception:
                return _request_error("Invalid base64")

            if len(pdf_bytes) > max_upload_bytes:
                return _request_error(
                    f"File too large: {len(pdf_bytes) / (1024 * 1024):.1f}MB "
                    f"(limit {ASYNC_MAX_MB:.0f}MB)",
                    status=413,
                    error_type="FileTooLarge",
                )

            if not pdf_bytes[:5] == b'%PDF-':
                return _request_error("Invalid PDF file")

            task_data['pdf_bytes'] = pdf_bytes
            logger.info(f"Compress request with base64: {len(pdf_bytes) / (1024*1024):.1f}MB")

        else:
            return _request_error("Missing file_download_link or file_content_base64")

        # Optional PDF.co-style fields (no-ops when absent)
        if isinstance(data.get("pages"), str):
            task_data["pages"] = data["pages"]
        if isinstance(data.get("password"), str):
            task_data["password"] = data["password"]
        if isinstance(data.get("name"), str):
            task_data["name"] = data["name"]
        if not task_data.get("name") and task_data.get("download_url"):
            task_data["name"] = task_data["download_url"]
        matter_id = data.get("matterId")
        if matter_id is None:
            matter_id = data.get("matter_id")
        if matter_id is not None:
            task_data["matter_id"] = matter_id
        if isinstance(data.get("callbackUrl"), str):
            callback_url = data["callbackUrl"]
        if callback_url is None and isinstance(data.get("callback"), str):
            callback_url = data["callback"]
        if data.get("async") is True:
            task_data["force_async"] = True  # retained for compatibility; processing is async already

    # Option 3: Multipart form upload
    else:
        split_override_raw = request.form.get("split_threshold_mb")
        if split_override_raw is None:
            split_override_raw = request.form.get("splitSizeMB")
        split_override = _resolve_split_threshold_mb(split_override_raw)
        if split_override is not None:
            task_data["split_threshold_mb"] = split_override
            task_data["split"] = True

        upload = request.files.get('pdf') or request.files.get('file')
        if not upload:
            return _request_error("Missing 'pdf' field")
        if not upload.filename:
            return _request_error("No file selected")

        pdf_bytes = upload.read()
        if len(pdf_bytes) > max_upload_bytes:
            return _request_error(
                f"File too large: {len(pdf_bytes) / (1024 * 1024):.1f}MB "
                f"(limit {ASYNC_MAX_MB:.0f}MB)",
                status=413,
                error_type="FileTooLarge",
            )
        if not pdf_bytes[:5] == b'%PDF-':
            return _request_error("Invalid PDF file")

        task_data['pdf_bytes'] = pdf_bytes
        task_data["name"] = upload.filename
        logger.info(f"Compress request with upload: {len(pdf_bytes) / (1024*1024):.1f}MB")

    if matter_id and not callback_url:
        callback_url = os.environ.get("SALESFORCE_CALLBACK_URL")
        if not callback_url:
            return _request_error("SALESFORCE_CALLBACK_URL not configured", status=500, error_type="ConfigError")

    if callback_url:
        try:
            utils.validate_external_url(callback_url)
        except DownloadError as exc:
            return _request_error(
                f"Invalid callback URL: {exc}",
                status=400,
                error_type="InvalidCallbackURL",
            )
        task_data["callback_url"] = callback_url
        base_url_hint = (BASE_URL or request.url_root.rstrip('/')).rstrip('/')
        if base_url_hint:
            task_data["base_url"] = base_url_hint

    # Create job (or reuse equivalent in-flight job) and enqueue for processing
    dedupe_key = _build_inflight_dedupe_key(task_data)
    job_id, reused_job = job_queue.create_or_reuse_job(dedupe_key=dedupe_key)
    if reused_job:
        logger.info("[%s] Reused in-flight job for duplicate request", job_id)
        response = {
            "success": True,
            "job_id": job_id,
            "status_url": f"/status/{job_id}",
            "deduped": True,
        }
        if matter_id:
            response["matterId"] = matter_id
            response["message"] = "Attached to existing in-flight job"
        return jsonify(response), 202

    # Pre-download URL-based PDFs immediately to prevent presigned URL expiration
    # while the job waits in queue. Falls back to worker-download on failure.
    if 'download_url' in task_data and 'pdf_bytes' not in task_data and 'input_path' not in task_data:
        predownload_path = UPLOAD_FOLDER / f"{job_id}_input.pdf"
        try:
            job_queue.download_pdf(
                task_data['download_url'],
                predownload_path,
                max_download_size_bytes=int(ASYNC_MAX_MB * 1024 * 1024),
            )
            task_data['input_path'] = str(predownload_path)
            # Keep download_url for name_hint but remove it from download triggers
            task_data.pop('download_url')
            track_file(predownload_path)
            logger.info(
                "[%s] Pre-downloaded %.1fMB to prevent URL expiration",
                job_id,
                predownload_path.stat().st_size / (1024 * 1024),
            )
        except Exception as exc:
            logger.warning(
                "[%s] Pre-download failed (%s); will retry in worker",
                job_id,
                exc,
            )
            # Leave download_url in task_data for worker to try
            predownload_path.unlink(missing_ok=True)

    try:
        job_queue.enqueue(job_id, task_data)
    except Exception as e:
        logger.warning(f"[{job_id}] Queue full or enqueue failed: {e}")
        job_queue.update_job(job_id, "failed", error=str(e), result={
            "error_type": "ServerBusy",
            "error_message": "Server is busy. Please retry shortly.",
        })
        return jsonify({
            "success": False,
            "error": "Server is busy. Please retry shortly.",
            "error_type": "ServerBusy",
            "error_message": "Server is busy. Please retry shortly.",
        }), 503

    response = {
        "success": True,
        "job_id": job_id,
        "status_url": f"/status/{job_id}"
    }
    if matter_id:
        response["matterId"] = matter_id
        response["message"] = "Processing started"
    return jsonify(response), 202


@require_auth
def get_status(job_id: str):
    """
    Get the status of a compression job.

    Args:
        job_id: The job identifier from /compress-async response.

    Returns:
        Job status and results when completed.
    """
    job = job_queue.get_job(job_id)

    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job.status == "queued":
        response = {
            "success": True,
            "status": "queued",
        }
        if job.progress:
            response["progress"] = job.progress
        return jsonify(response)

    if job.status == "processing":
        response = {
            "success": True,
            "status": "processing"
        }
        # Include progress data if available
        if job.progress:
            response["progress"] = job.progress
        return jsonify(response)

    if job.status == "failed":
        response = {
            "success": False,
            "status": "failed",
            "error": job.error,  # Backward compatibility
        }
        # Add structured error info if available
        if job.result and isinstance(job.result, dict):
            if "error_type" in job.result:
                response["error_type"] = job.result["error_type"]
            if "error_message" in job.result:
                response["error_message"] = job.result["error_message"]
        return jsonify(response)

    # Completed - normalize download URLs to be absolute
    result = dict(job.result or {})
    download_links = result.get("download_links")
    if isinstance(download_links, str):
        download_links = [download_links]
        result["download_links"] = download_links

    files = result.get("files")
    if isinstance(files, str):
        files = [files]
        result["files"] = files

    output_paths = result.get("output_paths") or ([result["output_path"]] if "output_path" in result else [])

    if not files:
        if output_paths:
            files = [build_download_url(p) for p in output_paths]
        elif download_links:
            files = [build_download_url(link) for link in download_links]
        if files:
            result["files"] = files
    if not download_links and files:
        result["download_links"] = files

    return jsonify({
        "success": True,
        "status": "completed",
        **result
    })


@require_auth
def job_check_pdfco():
    """
    Lightweight PDF.co-style job status check.

    Accepts JSON: {"jobid": "...", "force": true/false}
    - force is accepted for compatibility and ignored (no re-run to avoid duplicates).
    - Returns a minimal status map: working/success/failed/unknown plus optional links.
    """
    data = request.get_json(silent=True) or {}
    job_id = data.get("jobid") or data.get("jobId")
    if not job_id:
        return jsonify({"error": "Missing jobid"}), 400

    job = job_queue.get_job(job_id)
    force_flag = data.get("force")
    if force_flag:
        logger.info(f"[{job_id}] job/check force flag accepted (no-op for compatibility)")

    if not job:
        return jsonify({"jobId": job_id, "status": "unknown"}), 200

    status_map = {
        "queued": "working",
        "processing": "working",
        "completed": "success",
        "failed": "failed",
    }
    status = status_map.get(job.status, "unknown")

    payload = {"jobId": job_id, "status": status}

    if job.status == "failed":
        payload["error"] = job.error or "Job failed"
        if job.result and isinstance(job.result, dict):
            if "error_type" in job.result:
                payload["error_type"] = job.result["error_type"]
            if "error_message" in job.result:
                payload["error_message"] = job.result["error_message"]
    if job.status == "completed" and job.result:
        # Return download links when available for parity.
        links = job.result.get("download_links") or job.result.get("files") or []
        if isinstance(links, str):
            links = [links]
        payload["downloadLinks"] = links

    # Simple duration hint; job.created_at is set at creation time.
    payload["jobDuration"] = round(time.time() - job.created_at, 2)

    return jsonify(payload), 200




def download(filename):
    """
    Direct file download endpoint.

    Allows downloading compressed PDFs directly without base64 encoding.
    Files are automatically cleaned up after FILE_RETENTION_SECONDS.
    """
    # Security: Prevent path traversal attacks
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        logger.warning(f"[download] Invalid filename rejected: {filename}")
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    file_path = UPLOAD_FOLDER / safe_filename

    # Extra security: Verify file is within upload folder
    try:
        file_path.resolve().relative_to(UPLOAD_FOLDER.resolve())
    except ValueError:
        logger.warning(f"[download] Path traversal attempt blocked: {filename}")
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    if not file_path.exists():
        # Debug logging to help diagnose 404 issues
        logger.error(f"[download] File not found: {file_path}")
        logger.error(f"[download] UPLOAD_FOLDER resolved to: {UPLOAD_FOLDER.resolve()}")

        # List files in upload folder for debugging
        try:
            existing_files = list(UPLOAD_FOLDER.glob("*.pdf"))[:10]  # First 10 PDFs
            logger.error(f"[download] Sample files in uploads: {[f.name for f in existing_files]}")
        except Exception as e:
            logger.error(f"[download] Could not list upload folder: {e}")

        return jsonify({"success": False, "error": "File not found"}), 404

    track_file(file_path)
    logger.info(f"[download] Serving file: {file_path.name} ({file_path.stat().st_size / (1024*1024):.1f}MB)")
    display_name = request.args.get("name")
    if display_name:
        display_name = secure_filename(display_name)
        if not display_name:
            display_name = safe_filename
        elif not display_name.lower().endswith(".pdf"):
            display_name = f"{display_name}.pdf"
    else:
        display_name = safe_filename
    return send_file(file_path, as_attachment=True, download_name=display_name)


def favicon():
    """Serve an empty favicon to stop 500/404 noise."""
    return jsonify({"status": "ok"}), 200

@require_auth
def diagnose(job_id: str):
    """
    Get diagnostic information for a compression job.

    Provides detailed analysis of input/output files, size verification,
    and quality warnings. Useful for debugging size discrepancies.

    Args:
        job_id: The job identifier from /compress-async response.

    Returns:
        Diagnostic report with file analyses and size verification.
    """
    job = job_queue.get_job(job_id)

    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

    if job.status == "processing":
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "processing",
            "message": "Job still processing, diagnostics available after completion"
        })

    if job.status == "failed":
        return jsonify({
            "success": False,
            "job_id": job_id,
            "status": "failed",
            "error": job.error
        })

    # Job completed - run diagnostics
    result = job.result or {}
    download_links = result.get("download_links", [])
    reported_compressed_mb = result.get("compressed_mb", 0)

    # Convert download links back to file paths
    output_paths = []
    for link in download_links:
        parsed = urlparse(link)
        filename = Path(parsed.path).name
        output_paths.append(str(UPLOAD_FOLDER / filename))

    # Check if input file still exists
    input_path = UPLOAD_FOLDER / f"{job_id}_input.pdf"
    input_exists = input_path.exists()

    # Generate diagnostic report
    report = diagnose_for_job(
        input_path if input_exists else None,
        output_paths,
        reported_compressed_mb
    )

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": "completed",
        "job_result": result,
        "diagnostics": report,
        "input_file_available": input_exists,
        "timestamp": datetime.utcnow().isoformat()
    })
