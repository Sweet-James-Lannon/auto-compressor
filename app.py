"""Flask API for PDF compression with async processing and auto-split."""

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from flask import Flask, jsonify, request, render_template, send_file, has_request_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from compress import compress_pdf
from compress_ghostscript import get_ghostscript_command
from pdf_diagnostics import diagnose_for_job, get_quality_warnings
from exceptions import (
    PDFCompressionError,
    EncryptionError,
    StructureError,
    MetadataCorruptionError,
    SplitError,
    DownloadError,
)
import job_queue
import utils

# Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='dashboard')

# Constants
MAX_CONTENT_LENGTH = 314572800  # 300 MB
HARD_SYNC_LIMIT_MB = 300.0  # Absolute ceiling for sync endpoint
UPLOAD_FOLDER = Path(__file__).parent / "uploads"  # Absolute path for Azure compatibility
FILE_RETENTION_SECONDS = int(os.environ.get('FILE_RETENTION_SECONDS', '86400'))  # 24 hours
SPLIT_THRESHOLD_MB = float(os.environ.get('SPLIT_THRESHOLD_MB', '25'))
# Cap SYNC_MAX_MB at the hard ceiling even if env is higher
SYNC_MAX_MB = min(float(os.environ.get("SYNC_MAX_MB", str(HARD_SYNC_LIMIT_MB))), HARD_SYNC_LIMIT_MB)
API_TOKEN = os.environ.get('API_TOKEN')
# Public base URL for download links (e.g., https://yourapp.azurewebsites.net)
BASE_URL = os.environ.get('BASE_URL', '').rstrip('/')
# Raised default to 540s to align with gunicorn timeout and avoid 499/504s on large files.
SYNC_TIMEOUT_SECONDS = int(os.environ.get("SYNC_TIMEOUT_SECONDS", "540"))

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
UPLOAD_FOLDER.mkdir(exist_ok=True)


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
        cutoff = time.time() - FILE_RETENTION_SECONDS
        with file_lock:
            expired = [p for p, t in tracked_files.items() if t < cutoff]
        for path in expired:
            try:
                Path(path).unlink(missing_ok=True)
                with file_lock:
                    tracked_files.pop(path, None)
                logger.info(f"Cleaned up: {Path(path).name}")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# Start cleanup daemon
threading.Thread(target=cleanup_daemon, daemon=True).start()


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
        input_path = UPLOAD_FOLDER / f"{job_id}_input.pdf"

        progress_callback(5, "uploading", "Receiving file...")

        # Get PDF bytes - either from task_data or download from URL
        if 'pdf_bytes' in task_data:
            pdf_bytes = task_data['pdf_bytes']
            input_path.write_bytes(pdf_bytes)
        elif 'download_url' in task_data:
            job_queue.download_pdf(task_data['download_url'], input_path)
        else:
            raise ValueError("No PDF data provided")

        track_file(input_path)
        download_time = round(time.time() - start_time, 2)

        # Get quality warnings before compression
        warnings = get_quality_warnings(str(input_path))

        compress_start = time.time()
        progress_callback(10, "compressing", "Starting compression...")

        # Compress with splitting enabled
        result = compress_pdf(
            str(input_path),
            working_dir=UPLOAD_FOLDER,
            split_threshold_mb=SPLIT_THRESHOLD_MB,
            progress_callback=progress_callback
        )

        compress_time = round(time.time() - compress_start, 2)

        # Track all output files
        for path_str in result.get('output_paths', [result['output_path']]):
            track_file(Path(path_str))

        progress_callback(98, "finalizing", "Preparing download links...")

        # Build download links
        download_links = [
            f"/download/{Path(p).name}"
            for p in result.get('output_paths', [result['output_path']])
        ]

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

        job_queue.update_job(job_id, "completed", result={
            "was_split": result.get('was_split', False),
            "total_parts": result.get('total_parts', 1),
            "download_links": download_links,
            "original_mb": result['original_size_mb'],
            "compressed_mb": result['compressed_size_mb'],
            "reduction_percent": result['reduction_percent'],
            "compression_method": result['compression_method'],
            "request_id": job_id,
            "page_count": result.get('page_count'),
            "part_sizes": result.get('part_sizes'),
            "quality_warnings": warnings,
            "processing_time": timings,
            "performance_notes": perf_notes
        })

        logger.info(f"[{job_id}] Job completed: {len(download_links)} file(s)")

    except PDFCompressionError as e:
        # Our custom exceptions - log and return structured error
        logger.error(f"[{job_id}] Job failed ({e.error_type}): {e.message}")
        job_queue.update_job(job_id, "failed", error=str(e), result={
            "error_type": e.error_type,
            "error_message": e.message,
        })
    except Exception as e:
        # Unexpected errors - log full stack trace
        logger.exception(f"[{job_id}] Job failed with unexpected error: {e}")
        job_queue.update_job(job_id, "failed", error=str(e), result={
            "error_type": "UnknownError",
            "error_message": str(e),
        })


# Configure and start job queue worker
job_queue.set_processor(process_compression_job)
job_queue.start_workers(8)


# Error handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"success": False, "error": "File too large (max 300MB)"}), 413


@app.errorhandler(Exception)
def handle_error(e):
    logger.exception("Unhandled error")
    return create_error_response(e, get_error_status_code(e))


# Routes
@app.route('/')
def dashboard():
    """Serve the web UI."""
    return render_template('dashboard.html')


@app.route('/health')
def health():
    """Health check endpoint with instance info for debugging."""
    gs = get_ghostscript_command()

    # Count files in upload folder for debugging
    try:
        pdf_count = len(list(UPLOAD_FOLDER.glob("*.pdf")))
    except Exception:
        pdf_count = -1

    return jsonify({
        "status": "healthy" if gs else "degraded",
        "ghostscript": gs is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "upload_folder": str(UPLOAD_FOLDER.resolve()),
        "pdf_count": pdf_count,
        "instance_id": os.environ.get("WEBSITE_INSTANCE_ID", "local"),  # Azure instance ID
    })


@app.route('/compress', methods=['POST'])
@require_auth
def compress():
    """
    Compress a PDF file asynchronously.

    Accepts:
    - application/json with 'file_download_link' (URL to download PDF)
    - application/json with 'file_content_base64' (base64-encoded PDF)
    - multipart/form-data with 'pdf' field

    Returns job_id for polling status. Poll /status/<job_id> for results.
    """
    task_data: Dict[str, Any] = {}

    # Parse input
    if request.is_json:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Empty request body"}), 400

        # Option 1: URL to download PDF
        if 'file_download_link' in data:
            download_url = data['file_download_link']
            if not download_url or not isinstance(download_url, str):
                return jsonify({"success": False, "error": "Invalid file_download_link"}), 400
            task_data['download_url'] = download_url
            logger.info(f"Compress request with URL: {download_url[:100]}...")

        # Option 2: Base64-encoded PDF
        elif 'file_content_base64' in data:
            try:
                pdf_bytes = base64.b64decode(data['file_content_base64'])
            except Exception:
                return jsonify({"success": False, "error": "Invalid base64"}), 400

            if not pdf_bytes[:5] == b'%PDF-':
                return jsonify({"success": False, "error": "Invalid PDF file"}), 400

            task_data['pdf_bytes'] = pdf_bytes
            logger.info(f"Compress request with base64: {len(pdf_bytes) / (1024*1024):.1f}MB")

        else:
            return jsonify({
                "success": False,
                "error": "Missing file_download_link or file_content_base64"
            }), 400

    # Option 3: Multipart form upload
    else:
        if 'pdf' not in request.files:
            return jsonify({"success": False, "error": "Missing 'pdf' field"}), 400
        f = request.files['pdf']
        if not f.filename:
            return jsonify({"success": False, "error": "No file selected"}), 400

        pdf_bytes = f.read()
        if not pdf_bytes[:5] == b'%PDF-':
            return jsonify({"success": False, "error": "Invalid PDF file"}), 400

        task_data['pdf_bytes'] = pdf_bytes
        logger.info(f"Compress request with upload: {len(pdf_bytes) / (1024*1024):.1f}MB")

    # Create job and enqueue for processing
    job_id = job_queue.create_job()
    job_queue.enqueue(job_id, task_data)

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status_url": f"/status/{job_id}"
    }), 202


@app.route('/status/<job_id>', methods=['GET'])
@require_auth
def get_status(job_id: str):
    """
    Get the status of a compression job.

    Args:
        job_id: The job identifier from /compress response.

    Returns:
        Job status and results when completed.
    """
    job = job_queue.get_job(job_id)

    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404

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
    output_paths = result.get("output_paths") or ([result["output_path"]] if "output_path" in result else [])
    files = [build_download_url(p) for p in output_paths] if output_paths else []
    if files:
        result["files"] = files

    return jsonify({
        "success": True,
        "status": "completed",
        **result
    })


@app.route('/compress-sync', methods=['POST'])
@require_auth
def compress_sync():
    """
    Synchronous compression endpoint - handles both Salesforce and dashboard requests.
    Blocks until complete, returns download URLs directly.

    Accepts two input methods:
    1. JSON body: {"file_download_link": "https://..."} - for Salesforce/API calls
    2. Form-data: file field with PDF upload - for dashboard testing

    Response: {"success": true, "files": ["/download/part1.pdf", ...]}
    """
    # Generate unique ID for this request (supports concurrent users)
    file_id = str(uuid.uuid4()).replace('-', '')[:16]
    input_path = UPLOAD_FOLDER / f"{file_id}_input.pdf"

    try:
        # Dual input handling - JSON (Salesforce) or form-data (Dashboard)
        if request.is_json:
            # Salesforce/API flow - download PDF from URL
            data = request.get_json()
            file_url = data.get('file_download_link') or data.get('file_url')
            if not file_url:
                return jsonify({
                    "success": False,
                    "error": "Missing file_download_link",
                    "error_type": "MissingParameter",
                    "error_message": "Missing file_download_link"
                }), 400
            parsed = urlparse(file_url)
            host_hint = parsed.hostname or "unknown-host"
            path_hint = (parsed.path or "")[:50]
            logger.info(f"[sync:{file_id}] Downloading from host={host_hint} path={path_hint}...")
            utils.download_pdf(file_url, input_path)
            # Validate downloaded file is actually a PDF
            with open(input_path, 'rb') as f:
                header = f.read(5)
            if header != b'%PDF-':
                input_path.unlink(missing_ok=True)
                return jsonify({
                    "success": False,
                    "error": "Downloaded file is not a valid PDF",
                    "error_type": "InvalidPDF",
                    "error_message": "Downloaded file is not a valid PDF"
                }), 400
            downloaded_mb = input_path.stat().st_size / (1024 * 1024)
            logger.info(f"[sync:{file_id}] Downloaded: {downloaded_mb:.1f}MB (will split if > {SPLIT_THRESHOLD_MB}MB)")

        elif request.files and 'file' in request.files:
            # Dashboard flow - direct file upload
            f = request.files['file']
            if not f.filename:
                return jsonify({
                    "success": False,
                    "error": "No file selected",
                    "error_type": "MissingFile",
                    "error_message": "No file selected"
                }), 400
            pdf_bytes = f.read()
            if len(pdf_bytes) < 5 or pdf_bytes[:5] != b'%PDF-':
                return jsonify({
                    "success": False,
                    "error": "Invalid PDF file",
                    "error_type": "InvalidPDF",
                    "error_message": "Invalid PDF file"
                }), 400
            input_path.write_bytes(pdf_bytes)
            logger.info(f"[sync:{file_id}] Received upload: {f.filename} ({len(pdf_bytes) / (1024*1024):.1f}MB)")

        else:
            msg = "No file or URL provided. Send JSON with file_download_link or form-data with file field."
            return jsonify({
                "success": False,
                "error": msg,
                "error_type": "MissingFile",
                "error_message": msg
            }), 400

        track_file(input_path)

        # Check file size - reject files over configured or hard limit for sync endpoint
        file_size_mb = input_path.stat().st_size / (1024 * 1024)

        if file_size_mb > SYNC_MAX_MB:
            logger.warning(
                f"[sync:{file_id}] File too large for sync: {file_size_mb:.1f}MB > {SYNC_MAX_MB:.1f}MB limit "
                f"(hard cap {HARD_SYNC_LIMIT_MB:.0f}MB, Azure HTTP timeout is ~230s)"
            )
            input_path.unlink(missing_ok=True)
            with file_lock:
                tracked_files.pop(str(input_path), None)
            return jsonify({
                "success": False,
                "error": f"File too large for sync ({file_size_mb:.1f}MB). Max allowed: {SYNC_MAX_MB:.0f}MB.",
                "error_type": "FileTooLarge",
                "error_message": f"File exceeds sync processing limit. Azure HTTP timeout is ~230s; split the file and retry.",
                "recommendation": "Pre-split the file into smaller parts before upload or use the async endpoint"
            }), 413

        # Run compression with a timeout to avoid gateway kills
        def run_compress():
            return compress_pdf(
                str(input_path),
                working_dir=UPLOAD_FOLDER,
                split_threshold_mb=SPLIT_THRESHOLD_MB
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_compress)
            try:
                result = future.result(timeout=SYNC_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                logger.error(f"[sync:{file_id}] Timed out after {SYNC_TIMEOUT_SECONDS}s")
                future.cancel()
                return jsonify({
                    "success": False,
                    "error": f"Processing timed out after {SYNC_TIMEOUT_SECONDS}s. Try a smaller file or retry later.",
                    "error_type": "Timeout",
                    "error_message": f"Processing timed out after {SYNC_TIMEOUT_SECONDS}s. Try a smaller file or retry later."
                }), 504

        # =====================================================================
        # CRITICAL: Verify all output files exist before returning download URLs
        # This prevents 404 errors when files weren't created properly
        # =====================================================================
        output_paths = result.get('output_paths', [result['output_path']])

        logger.info(f"[sync:{file_id}] Verifying {len(output_paths)} output files...")

        for path_str in output_paths:
            p = Path(path_str)
            logger.info(f"[sync:{file_id}] Checking: {p} (absolute: {p.resolve()})")

            if not p.exists():
                logger.error(f"[sync:{file_id}] Output file missing: {p}")
                logger.error(f"[sync:{file_id}] UPLOAD_FOLDER is: {UPLOAD_FOLDER.resolve()}")
                return jsonify({
                    "success": False,
                    "error": "Compression failed - output file not created",
                    "error_type": "OutputMissing",
                    "error_message": "Compression failed - output file not created"
                }), 500
            if p.stat().st_size == 0:
                logger.error(f"[sync:{file_id}] Output file is empty: {p}")
                return jsonify({
                    "success": False,
                    "error": "Compression failed - output file is empty",
                    "error_type": "OutputMissing",
                    "error_message": "Compression failed - output file is empty"
                }), 500

            logger.info(f"[sync:{file_id}] Verified: {p.name} ({p.stat().st_size / (1024*1024):.1f}MB)")
            track_file(p)

        # Always return part sizes for Salesforce verification
        part_sizes = result.get('part_sizes')
        if not part_sizes:
            part_sizes = [Path(p).stat().st_size for p in output_paths]

        files = [build_download_url(p) for p in output_paths]

        logger.info(f"[sync:{file_id}] Complete: {len(files)} file(s), {result['original_size_mb']:.1f}MB → {result['compressed_size_mb']:.1f}MB")
        logger.info(f"[sync:{file_id}] Download URLs: {files}")
        return jsonify({
            "success": True,
            "files": files,
            "original_mb": result['original_size_mb'],
            "compressed_mb": result['compressed_size_mb'],
            "was_split": result.get('was_split', False),
            "total_parts": result.get('total_parts', 1),
            "part_sizes": part_sizes  # Individual file sizes in bytes for verification
        })

    except DownloadError as e:
        logger.warning(f"[sync:{file_id}] Download error: {e}")
        return create_error_response(e, get_error_status_code(e))
    except Exception as e:
        logger.exception(f"[sync:{file_id}] Failed: {e}")
        return create_error_response(e, get_error_status_code(e))


@app.route('/download/<filename>')
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

    logger.info(f"[download] Serving file: {file_path.name} ({file_path.stat().st_size / (1024*1024):.1f}MB)")
    return send_file(file_path, as_attachment=True, download_name=safe_filename)


@app.route('/favicon.ico')
def favicon():
    """Serve an empty favicon to stop 500/404 noise."""
    return jsonify({"status": "ok"}), 200

@app.route('/diagnose/<job_id>', methods=['GET'])
@require_auth
def diagnose(job_id: str):
    """
    Get diagnostic information for a compression job.

    Provides detailed analysis of input/output files, size verification,
    and quality warnings. Useful for debugging size discrepancies.

    Args:
        job_id: The job identifier from /compress response.

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
        filename = link.split("/")[-1]
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    logger.info(f"Starting server on port {port}")
    logger.info(f"File retention: {FILE_RETENTION_SECONDS}s")
    app.run(host='0.0.0.0', port=port, debug=False)
