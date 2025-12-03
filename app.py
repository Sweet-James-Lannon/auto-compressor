"""Flask API for PDF compression with async processing and auto-split."""

import base64
import logging
import os
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, render_template, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from compress import compress_pdf
from compress_ghostscript import get_ghostscript_command
from exceptions import (
    PDFCompressionError,
    EncryptionError,
    StructureError,
    MetadataCorruptionError,
    CompressionFailureError,
    SplitError,
)
import job_queue

# Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='dashboard')

# Constants
MAX_CONTENT_LENGTH = 314572800  # 300 MB
UPLOAD_FOLDER = Path(__file__).parent / "uploads"  # Absolute path for Azure compatibility
FILE_RETENTION_SECONDS = int(os.environ.get('FILE_RETENTION_SECONDS', '86400'))  # 24 hours
SPLIT_THRESHOLD_MB = float(os.environ.get('SPLIT_THRESHOLD_MB', '25'))
API_TOKEN = os.environ.get('API_TOKEN')

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
UPLOAD_FOLDER.mkdir(exist_ok=True)


def require_auth(f):
    """Decorator to require Bearer token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)  # No token configured = open access
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"success": False, "error": "Missing authorization"}), 401
        if auth_header[7:] != API_TOKEN:
            return jsonify({"success": False, "error": "Invalid token"}), 403
        return f(*args, **kwargs)
    return decorated


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
    if isinstance(error, (EncryptionError, StructureError, MetadataCorruptionError, SplitError)):
        return 422  # Unprocessable Entity - valid request but can't process the PDF
    if isinstance(error, FileNotFoundError):
        return 404
    if isinstance(error, ValueError):
        return 400
    return 500


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
    try:
        input_path = UPLOAD_FOLDER / f"{job_id}_input.pdf"

        # Get PDF bytes - either from task_data or download from URL
        if 'pdf_bytes' in task_data:
            pdf_bytes = task_data['pdf_bytes']
            input_path.write_bytes(pdf_bytes)
        elif 'download_url' in task_data:
            job_queue.download_pdf(task_data['download_url'], input_path)
        else:
            raise ValueError("No PDF data provided")

        track_file(input_path)

        # Compress with splitting enabled
        result = compress_pdf(
            str(input_path),
            working_dir=UPLOAD_FOLDER,
            split_threshold_mb=SPLIT_THRESHOLD_MB
        )

        # Track all output files
        for path_str in result.get('output_paths', [result['output_path']]):
            track_file(Path(path_str))

        # Build download links
        download_links = [
            f"/download/{Path(p).name}"
            for p in result.get('output_paths', [result['output_path']])
        ]

        job_queue.update_job(job_id, "completed", result={
            "was_split": result.get('was_split', False),
            "total_parts": result.get('total_parts', 1),
            "download_links": download_links,
            "original_mb": result['original_size_mb'],
            "compressed_mb": result['compressed_size_mb'],
            "reduction_percent": result['reduction_percent'],
            "compression_method": result['compression_method'],
            "request_id": job_id
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
job_queue.start_worker()


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
    """Health check endpoint."""
    gs = get_ghostscript_command()
    return jsonify({
        "status": "healthy" if gs else "degraded",
        "ghostscript": gs is not None,
        "timestamp": datetime.utcnow().isoformat()
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
        return jsonify({
            "success": True,
            "status": "processing"
        })

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

    # Completed
    return jsonify({
        "success": True,
        "status": "completed",
        **job.result
    })


@app.route('/download/<filename>')
def download(filename):
    """
    Direct file download endpoint.

    Allows downloading compressed PDFs directly without base64 encoding.
    Files are automatically cleaned up after FILE_RETENTION_SECONDS.
    """
    # Security: Prevent path traversal attacks
    from werkzeug.utils import secure_filename
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    file_path = UPLOAD_FOLDER / safe_filename

    # Extra security: Verify file is within upload folder
    try:
        file_path.resolve().relative_to(UPLOAD_FOLDER.resolve())
    except ValueError:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    if not file_path.exists():
        return jsonify({"success": False, "error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=safe_filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    logger.info(f"Starting server on port {port}")
    logger.info(f"File retention: {FILE_RETENTION_SECONDS}s")
    app.run(host='0.0.0.0', port=port, debug=False)
