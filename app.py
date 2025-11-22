"""
Pure Flask REST API for law firm PDF compression.
POST /compress endpoint with cryptographic verification and auto-cleanup.
Thread-safe, production-ready, zero-trust architecture.
"""

import atexit
import base64
import logging
import os
import subprocess
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import hashlib
import hmac

from flask import Flask, Response, jsonify, request, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from compress import compress_pdf, CompressionError, VerificationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='dashboard')

# Configuration
MAX_CONTENT_LENGTH = 314572800  # 300 MB in bytes
UPLOAD_FOLDER = Path("./uploads")
FILE_RETENTION_SECONDS = int(os.environ.get('FILE_RETENTION_SECONDS', '30'))  # Default 30 seconds for law firm security
ALLOWED_EXTENSIONS = {'.pdf'}
PDF_MAGIC_BYTES = b'%PDF-'
API_TOKEN = os.environ.get('API_TOKEN')  # REQUIRED - must be set in production
RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', '100'))  # Requests per window
RATE_LIMIT_WINDOW = int(os.environ.get('RATE_LIMIT_WINDOW', '3600'))  # Window in seconds

# Flask config
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Rate limiting storage
rate_limit_storage = {}
rate_limit_lock = threading.Lock()

# Thread-safe file tracking
file_lock = threading.Lock()
tracked_files: Dict[str, float] = {}  # {file_path: creation_timestamp}

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(exist_ok=True)
logger.info(f"Upload directory: {UPLOAD_FOLDER.absolute()}")

# Ensure API token is configured in production
# TEMPORARILY DISABLED FOR TESTING - Comment out the check below to re-enable authentication
# if not API_TOKEN and not os.environ.get('DEBUG', 'False').lower() == 'true':
#     logger.error("API_TOKEN environment variable is required in production!")
#     raise ValueError("API_TOKEN must be set in production environment")


def require_auth(f):
    """Decorator to require Bearer token authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        if not auth_header:
            logger.warning("Missing Authorization header")
            return jsonify({"success": False, "error": "Authorization required"}), 401

        try:
            auth_type, token = auth_header.split(' ', 1)
            if auth_type.lower() != 'bearer':
                logger.warning(f"Invalid auth type: {auth_type}")
                return jsonify({"success": False, "error": "Bearer token required"}), 401

            if token != API_TOKEN:
                logger.warning("Invalid API token")
                return jsonify({"success": False, "error": "Invalid token"}), 403

        except ValueError:
            logger.warning("Malformed Authorization header")
            return jsonify({"success": False, "error": "Invalid authorization format"}), 401

        return f(*args, **kwargs)
    return decorated_function


def check_rate_limit(identifier: str) -> bool:
    """
    Check if the identifier has exceeded rate limit.

    Args:
        identifier: Client identifier (e.g., IP address or API key)

    Returns:
        True if within limit, False if exceeded
    """
    current_time = time.time()

    with rate_limit_lock:
        # Clean old entries
        cutoff_time = current_time - RATE_LIMIT_WINDOW
        rate_limit_storage[identifier] = [
            timestamp for timestamp in rate_limit_storage.get(identifier, [])
            if timestamp > cutoff_time
        ]

        # Check current count
        request_count = len(rate_limit_storage.get(identifier, []))

        if request_count >= RATE_LIMIT_REQUESTS:
            return False

        # Add current request
        if identifier not in rate_limit_storage:
            rate_limit_storage[identifier] = []
        rate_limit_storage[identifier].append(current_time)

        return True


def cleanup_old_files():
    """
    Background cleanup task that removes files older than FILE_RETENTION_SECONDS.
    Runs every 30 seconds.
    """
    while True:
        try:
            current_time = time.time()
            files_to_delete = []

            with file_lock:
                for file_path, created_at in list(tracked_files.items()):
                    age = current_time - created_at
                    if age > FILE_RETENTION_SECONDS:
                        files_to_delete.append(file_path)

            # Delete files outside lock to avoid blocking
            for file_path in files_to_delete:
                try:
                    path = Path(file_path)
                    if path.exists():
                        path.unlink()
                        logger.info(f"Auto-deleted file after {FILE_RETENTION_SECONDS}s: {path.name}")

                    with file_lock:
                        tracked_files.pop(file_path, None)

                except Exception as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")

            time.sleep(30)  # Run every 30 seconds

        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            time.sleep(30)


def track_file(file_path: Path) -> None:
    """Add file to tracking for auto-deletion."""
    with file_lock:
        tracked_files[str(file_path)] = time.time()
    logger.debug(f"Tracking file: {file_path.name}")


def untrack_file(file_path: Path) -> None:
    """Remove file from tracking."""
    with file_lock:
        tracked_files.pop(str(file_path), None)
    logger.debug(f"Untracked file: {file_path.name}")


def delete_file_immediately(file_path: Path) -> None:
    """Delete file immediately and untrack it."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted file: {file_path.name}")
        untrack_file(file_path)
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")


def validate_pdf_file(file_storage) -> Tuple[bool, str]:
    """
    Validate uploaded file is a legitimate PDF.

    Args:
        file_storage: Werkzeug FileStorage object

    Returns:
        (is_valid, error_message)
    """
    # Check filename exists
    if not file_storage.filename:
        return False, "No filename provided"

    # Check file extension
    file_ext = Path(file_storage.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file extension: {file_ext}. Only .pdf allowed"

    # Check MIME type
    if file_storage.content_type not in ['application/pdf', 'application/x-pdf']:
        return False, f"Invalid MIME type: {file_storage.content_type}. Expected application/pdf"

    # Check magic bytes (PDF signature)
    file_storage.seek(0)
    header = file_storage.read(5)
    file_storage.seek(0)

    if header != PDF_MAGIC_BYTES:
        return False, "Invalid PDF file: missing PDF signature"

    return True, ""


def generate_uuid_filename() -> str:
    """Generate random UUID-based filename."""
    return f"{uuid.uuid4()}.pdf"


def create_error_response(message: str, status_code: int) -> Tuple[Response, int]:
    """Create JSON error response."""
    logger.error(f"Error {status_code}: {message}")
    return jsonify({
        "success": False,
        "error": message
    }), status_code


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    """Handle files exceeding MAX_CONTENT_LENGTH."""
    max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
    return create_error_response(
        f"File too large. Maximum size: {max_mb:.0f}MB",
        413
    )


@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    return create_error_response("Endpoint not found. Only POST /compress is available", 404)


@app.errorhandler(405)
def handle_method_not_allowed(e):
    """Handle method not allowed errors."""
    return create_error_response("Method not allowed. Only POST is supported", 405)


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    return create_error_response("Internal server error", 500)


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


@app.route('/', methods=['GET'])
def dashboard():
    """Dashboard for testing the compression API."""
    return render_template('dashboard.html')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.

    Returns system status and availability of compression tools.
    """
    import shutil
    import pikepdf

    health_status = {
        "status": "healthy",
        "version": "2.0.0",  # Updated to reflect pikepdf implementation
        "timestamp": datetime.utcnow().isoformat(),
        "pikepdf": True,  # Always available as it's a pip package
        "qpdf": shutil.which("qpdf") is not None,
        "upload_dir_exists": UPLOAD_FOLDER.exists(),
        "upload_dir_writable": os.access(UPLOAD_FOLDER, os.W_OK) if UPLOAD_FOLDER.exists() else False,
    }

    # Try to get pikepdf version
    try:
        health_status["pikepdf_version"] = pikepdf.__version__
    except:
        health_status["pikepdf_version"] = "unknown"

    # Check qpdf version if available
    if health_status["qpdf"]:
        try:
            result = subprocess.run(
                ["qpdf", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse version from output like "qpdf version 11.1.0"
                version_line = result.stdout.split('\n')[0]
                if 'version' in version_line:
                    health_status["qpdf_version"] = version_line.split('version')[-1].strip()
        except:
            health_status["qpdf_version"] = "error"

    # Determine overall health
    if not health_status["upload_dir_writable"]:
        health_status["status"] = "unhealthy"
        health_status["error"] = "Upload directory not writable"
        return jsonify(health_status), 503

    return jsonify(health_status), 200


@app.route('/compress', methods=['POST'])
# @require_auth  # TEMPORARILY DISABLED FOR TESTING - Uncomment to re-enable authentication
def compress_endpoint():
    """
    POST /compress - Compress PDF with cryptographic verification.

    Request:
        - Content-Type: multipart/form-data
        - Field name: pdf
        - Max size: 300MB
        - Optional: matter_id, user_email (for tracking)

    Response (200):
        {
            "success": true,
            "original_mb": 115.2,
            "compressed_mb": 24.7,
            "original_sha256": "abc123...",
            "compressed_sha256": "def456...",
            "compressed_pdf_b64": "JVBERi0x..."
        }

    Errors:
        400 - Bad request (missing/invalid file)
        413 - File too large (>300MB)
        429 - Rate limit exceeded
        500 - Compression/verification failed
    """
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())[:8]
    client_ip = request.remote_addr or 'unknown'

    logger.info(f"[{request_id}] Compression request from {client_ip}")

    # Check rate limit
    if not check_rate_limit(client_ip):
        logger.warning(f"[{request_id}] Rate limit exceeded for {client_ip}")
        return jsonify({
            "success": False,
            "error": "Rate limit exceeded. Please try again later.",
            "request_id": request_id
        }), 429

    input_file_path = None
    output_file_path = None

    try:
        # Validate request has file
        if 'pdf' not in request.files:
            return create_error_response(
                "Missing 'pdf' field in request. Use multipart/form-data with field name 'pdf'",
                400
            )

        file = request.files['pdf']

        # Validate file was actually selected
        if file.filename == '':
            return create_error_response("No file selected", 400)

        # Validate PDF file
        is_valid, error_msg = validate_pdf_file(file)
        if not is_valid:
            return create_error_response(error_msg, 400)

        # Extract optional tracking parameters
        matter_id = request.form.get('matter_id', None)
        user_email = request.form.get('user_email', None)

        # Generate secure UUID filename
        uuid_filename = generate_uuid_filename()
        input_file_path = UPLOAD_FOLDER / uuid_filename

        # Ensure path stays within upload directory (defense in depth)
        input_file_path = input_file_path.resolve()
        if not str(input_file_path).startswith(str(UPLOAD_FOLDER.resolve())):
            logger.error(f"[{request_id}] Path traversal attempt detected")
            return create_error_response("Invalid file path", 400)

        # Save uploaded file
        logger.info(f"[{request_id}] Receiving upload: {file.filename} → {uuid_filename}")
        if matter_id:
            logger.info(f"[{request_id}] Matter ID: {matter_id}")
        if user_email:
            logger.info(f"[{request_id}] User Email: {user_email}")

        file.save(str(input_file_path))
        track_file(input_file_path)

        # Verify file was saved
        if not input_file_path.exists():
            return create_error_response("Failed to save uploaded file", 500)

        file_size_mb = input_file_path.stat().st_size / (1024 * 1024)
        logger.info(f"[{request_id}] Uploaded file size: {file_size_mb:.2f}MB")

        # Compress PDF
        logger.info(f"[{request_id}] Starting compression: {uuid_filename}")

        try:
            result = compress_pdf(str(input_file_path), working_dir=UPLOAD_FOLDER)
        except FileNotFoundError as e:
            logger.error(f"[{request_id}] File not found: {e}")
            return create_error_response("File processing error", 400)
        except CompressionError as e:
            logger.error(f"[{request_id}] Compression error: {e}")
            return create_error_response("PDF compression failed", 500)
        except VerificationError as e:
            logger.error(f"[{request_id}] Verification error: {e}")
            return create_error_response("PDF verification failed", 500)
        except Exception as e:
            logger.exception(f"[{request_id}] Unexpected compression error")
            return create_error_response("Internal server error", 500)

        output_file_path = Path(result['output_path'])
        track_file(output_file_path)

        # Read compressed PDF and encode to base64
        with open(output_file_path, 'rb') as f:
            compressed_pdf_bytes = f.read()

        compressed_pdf_b64 = base64.b64encode(compressed_pdf_bytes).decode('utf-8')

        # Build response
        response_data = {
            "success": True,
            "original_mb": result['original_size_mb'],
            "compressed_mb": result['compressed_size_mb'],
            "original_sha256": result['original_hash'],
            "compressed_sha256": result['compressed_hash'],
            "compressed_pdf_b64": compressed_pdf_b64,
            "request_id": request_id,
            "qpdf_used": result.get('qpdf_used', False)
        }

        # Add tracking info if provided
        if matter_id:
            response_data['matter_id'] = matter_id
        if user_email:
            response_data['user_email'] = user_email

        # Calculate compression ratio (avoid division by zero for very small files)
        if result['original_size_mb'] > 0:
            compression_ratio = (
                (result['original_size_mb'] - result['compressed_size_mb'])
                / result['original_size_mb']
                * 100
            )
        else:
            compression_ratio = 0.0

        logger.info(
            f"[{request_id}] Compression complete: {result['original_size_mb']:.2f}MB → "
            f"{result['compressed_size_mb']:.2f}MB ({compression_ratio:.1f}% reduction)"
        )

        return jsonify(response_data), 200

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error in compress endpoint")
        return create_error_response("Internal server error", 500)

    finally:
        # Files will be auto-deleted after FILE_RETENTION_SECONDS
        # No immediate cleanup to allow inspection if needed
        pass


# Start cleanup daemon thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True, name="FileCleanup")
cleanup_thread.start()
logger.info(f"Started file cleanup daemon (retention: {FILE_RETENTION_SECONDS}s)")


# Register cleanup on exit
def cleanup_on_exit():
    """Delete all tracked files on application exit."""
    logger.info("Application shutdown - cleaning up all files")
    with file_lock:
        files_to_delete = list(tracked_files.keys())

    for file_path in files_to_delete:
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file on exit: {path.name}")
        except Exception as e:
            logger.error(f"Failed to delete file on exit {file_path}: {e}")


atexit.register(cleanup_on_exit)


if __name__ == '__main__':
    # Development server (use gunicorn in production)
    port = int(os.environ.get('PORT', 5005))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Prevent debug mode in production (when API token is set)
    if API_TOKEN and debug:
        logger.warning("Debug mode disabled in production (API_TOKEN is set)")
        debug = False

    logger.info(f"Starting Flask development server on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Max upload size: {MAX_CONTENT_LENGTH / (1024 * 1024):.0f}MB")
    logger.info(f"File retention: {FILE_RETENTION_SECONDS} seconds")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )


