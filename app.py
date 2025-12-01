"""Flask API for PDF compression. Clean, simple, working."""

import base64
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, render_template, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from compress import compress_pdf
from compress_ghostscript import get_ghostscript_command

# Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='dashboard')

# Constants
MAX_CONTENT_LENGTH = 314572800  # 300 MB
UPLOAD_FOLDER = Path("./uploads")
FILE_RETENTION_SECONDS = int(os.environ.get('FILE_RETENTION_SECONDS', '600'))
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


# Error handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"success": False, "error": "File too large (max 300MB)"}), 413


@app.errorhandler(Exception)
def handle_error(e):
    logger.exception("Unhandled error")
    return jsonify({"success": False, "error": str(e)}), 500


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
    Compress a PDF file.

    Accepts:
    - multipart/form-data with 'pdf' field
    - application/json with 'file_content_base64' field

    Returns compressed PDF as base64.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Compress request")

    # Parse input (multipart or JSON)
    if request.is_json:
        data = request.get_json()
        if not data or 'file_content_base64' not in data:
            return jsonify({"success": False, "error": "Missing file_content_base64"}), 400
        try:
            pdf_bytes = base64.b64decode(data['file_content_base64'])
        except Exception:
            return jsonify({"success": False, "error": "Invalid base64"}), 400
    else:
        if 'pdf' not in request.files:
            return jsonify({"success": False, "error": "Missing 'pdf' field"}), 400
        f = request.files['pdf']
        if not f.filename:
            return jsonify({"success": False, "error": "No file selected"}), 400
        pdf_bytes = f.read()

    # Validate PDF
    if not pdf_bytes[:5] == b'%PDF-':
        return jsonify({"success": False, "error": "Invalid PDF file"}), 400

    # Save input file
    input_path = UPLOAD_FOLDER / f"{uuid.uuid4()}.pdf"
    input_path.write_bytes(pdf_bytes)
    track_file(input_path)

    file_mb = len(pdf_bytes) / (1024 * 1024)
    logger.info(f"[{request_id}] Input: {file_mb:.1f}MB")

    # Compress
    try:
        result = compress_pdf(str(input_path), working_dir=UPLOAD_FOLDER)
    except Exception as e:
        logger.error(f"[{request_id}] Compression failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    output_path = Path(result['output_path'])
    track_file(output_path)

    # Read and encode result
    compressed_b64 = base64.b64encode(output_path.read_bytes()).decode()

    logger.info(
        f"[{request_id}] Done: {result['original_size_mb']}MB -> "
        f"{result['compressed_size_mb']}MB ({result['reduction_percent']}%)"
    )

    return jsonify({
        "success": True,
        "original_mb": result['original_size_mb'],
        "compressed_mb": result['compressed_size_mb'],
        "reduction_percent": result['reduction_percent'],
        "compressed_pdf_b64": compressed_b64,
        "compression_method": result['compression_method'],
        "request_id": request_id
    })


@app.route('/download/<filename>')
def download(filename):
    """
    Direct file download endpoint.

    Allows downloading compressed PDFs directly without base64 encoding.
    Files are automatically cleaned up after FILE_RETENTION_SECONDS.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        return jsonify({"success": False, "error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    logger.info(f"Starting server on port {port}")
    logger.info(f"File retention: {FILE_RETENTION_SECONDS}s")
    app.run(host='0.0.0.0', port=port, debug=False)
