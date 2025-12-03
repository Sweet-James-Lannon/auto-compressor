"""Job queue module for async PDF processing.

Provides job queue and status tracking for long-running compression tasks.
Handles background processing to avoid blocking the main request threads.
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

# Constants
JOB_TTL_SECONDS: int = 3600  # Jobs expire after 1 hour
DOWNLOAD_TIMEOUT: int = 300  # 5 minute timeout for downloading PDFs
MAX_DOWNLOAD_SIZE: int = 314572800  # 300 MB max download

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents an async processing job."""

    job_id: str
    status: str  # "processing", "completed", "failed"
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Module-level state
_jobs: Dict[str, Job] = {}
_job_lock = threading.Lock()
_work_queue: queue.Queue = queue.Queue()
_processor: Optional[Callable] = None


def create_job() -> str:
    """Create a new job and return its ID.

    Returns:
        The unique job ID (16 characters for sufficient entropy).
    """
    job_id = str(uuid.uuid4()).replace('-', '')[:16]
    job = Job(job_id=job_id, status="processing")

    with _job_lock:
        _jobs[job_id] = job

    logger.info(f"[{job_id}] Job created")
    return job_id


def get_job(job_id: str) -> Optional[Job]:
    """Get a job by its ID.

    Args:
        job_id: The job identifier.

    Returns:
        The Job object if found, None otherwise.
    """
    with _job_lock:
        return _jobs.get(job_id)


def update_job(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> None:
    """Update a job's status and result.

    Args:
        job_id: The job identifier.
        status: New status ("processing", "completed", "failed").
        result: Optional result data for completed jobs.
        error: Optional error message for failed jobs.
    """
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].status = status
            _jobs[job_id].result = result
            _jobs[job_id].error = error

    logger.info(f"[{job_id}] Status updated: {status}")


def enqueue(job_id: str, task_data: Dict[str, Any]) -> None:
    """Add a job to the processing queue.

    Args:
        job_id: The job identifier.
        task_data: Data needed to process the job (pdf_bytes, download_url, etc).
    """
    _work_queue.put((job_id, task_data))
    logger.info(f"[{job_id}] Job enqueued")


def _is_safe_url(url: str) -> bool:
    """Check if URL is safe to fetch (not internal/metadata endpoints).

    Args:
        url: The URL to validate.

    Returns:
        True if URL appears safe, False otherwise.
    """
    from urllib.parse import urlparse
    import ipaddress

    try:
        parsed = urlparse(url)

        # Only allow http/https schemes
        if parsed.scheme not in ('http', 'https'):
            return False

        # Block common metadata and internal endpoints
        blocked_hosts = [
            '169.254.169.254',  # Azure/AWS metadata
            'metadata.google.internal',
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
        ]

        hostname = parsed.hostname or ''
        if hostname.lower() in blocked_hosts:
            return False

        # Block private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass  # Not an IP address, hostname is OK

        return True
    except Exception:
        return False


def download_pdf(url: str, output_path: Path) -> None:
    """Download a PDF from a URL.

    Args:
        url: The URL to download from.
        output_path: Path to save the downloaded file.

    Raises:
        RuntimeError: If download fails or file is too large.
    """
    # Security: Validate URL to prevent SSRF
    if not _is_safe_url(url):
        raise RuntimeError("Invalid or blocked URL")

    logger.info(f"Downloading PDF from {url[:100]}...")

    try:
        response = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
            headers={'User-Agent': 'SJ-PDF-Compressor/1.0'},
            allow_redirects=False  # Don't follow redirects to internal URLs
        )
        response.raise_for_status()

        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            raise RuntimeError(f"File too large: {int(content_length) / (1024*1024):.1f}MB")

        # Stream download with size check
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_SIZE:
                    raise RuntimeError("File too large (exceeded 300MB during download)")
                f.write(chunk)

        logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB to {output_path.name}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}") from e


def set_processor(processor_func: Callable[[str, Dict[str, Any]], None]) -> None:
    """Set the function that processes jobs.

    Args:
        processor_func: Function that takes (job_id, task_data) and processes the job.
    """
    global _processor
    _processor = processor_func


def _worker() -> None:
    """Background worker that processes jobs from the queue."""
    logger.info("Job queue worker started")

    while True:
        try:
            job_id, task_data = _work_queue.get()
            logger.info(f"[{job_id}] Processing started")

            if _processor is None:
                update_job(job_id, "failed", error="No processor configured")
                continue

            try:
                _processor(job_id, task_data)
            except Exception as e:
                logger.exception(f"[{job_id}] Processing failed: {e}")
                update_job(job_id, "failed", error=str(e))

            _work_queue.task_done()

        except Exception as e:
            logger.exception(f"Worker error: {e}")


def _cleanup_expired_jobs() -> None:
    """Background task to clean up expired jobs."""
    while True:
        time.sleep(300)  # Check every 5 minutes
        cutoff = time.time() - JOB_TTL_SECONDS

        with _job_lock:
            expired = [jid for jid, job in _jobs.items() if job.created_at < cutoff]
            for jid in expired:
                del _jobs[jid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired jobs")


def start_worker() -> None:
    """Start the background worker thread."""
    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()

    cleanup_thread = threading.Thread(target=_cleanup_expired_jobs, daemon=True)
    cleanup_thread.start()

    logger.info("Job queue worker and cleanup threads started")
