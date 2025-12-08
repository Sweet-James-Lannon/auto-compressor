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
from typing import Any, Callable, Dict, Optional

# Import download_pdf from utils to avoid duplication
from utils import download_pdf  # noqa: F401 - re-exported for backwards compatibility

# Constants
JOB_TTL_SECONDS: int = 3600  # Jobs expire after 1 hour

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents an async processing job."""

    job_id: str
    status: str  # "processing", "completed", "failed"
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None  # Real-time progress info


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
    error: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None
) -> None:
    """Update a job's status and result.

    Args:
        job_id: The job identifier.
        status: New status ("processing", "completed", "failed").
        result: Optional result data for completed jobs.
        error: Optional error message for failed jobs.
        progress: Optional progress data (percent, stage, message).
    """
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].status = status
            _jobs[job_id].result = result
            _jobs[job_id].error = error
            if progress is not None:
                _jobs[job_id].progress = progress

    if progress:
        logger.debug(f"[{job_id}] Progress: {progress.get('percent', 0)}% - {progress.get('stage', 'unknown')}")
    else:
        logger.info(f"[{job_id}] Status updated: {status}")


def enqueue(job_id: str, task_data: Dict[str, Any]) -> None:
    """Add a job to the processing queue.

    Args:
        job_id: The job identifier.
        task_data: Data needed to process the job (pdf_bytes, download_url, etc).
    """
    _work_queue.put((job_id, task_data))
    logger.info(f"[{job_id}] Job enqueued")


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


def start_workers(num_workers: int = 8) -> None:
    """Start multiple background worker threads."""
    for i in range(num_workers):
        worker_thread = threading.Thread(target=_worker, daemon=True, name=f"compression-worker-{i}")
        worker_thread.start()
    
    cleanup_thread = threading.Thread(target=_cleanup_expired_jobs, daemon=True)
    cleanup_thread.start()
    
    logger.info(f"Started {num_workers} compression workers and cleanup thread")
