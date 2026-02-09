"""Job queue module for async PDF processing.

Provides job queue and status tracking for long-running compression tasks.
Handles background processing to avoid blocking the main request threads.
"""

import logging
import os
import queue
import threading
import time
import uuid
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Import download_pdf from utils to avoid duplication
from utils import download_pdf  # noqa: F401 - re-exported for backwards compatibility

# Constants
JOB_TTL_SECONDS: int = 3600  # Jobs expire after 1 hour
MAX_QUEUE_SIZE: int = int(os.environ.get("MAX_QUEUE_SIZE", "50"))  # Prevent unbounded growth
JOB_STATE_FILE = os.environ.get("JOB_STATE_FILE", "").strip()

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Represents an async processing job."""

    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    created_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None  # Real-time progress info


# Module-level state
_jobs: Dict[str, Job] = {}
_job_lock = threading.Lock()
_work_queue: queue.Queue = queue.Queue()
_processor: Optional[Callable] = None
_workers_started = False
_worker_count = 0
_inflight_by_key: Dict[str, str] = {}
_job_key: Dict[str, str] = {}


def _is_inflight_status(status: str) -> bool:
    return status in {"queued", "processing"}


def _job_to_dict(job: Job) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "result": job.result,
        "error": job.error,
        "progress": job.progress,
    }


def _job_from_dict(payload: Dict[str, Any]) -> Optional[Job]:
    try:
        job_id = str(payload["job_id"])
        status = str(payload["status"])
        created_at = float(payload.get("created_at", time.time()))
    except (KeyError, TypeError, ValueError):
        return None

    return Job(
        job_id=job_id,
        status=status,
        created_at=created_at,
        result=payload.get("result"),
        error=payload.get("error"),
        progress=payload.get("progress"),
    )


def _persist_jobs() -> None:
    if not JOB_STATE_FILE:
        return

    state_path = Path(JOB_STATE_FILE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")

    with _job_lock:
        payload = [_job_to_dict(job) for job in _jobs.values()]

    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp_path.replace(state_path)
    except Exception as exc:
        logger.warning("Failed to persist job state: %s", exc)
        tmp_path.unlink(missing_ok=True)


def _load_jobs() -> None:
    if not JOB_STATE_FILE:
        return

    state_path = Path(JOB_STATE_FILE)
    if not state_path.exists():
        return

    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load job state: %s", exc)
        return

    if not isinstance(payload, list):
        return

    restored = 0
    with _job_lock:
        for item in payload:
            if not isinstance(item, dict):
                continue
            job = _job_from_dict(item)
            if job is None:
                continue
            _jobs[job.job_id] = job
            restored += 1

    if restored:
        logger.info("Restored %s job(s) from durable state", restored)


def create_job() -> str:
    """Create a new job and return its ID.

    Returns:
        The unique job ID (16 characters for sufficient entropy).
    """
    job_id, _ = create_or_reuse_job()
    return job_id


def create_or_reuse_job(dedupe_key: Optional[str] = None) -> tuple[str, bool]:
    """Create a new job or reuse an in-flight job when dedupe_key matches.

    Args:
        dedupe_key: Optional stable key identifying equivalent work.

    Returns:
        Tuple of (job_id, reused_existing).
    """
    with _job_lock:
        if dedupe_key:
            existing_job_id = _inflight_by_key.get(dedupe_key)
            if existing_job_id:
                existing_job = _jobs.get(existing_job_id)
                if existing_job and _is_inflight_status(existing_job.status):
                    logger.info("[%s] Reusing in-flight job via dedupe key", existing_job_id)
                    return existing_job_id, True
                _inflight_by_key.pop(dedupe_key, None)

        job_id = str(uuid.uuid4()).replace('-', '')[:16]
        job = Job(
            job_id=job_id,
            status="queued",
            progress={
                "percent": 0,
                "stage": "queued",
                "message": "Waiting for worker",
            },
        )
        _jobs[job_id] = job
        if dedupe_key:
            _inflight_by_key[dedupe_key] = job_id
            _job_key[job_id] = dedupe_key

    _persist_jobs()
    logger.info(f"[{job_id}] Job created")
    return job_id, False


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
        status: New status ("queued", "processing", "completed", "failed").
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
            if not _is_inflight_status(status):
                key = _job_key.pop(job_id, None)
                if key:
                    current = _inflight_by_key.get(key)
                    if current == job_id:
                        _inflight_by_key.pop(key, None)
    _persist_jobs()

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
    if MAX_QUEUE_SIZE and _work_queue.qsize() >= MAX_QUEUE_SIZE:
        logger.warning(f"[{job_id}] Queue full ({MAX_QUEUE_SIZE}), rejecting job")
        raise queue.Full("Job queue is full")

    _work_queue.put((job_id, task_data))
    logger.info(f"[{job_id}] Job enqueued (size={_work_queue.qsize()})")


def set_processor(processor_func: Callable[[str, Dict[str, Any]], None]) -> None:
    """Set the function that processes jobs.

    Args:
        processor_func: Function that takes (job_id, task_data) and processes the job.
    """
    global _processor
    _processor = processor_func


def get_stats() -> Dict[str, Any]:
    """Return lightweight queue and job stats for health/debug endpoints."""
    with _job_lock:
        total_jobs = len(_jobs)
        status_counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
        for job in _jobs.values():
            status_counts[job.status] = status_counts.get(job.status, 0) + 1

    return {
        "queue_size": _work_queue.qsize(),
        "queue_max": MAX_QUEUE_SIZE,
        "total_jobs": total_jobs,
        "queued": status_counts.get("queued", 0),
        "processing": status_counts.get("processing", 0),
        "completed": status_counts.get("completed", 0),
        "failed": status_counts.get("failed", 0),
        "workers_started": _workers_started,
        "worker_count": _worker_count,
        "inflight_dedupe_keys": len(_inflight_by_key),
    }


def get_recent_jobs(limit: int = 10) -> list[Dict[str, Any]]:
    """Return a small list of recent jobs for dashboard visibility."""
    now = time.time()
    with _job_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)[:max(0, limit)]
        summaries = []
        for job in jobs:
            summaries.append({
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at,
                "age_seconds": int(now - job.created_at),
                "progress": job.progress,
                "result": job.result,
                "error": job.error,
            })
    return summaries


def _worker() -> None:
    """Background worker that processes jobs from the queue."""
    logger.info("Job queue worker started")

    while True:
        try:
            job_id, task_data = _work_queue.get()
            update_job(job_id, "processing", progress={
                "percent": 1,
                "stage": "starting",
                "message": "Starting worker",
            })
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
                key = _job_key.pop(jid, None)
                if key and _inflight_by_key.get(key) == jid:
                    _inflight_by_key.pop(key, None)
                del _jobs[jid]
        if expired:
            _persist_jobs()

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired jobs")


def start_workers(num_workers: int = 8) -> None:
    """Start multiple background worker threads."""
    global _workers_started, _worker_count
    if os.environ.get("DISABLE_ASYNC_WORKERS", "").lower() in ("1", "true", "yes"):
        logger.info("Async workers disabled by DISABLE_ASYNC_WORKERS")
        return
    if _workers_started:
        logger.info("Workers already started, skipping duplicate start")
        return

    for i in range(num_workers):
        worker_thread = threading.Thread(target=_worker, daemon=True, name=f"compression-worker-{i}")
        worker_thread.start()
    
    cleanup_thread = threading.Thread(target=_cleanup_expired_jobs, daemon=True)
    cleanup_thread.start()
    
    _worker_count = num_workers
    logger.info(f"Started {num_workers} compression workers and cleanup thread")
    _workers_started = True


_load_jobs()
