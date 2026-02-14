"""Runtime bootstrap for background workers and cleanup daemons."""

from __future__ import annotations

import threading

from pdf_compressor.services import compression_service

_bootstrap_lock = threading.Lock()
_bootstrap_started = False


def bootstrap_runtime() -> None:
    """Start background services once per process."""
    global _bootstrap_started
    with _bootstrap_lock:
        if _bootstrap_started:
            return

        threading.Thread(
            target=compression_service.cleanup_daemon,
            daemon=True,
            name="compression-cleanup-daemon",
        ).start()
        compression_service.job_queue.set_processor(compression_service.process_compression_job)
        compression_service.job_queue.start_workers(compression_service.ASYNC_WORKERS)
        _bootstrap_started = True


def is_bootstrapped() -> bool:
    """Expose runtime bootstrap state for diagnostics/tests."""
    return _bootstrap_started
