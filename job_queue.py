"""Compatibility wrapper for legacy imports."""

from pdf_compressor.workers.job_queue import *  # noqa: F401,F403
import pdf_compressor.workers.job_queue as _impl


def __getattr__(name):
    return getattr(_impl, name)
