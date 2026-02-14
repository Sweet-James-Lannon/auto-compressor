"""Compatibility wrapper for legacy imports."""

from pdf_compressor.core.settings import *  # noqa: F401,F403
import pdf_compressor.core.settings as _impl


def __getattr__(name):
    return getattr(_impl, name)
