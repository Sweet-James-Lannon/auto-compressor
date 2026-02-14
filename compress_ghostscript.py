"""Compatibility wrapper for legacy imports."""

from pdf_compressor.engine.ghostscript import *  # noqa: F401,F403
import pdf_compressor.engine.ghostscript as _impl


def __getattr__(name):
    return getattr(_impl, name)
