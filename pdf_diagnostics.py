"""Compatibility wrapper for legacy imports."""

from pdf_compressor.engine.pdf_diagnostics import *  # noqa: F401,F403
import pdf_compressor.engine.pdf_diagnostics as _impl


def __getattr__(name):
    return getattr(_impl, name)
