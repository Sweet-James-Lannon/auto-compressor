"""Application configuration loading."""

from __future__ import annotations

from dataclasses import dataclass

from pdf_compressor.core.utils import MAX_DOWNLOAD_SIZE


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration values consumed by the Flask app."""

    max_content_length: int = MAX_DOWNLOAD_SIZE


def load_runtime_config() -> RuntimeConfig:
    """Load runtime configuration from environment-backed module constants."""
    return RuntimeConfig()
