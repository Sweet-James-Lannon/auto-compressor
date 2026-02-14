"""File tracking and cleanup service wrappers."""

from pathlib import Path

from pdf_compressor.services import compression_service


def track_file(path: Path) -> None:
    compression_service.track_file(path)


def cleanup_daemon() -> None:
    compression_service.cleanup_daemon()
