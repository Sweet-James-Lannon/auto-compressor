"""Shared utility functions for PDF compression service."""

from pathlib import Path


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.
    """
    return path.stat().st_size / (1024 * 1024)
