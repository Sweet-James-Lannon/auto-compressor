"""PDF compression using Ghostscript. Simple, working code."""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import compress_ghostscript

logger = logging.getLogger(__name__)


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def compress_pdf(
    input_path: str,
    working_dir: Optional[Path] = None
) -> Dict:
    """
    Compress a PDF file using Ghostscript.

    Args:
        input_path: Path to input PDF
        working_dir: Directory for output file (default: same as input)

    Returns:
        Dict with: output_path, original_size_mb, compressed_size_mb,
                   reduction_percent, success, compression_method

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If compression fails
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    working_dir = working_dir or input_path.parent
    output_path = working_dir / f"{input_path.stem}_compressed.pdf"
    original_size = get_file_size_mb(input_path)

    logger.info(f"Compressing: {input_path.name} ({original_size:.1f}MB)")

    # Try Ghostscript compression
    success, message = compress_ghostscript.compress_pdf_with_ghostscript(input_path, output_path)

    if not success:
        raise RuntimeError(f"Compression failed: {message}")

    compressed_size = get_file_size_mb(output_path)

    # If compression made it bigger, return original
    if compressed_size >= original_size:
        logger.warning(f"Compression increased size, returning original")
        output_path.unlink()
        shutil.copy2(input_path, output_path)
        return {
            "output_path": str(output_path),
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(original_size, 2),
            "reduction_percent": 0.0,
            "compression_method": "none",
            "success": True,
            "note": "Already optimized"
        }

    reduction = ((original_size - compressed_size) / original_size) * 100

    logger.info(f"Done: {original_size:.1f}MB -> {compressed_size:.1f}MB ({reduction:.1f}%)")

    return {
        "output_path": str(output_path),
        "original_size_mb": round(original_size, 2),
        "compressed_size_mb": round(compressed_size, 2),
        "reduction_percent": round(reduction, 1),
        "compression_method": "ghostscript",
        "success": True
    }
