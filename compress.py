"""PDF compression using Ghostscript with optional splitting."""

import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

from PyPDF2 import PdfReader

import compress_ghostscript
import split_pdf
from exceptions import (
    PDFCompressionError,
    EncryptionError,
    StructureError,
    MetadataCorruptionError,
    SplitError,
)
from utils import get_file_size_mb

logger = logging.getLogger(__name__)


def compress_pdf(
    input_path: str,
    working_dir: Optional[Path] = None,
    split_threshold_mb: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str, str], None]] = None
) -> Dict:
    """
    Compress a PDF file using Ghostscript, optionally splitting if too large.

    Args:
        input_path: Path to input PDF.
        working_dir: Directory for output file (default: same as input).
        split_threshold_mb: If set, split output into parts under this size.
        progress_callback: Optional callback(percent, stage, message) for progress updates.

    Returns:
        Dict with: output_path(s), original_size_mb, compressed_size_mb,
                   reduction_percent, success, compression_method,
                   was_split, output_paths (list if split).

    Raises:
        FileNotFoundError: If input file doesn't exist.
        EncryptionError: If PDF is password-protected.
        StructureError: If PDF is corrupted or malformed.
        MetadataCorruptionError: If PDF metadata is corrupted.
        SplitError: If PDF cannot be split into small enough parts.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Helper to report progress
    def report_progress(percent: int, stage: str, message: str):
        if progress_callback:
            progress_callback(percent, stage, message)

    report_progress(5, "validating", "Validating PDF...")

    # Validate PDF before processing - catch encrypted files early
    page_count = None
    try:
        with open(input_path, 'rb') as f:
            reader = PdfReader(f)
            if reader.is_encrypted:
                raise EncryptionError.for_file(input_path.name)
            page_count = len(reader.pages)
    except EncryptionError:
        raise
    except Exception as e:
        # If we can't read it at all, it might be encrypted or corrupted
        error_str = str(e).lower()
        if 'encrypt' in error_str or 'password' in error_str:
            raise EncryptionError.for_file(input_path.name) from e
        # Let Ghostscript try - it might handle some edge cases
        logger.warning(f"PDF pre-validation warning: {e}")

    working_dir = working_dir or input_path.parent
    output_path = working_dir / f"{input_path.stem}_compressed.pdf"
    original_size = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size

    logger.info(f"[SIZE_CHECK] Input: {input_path.name} = {original_bytes} bytes ({original_size:.2f}MB)")
    logger.info(f"Compressing: {input_path.name} ({original_size:.1f}MB)")

    report_progress(15, "compressing", "Compressing PDF...")

    # Try Ghostscript compression
    success, message = compress_ghostscript.compress_pdf_with_ghostscript(input_path, output_path)

    if not success:
        # Map error message to specific exception type
        msg_lower = message.lower()
        if 'password' in msg_lower or 'encrypt' in msg_lower or 'locked' in msg_lower:
            raise EncryptionError.for_file(input_path.name)
        elif 'corrupted' in msg_lower or 'damaged' in msg_lower or 'structure' in msg_lower:
            raise StructureError.for_file(input_path.name, message)
        elif 'metadata' in msg_lower or 'internal data' in msg_lower:
            raise MetadataCorruptionError.for_file(input_path.name)
        else:
            raise StructureError.for_file(input_path.name, message)

    compressed_size = get_file_size_mb(output_path)
    compressed_bytes = output_path.stat().st_size

    logger.info(f"[SIZE_CHECK] After GS: {output_path.name} = {compressed_bytes} bytes ({compressed_size:.2f}MB)")

    report_progress(60, "compressing", "Compression complete, verifying...")

    # If compression made it bigger, return original
    if compressed_size >= original_size:
        logger.warning(f"Compression increased size, returning original")
        output_path.unlink()
        shutil.copy2(input_path, output_path)
        compressed_size = original_size

        # Check if we still need to split the original
        if split_threshold_mb and compressed_size > split_threshold_mb:
            output_paths = split_pdf.split_pdf(
                output_path, working_dir, input_path.stem
            )
            return {
                "output_path": str(output_paths[0]),
                "output_paths": [str(p) for p in output_paths],
                "original_size_mb": round(original_size, 2),
                "compressed_size_mb": round(original_size, 2),
                "reduction_percent": 0.0,
                "compression_method": "none",
                "was_split": len(output_paths) > 1,
                "total_parts": len(output_paths),
                "success": True,
                "note": "Already optimized, split for size",
                "page_count": page_count,
                "part_sizes": [p.stat().st_size for p in output_paths]
            }

        return {
            "output_path": str(output_path),
            "output_paths": [str(output_path)],
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(original_size, 2),
            "reduction_percent": 0.0,
            "compression_method": "none",
            "was_split": False,
            "total_parts": 1,
            "success": True,
            "note": "Already optimized",
            "page_count": page_count,
            "part_sizes": None
        }

    reduction = ((original_size - compressed_size) / original_size) * 100

    logger.info(f"Done: {original_size:.1f}MB -> {compressed_size:.1f}MB ({reduction:.1f}%)")

    # Check if splitting is needed
    if split_threshold_mb and compressed_size > split_threshold_mb:
        logger.info(f"Compressed file still {compressed_size:.1f}MB, splitting...")
        report_progress(70, "splitting", "Splitting into parts...")

        output_paths = split_pdf.split_pdf(
            output_path, working_dir, input_path.stem
        )

        # Log size of each split part
        for i, part_path in enumerate(output_paths):
            part_bytes = part_path.stat().st_size
            part_mb = part_bytes / (1024 * 1024)
            logger.info(f"[SIZE_CHECK] Part {i+1}: {part_path.name} = {part_bytes} bytes ({part_mb:.2f}MB)")

        report_progress(95, "finalizing", "Finalizing...")
        return {
            "output_path": str(output_paths[0]),
            "output_paths": [str(p) for p in output_paths],
            "original_size_mb": round(original_size, 2),
            "compressed_size_mb": round(compressed_size, 2),
            "reduction_percent": round(reduction, 1),
            "compression_method": "ghostscript",
            "was_split": len(output_paths) > 1,
            "total_parts": len(output_paths),
            "success": True,
            "page_count": page_count,
            "part_sizes": [p.stat().st_size for p in output_paths]
        }

    # No splitting needed
    logger.info(f"[SIZE_CHECK] Final: {output_path.name} = {compressed_bytes} bytes ({compressed_size:.2f}MB)")
    report_progress(95, "finalizing", "Finalizing...")

    return {
        "output_path": str(output_path),
        "output_paths": [str(output_path)],
        "original_size_mb": round(original_size, 2),
        "compressed_size_mb": round(compressed_size, 2),
        "reduction_percent": round(reduction, 1),
        "compression_method": "ghostscript",
        "was_split": False,
        "total_parts": 1,
        "success": True,
        "page_count": page_count,
        "part_sizes": None
    }
