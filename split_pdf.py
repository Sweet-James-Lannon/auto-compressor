"""PDF splitting module for large file handling.

This module provides functionality to split PDFs that exceed
the size threshold into smaller parts for email attachment limits.
"""

import logging
import os
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError

from compress_ghostscript import optimize_split_part, compress_pdf_with_ghostscript
from exceptions import EncryptionError, StructureError, SplitError

# Constants
SPLIT_THRESHOLD_MB: float = float(os.environ.get('SPLIT_THRESHOLD_MB', '25'))
SAFETY_BUFFER_MB: float = 0.5  # Target 24.5MB to ensure we're under 25MB limit
MAX_SPLIT_PARTS: int = int(os.environ.get('MAX_SPLIT_PARTS', '50'))  # Allow more splits for large files

logger = logging.getLogger(__name__)


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.
    """
    return path.stat().st_size / (1024 * 1024)


def needs_splitting(pdf_path: Path, threshold_mb: float = SPLIT_THRESHOLD_MB) -> bool:
    """Check if PDF exceeds size threshold and needs splitting.

    Args:
        pdf_path: Path to the PDF file.
        threshold_mb: Size threshold in MB. Defaults to SPLIT_THRESHOLD_MB.

    Returns:
        True if file exceeds threshold, False otherwise.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False

    size_mb = get_file_size_mb(pdf_path)
    return size_mb > threshold_mb


def split_pdf(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float = SPLIT_THRESHOLD_MB
) -> List[Path]:
    """Split PDF into parts based on page count estimation, then compress each part.

    Calculates number of parts needed based on file size, then divides
    pages evenly. After splitting, each part is compressed with Ghostscript
    to counteract PyPDF2's size inflation (resource duplication).

    This ensures a 43MB file splits into 2 parts of ~21MB each, not 2 parts
    of ~40MB each.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory to write output files.
        base_name: Base filename for output parts (e.g., "doc" -> "doc_part1.pdf").
        threshold_mb: Maximum size per part in MB. Defaults to SPLIT_THRESHOLD_MB.

    Returns:
        List of paths to the created PDF parts, all guaranteed under threshold_mb.

    Raises:
        FileNotFoundError: If source PDF doesn't exist.
        EncryptionError: If PDF is password-protected.
        StructureError: If PDF is corrupted or malformed.
        SplitError: If PDF cannot be split into parts under threshold.
    """
    import math

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = get_file_size_mb(pdf_path)

    # If already under threshold, just return the original
    if file_size_mb <= threshold_mb:
        logger.info(f"PDF already under {threshold_mb}MB, no split needed")
        return [pdf_path]

    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)

            # Check for encrypted PDFs FIRST - don't waste time on password-protected files
            if reader.is_encrypted:
                raise EncryptionError.for_file(pdf_path.name)

            total_pages = len(reader.pages)

            if total_pages == 0:
                raise StructureError.for_file(pdf_path.name, "PDF has no pages")

            # Start with minimum parts needed mathematically
            num_parts = math.ceil(file_size_mb / threshold_mb)

            # Allow up to 2x the minimum parts to handle compression variability
            # Cap at MAX_SPLIT_PARTS or total_pages (can't have more parts than pages)
            max_attempts = min(num_parts * 2, MAX_SPLIT_PARTS, total_pages)

            for attempt in range(num_parts, max_attempts + 1):
                pages_per_part = math.ceil(total_pages / attempt)

                logger.info(f"Attempt {attempt - num_parts + 1}: Splitting {pdf_path.name}: "
                           f"{total_pages} pages, {file_size_mb:.1f}MB -> {attempt} parts")

                output_paths: List[Path] = []
                all_under_threshold = True

                for part_num in range(1, attempt + 1):
                    start_page = (part_num - 1) * pages_per_part
                    end_page = min(part_num * pages_per_part, total_pages)

                    if start_page >= total_pages:
                        break

                    writer = PdfWriter()
                    for page_idx in range(start_page, end_page):
                        writer.add_page(reader.pages[page_idx])

                    # Write uncompressed part to temporary path
                    temp_part_path = output_dir / f"{base_name}_part{part_num}_temp.pdf"
                    with open(temp_part_path, 'wb') as out_f:
                        writer.write(out_f)

                    uncompressed_size_mb = get_file_size_mb(temp_part_path)
                    pages_in_part = end_page - start_page
                    logger.info(f"Created {temp_part_path.name}: {pages_in_part} pages, {uncompressed_size_mb:.1f}MB")

                    # Only optimize if part exceeds threshold (Ghostscript can sometimes make files bigger)
                    part_path = output_dir / f"{base_name}_part{part_num}.pdf"

                    if uncompressed_size_mb > threshold_mb:
                        # Part exceeds threshold - try full compression first
                        logger.info(f"Part {part_num} exceeds threshold ({uncompressed_size_mb:.1f}MB), compressing...")
                        success, message = compress_pdf_with_ghostscript(temp_part_path, part_path)
                        if success:
                            compressed_size_mb = get_file_size_mb(part_path)
                            # If compression made it bigger, try optimization instead
                            if compressed_size_mb > uncompressed_size_mb:
                                logger.info(f"Compression made bigger, trying optimization...")
                                part_path.unlink()
                                success2, _ = optimize_split_part(temp_part_path, part_path)
                                if success2:
                                    compressed_size_mb = get_file_size_mb(part_path)
                                    if compressed_size_mb > uncompressed_size_mb:
                                        part_path.unlink()
                                        temp_part_path.rename(part_path)
                                        compressed_size_mb = uncompressed_size_mb
                                    else:
                                        temp_part_path.unlink()
                                else:
                                    temp_part_path.rename(part_path)
                                    compressed_size_mb = uncompressed_size_mb
                            else:
                                logger.info(f"Compressed {part_path.name}: {compressed_size_mb:.1f}MB ({message})")
                                temp_part_path.unlink()
                        else:
                            logger.warning(f"Compression failed: {message}")
                            temp_part_path.rename(part_path)
                            compressed_size_mb = uncompressed_size_mb
                    else:
                        # Part already under threshold, just rename
                        temp_part_path.rename(part_path)
                        compressed_size_mb = uncompressed_size_mb
                        logger.info(f"Part {part_num} already under threshold: {compressed_size_mb:.1f}MB")

                    # Use safety buffer to ensure we're WELL under the hard limit
                    target_size = threshold_mb - SAFETY_BUFFER_MB  # 24.5MB target
                    if compressed_size_mb > target_size:
                        all_under_threshold = False

                    output_paths.append(part_path)

                if all_under_threshold:
                    logger.info(f"Split complete: {len(output_paths)} parts, all under {threshold_mb}MB")
                    return output_paths
                elif attempt < max_attempts:
                    # Clean up and try with more parts
                    logger.info(f"Some parts over threshold, trying with {attempt + 1} parts...")
                    for p in output_paths:
                        if p.exists():
                            p.unlink()
                    output_paths = []
                else:
                    # CRITICAL: NEVER return files over threshold - clean up and raise error
                    logger.error(f"Could not split into parts under {threshold_mb}MB after {max_attempts - num_parts + 1} attempts")

                    # Clean up all created files
                    for p in output_paths:
                        if p.exists():
                            p.unlink()

                    # Raise clear error explaining what happened
                    raise SplitError.for_file(
                        pdf_path.name,
                        threshold_mb,
                        max_attempts - num_parts + 1
                    )

            # Should not reach here, but just in case
            return output_paths

    except (EncryptionError, StructureError, SplitError):
        # Re-raise our custom exceptions as-is (already have good messages)
        raise
    except PdfReadError as e:
        # Translate PyPDF2 errors to our custom exceptions
        error_str = str(e).lower()
        if 'encrypt' in error_str or 'password' in error_str:
            raise EncryptionError.for_file(pdf_path.name) from e
        raise StructureError.for_file(pdf_path.name, str(e)) from e
    except Exception as e:
        logger.error(f"Failed to split PDF: {e}")
        raise StructureError.for_file(pdf_path.name, str(e)) from e
