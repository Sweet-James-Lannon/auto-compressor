"""PDF splitting module for large file handling.

This module provides functionality to split PDFs that exceed
the size threshold into smaller parts for email attachment limits.
"""

import logging
import os
from pathlib import Path
from typing import List

from PyPDF2 import PdfReader, PdfWriter

# Constants
SPLIT_THRESHOLD_MB: float = float(os.environ.get('SPLIT_THRESHOLD_MB', '25'))
BUFFER_PERCENTAGE: float = 0.9  # Target 90% of threshold for safety margin

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
    """Split PDF into parts under the size threshold.

    Splits a PDF file into multiple smaller PDFs, each under the specified
    size threshold. Uses a binary search approach to find optimal page
    groupings.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory to write output files.
        base_name: Base filename for output parts (e.g., "doc" -> "doc_part1.pdf").
        threshold_mb: Maximum size per part in MB. Defaults to SPLIT_THRESHOLD_MB.

    Returns:
        List of paths to the created PDF parts.

    Raises:
        FileNotFoundError: If source PDF doesn't exist.
        RuntimeError: If splitting fails.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # If already under threshold, just return the original
    if not needs_splitting(pdf_path, threshold_mb):
        logger.info(f"PDF already under {threshold_mb}MB, no split needed")
        return [pdf_path]

    target_size = threshold_mb * BUFFER_PERCENTAGE * 1024 * 1024  # Convert to bytes

    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)

            if total_pages == 0:
                raise RuntimeError("PDF has no pages")

            logger.info(f"Splitting {pdf_path.name}: {total_pages} pages, "
                       f"{get_file_size_mb(pdf_path):.1f}MB")

            output_paths: List[Path] = []
            current_page = 0
            part_num = 1

            while current_page < total_pages:
                writer = PdfWriter()
                part_path = output_dir / f"{base_name}_part{part_num}.pdf"

                # Track the starting page for this part
                start_page = current_page

                # Add pages until we approach the target size
                pages_added = 0
                while current_page < total_pages:
                    writer.add_page(reader.pages[current_page])
                    pages_added += 1
                    current_page += 1

                    # Write to temp to check size
                    with open(part_path, 'wb') as out_f:
                        writer.write(out_f)

                    current_size = part_path.stat().st_size

                    # If we exceeded target and have more than 1 page, back up
                    if current_size > target_size and pages_added > 1:
                        # Remove the last page and rewrite
                        current_page -= 1
                        pages_added -= 1

                        writer = PdfWriter()
                        # Use start_page to ensure correct page range
                        for i in range(start_page, start_page + pages_added):
                            writer.add_page(reader.pages[i])

                        with open(part_path, 'wb') as out_f:
                            writer.write(out_f)
                        break

                    # If single page exceeds limit, log warning but continue
                    if current_size > target_size and pages_added == 1:
                        logger.warning(
                            f"Single page {current_page} exceeds {threshold_mb}MB "
                            f"({current_size / (1024*1024):.1f}MB) - cannot split further"
                        )
                        break

                output_paths.append(part_path)
                part_size_mb = get_file_size_mb(part_path)
                logger.info(f"Created {part_path.name}: {pages_added} pages, {part_size_mb:.1f}MB")
                part_num += 1

            logger.info(f"Split complete: {len(output_paths)} parts created")
            return output_paths

    except Exception as e:
        logger.error(f"Failed to split PDF: {e}")
        raise RuntimeError(f"PDF split failed: {e}") from e
