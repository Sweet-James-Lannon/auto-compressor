"""PDF splitting module for large file handling.

This module provides functionality to split PDFs that exceed
the size threshold into smaller parts for email attachment limits.
"""

import logging
import math
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
    """Split PDF into optimal number of parts using smart byte-based distribution.

    Uses Adobe-style splitting: calculates exact parts needed (ceil(size/threshold)),
    then distributes pages to achieve balanced part sizes. Estimates page sizes
    from content streams to ensure each part is roughly (total_size / num_parts).

    Example: 53MB file -> ceil(53/24.5) = 3 parts of ~17.8MB each.

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

            # SIMPLE: compressed_size ÷ 25MB = number of parts
            num_parts = math.ceil(file_size_mb / threshold_mb)
            target_size = threshold_mb - SAFETY_BUFFER_MB  # 24.5MB max per part
            logger.info(f"Split: {file_size_mb:.2f}MB ÷ {threshold_mb}MB = {num_parts} parts needed")

            def measure_pages(start_page: int, end_page: int, optimize: bool = True) -> float:
                """Create a temp PDF, optionally optimize it, and measure its size.

                Args:
                    start_page: Starting page index (0-based, inclusive)
                    end_page: Ending page index (0-based, exclusive)
                    optimize: If True, apply Ghostscript optimization to get accurate size
                """
                writer = PdfWriter()
                for idx in range(start_page, end_page):
                    writer.add_page(reader.pages[idx])
                temp_path = output_dir / f"{base_name}_measure_temp.pdf"
                with open(temp_path, 'wb') as f:
                    writer.write(f)

                if optimize:
                    # Apply Ghostscript optimization to get accurate final size
                    optimized_path = output_dir / f"{base_name}_measure_opt.pdf"
                    success, _ = optimize_split_part(temp_path, optimized_path)
                    if success and optimized_path.exists():
                        size = get_file_size_mb(optimized_path)
                        optimized_path.unlink()
                    else:
                        size = get_file_size_mb(temp_path)
                else:
                    size = get_file_size_mb(temp_path)

                temp_path.unlink(missing_ok=True)
                return size

            # Try to find split points, increasing num_parts if needed
            max_attempts = 10  # Prevent infinite loops
            groups = None

            for attempt in range(max_attempts):
                if attempt > 0:
                    num_parts += 1
                    logger.info(f"Attempt {attempt + 1}: Trying with {num_parts} parts...")

                # Find split points using binary search
                split_points = [0]  # Start of first part

                for part_idx in range(num_parts - 1):
                    start = split_points[-1]
                    remaining_parts = num_parts - part_idx
                    remaining_pages = total_pages - start

                    # Binary search for the page where this part is just under target_size
                    low = start + 1
                    high = start + remaining_pages - (remaining_parts - 1)  # Leave pages for other parts
                    best_split = low

                    logger.info(f"Finding split point {part_idx + 1}: searching pages {low}-{high}")

                    while low <= high:
                        mid = (low + high) // 2
                        size = measure_pages(start, mid)
                        logger.info(f"  Pages {start}-{mid}: {size:.1f}MB")

                        if size <= target_size:
                            best_split = mid
                            low = mid + 1  # Try more pages
                        else:
                            high = mid - 1  # Too big, try fewer pages

                    split_points.append(best_split)
                    logger.info(f"Split point {part_idx + 1}: page {best_split}")

                split_points.append(total_pages)  # End of last part

                # Build groups from split points
                groups = []
                for i in range(len(split_points) - 1):
                    groups.append(list(range(split_points[i], split_points[i + 1])))

                logger.info(f"Candidate split: {[f'pages {g[0]+1}-{g[-1]+1}' for g in groups]}")

                # CRITICAL: Verify ALL parts including the last one
                all_parts_valid = True
                for part_idx, group in enumerate(groups, 1):
                    part_size = measure_pages(group[0], group[-1] + 1)
                    logger.info(f"Verifying Part {part_idx} (pages {group[0]+1}-{group[-1]+1}): {part_size:.1f}MB")

                    if part_size > threshold_mb:
                        logger.warning(f"Part {part_idx} exceeds {threshold_mb}MB! Need more parts.")
                        all_parts_valid = False
                        break

                if all_parts_valid:
                    logger.info(f"All {num_parts} parts verified under {threshold_mb}MB")
                    break

            if not all_parts_valid:
                raise SplitError(
                    f"Could not split PDF into parts under {threshold_mb}MB after {max_attempts} attempts. "
                    f"The PDF may have individual pages that are too large."
                )

            # Now create the actual split parts - ALWAYS optimize to match measured sizes
            output_paths: List[Path] = []

            for part_num, page_indices in enumerate(groups, 1):
                writer = PdfWriter()
                for page_idx in page_indices:
                    writer.add_page(reader.pages[page_idx])

                # Write raw PyPDF2 part to temporary path
                temp_part_path = output_dir / f"{base_name}_part{part_num}_temp.pdf"
                with open(temp_part_path, 'wb') as out_f:
                    writer.write(out_f)

                raw_size_mb = get_file_size_mb(temp_part_path)
                pages_in_part = len(page_indices)
                logger.info(f"Created {temp_part_path.name}: {pages_in_part} pages, {raw_size_mb:.1f}MB (raw)")

                part_path = output_dir / f"{base_name}_part{part_num}.pdf"

                # ALWAYS optimize - this matches what we measured during binary search
                success, message = optimize_split_part(temp_part_path, part_path)
                if success:
                    optimized_size_mb = get_file_size_mb(part_path)
                    logger.info(f"Optimized {part_path.name}: {raw_size_mb:.1f}MB -> {optimized_size_mb:.1f}MB")
                    temp_part_path.unlink()

                    # If still over threshold after optimization, try full compression
                    if optimized_size_mb > target_size:
                        logger.info(f"Part {part_num} still {optimized_size_mb:.1f}MB, trying full compression...")
                        compressed_path = part_path.with_suffix('.compressed.pdf')
                        success2, msg2 = compress_pdf_with_ghostscript(part_path, compressed_path)
                        if success2:
                            new_size = get_file_size_mb(compressed_path)
                            if new_size < optimized_size_mb:
                                part_path.unlink()
                                compressed_path.rename(part_path)
                                logger.info(f"Full compression: {optimized_size_mb:.1f}MB -> {new_size:.1f}MB")
                            else:
                                compressed_path.unlink()
                        else:
                            logger.warning(f"Full compression failed: {msg2}")
                else:
                    # Optimization failed - just rename
                    logger.warning(f"Optimization failed: {message}")
                    temp_part_path.rename(part_path)

                output_paths.append(part_path)

            # Return the parts - all verified under threshold
            logger.info(f"Split complete: {len(output_paths)} parts")
            for i, p in enumerate(output_paths):
                size = get_file_size_mb(p)
                logger.info(f"  Part {i+1}: {size:.1f}MB")
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
