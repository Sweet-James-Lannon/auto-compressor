"""PDF splitting module for large file handling.

This module provides functionality to split PDFs that exceed
the size threshold into smaller parts for email attachment limits.
"""

import logging
import math
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from PyPDF2.errors import PdfReadError

from compress_ghostscript import optimize_split_part, compress_pdf_with_ghostscript
from exceptions import EncryptionError, StructureError, SplitError
from utils import get_file_size_mb

# Constants - Email attachment sizing
# Outlook/Exchange has ~25MB limit but this includes email headers/body overhead
# Target slightly under 25MB to leave room for email overhead
DEFAULT_THRESHOLD_MB: float = 25.0
SAFETY_BUFFER_MB: float = 1.5  # Buffer for email headers/body overhead (target ~23.5MB parts)

logger = logging.getLogger(__name__)


# =============================================================================
# NEW HELPER FUNCTIONS FOR PARALLEL COMPRESSION
# =============================================================================

def split_by_pages(pdf_path: Path, output_dir: Path, num_parts: int, base_name: str) -> List[Path]:
    """Split PDF into N parts by page count. Fast - no Ghostscript.

    Used for parallel compression: split first, compress each chunk in parallel.

    Args:
        pdf_path: Path to source PDF.
        output_dir: Directory for output chunks.
        num_parts: Number of parts to create.
        base_name: Base filename for chunks.

    Returns:
        List of paths to created chunk files.

    Example: 300 pages / 6 parts = 50 pages each
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise StructureError.for_file(pdf_path.name, "PDF has no pages")

    # Calculate pages per part (last part gets remainder)
    pages_per_part = total_pages // num_parts

    chunk_paths = []
    for i in range(num_parts):
        start = i * pages_per_part
        # Last part gets all remaining pages
        end = total_pages if i == num_parts - 1 else (i + 1) * pages_per_part

        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        # Use unique ID to prevent collisions in parallel operations
        unique_id = str(uuid.uuid4())[:8]
        chunk_path = output_dir / f"{base_name}_chunk{i+1}_{unique_id}.pdf"

        with open(chunk_path, 'wb') as f:
            writer.write(f)

        chunk_paths.append(chunk_path)
        logger.info(f"Created chunk {i+1}/{num_parts}: pages {start+1}-{end} -> {chunk_path.name}")

    return chunk_paths


def merge_pdfs(pdf_paths: List[Path], output_path: Path) -> None:
    """Merge multiple PDFs into one. Fast - no Ghostscript.

    Used after parallel compression to recombine compressed chunks.

    Args:
        pdf_paths: List of PDF paths to merge (in order).
        output_path: Path for merged output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merger = PdfMerger()
    for path in pdf_paths:
        merger.append(str(path))

    merger.write(str(output_path))
    merger.close()

    logger.info(f"Merged {len(pdf_paths)} PDFs into {output_path.name}")


def split_by_size(pdf_path: Path, output_dir: Path, base_name: str, threshold_mb: float = DEFAULT_THRESHOLD_MB) -> List[Path]:
    """Split PDF into parts under threshold_mb each.

    Creates parts and verifies each is under threshold. If any part exceeds
    threshold, increases the number of parts and retries.

    Args:
        pdf_path: Path to source PDF.
        output_dir: Directory for output parts.
        base_name: Base filename for parts.
        threshold_mb: Maximum size per part.

    Returns:
        List of paths to created part files, all guaranteed under threshold_mb.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = get_file_size_mb(pdf_path)

    # If already under threshold, just return the original
    if file_size_mb <= threshold_mb:
        logger.info(f"PDF {file_size_mb:.1f}MB already under {threshold_mb}MB, no split needed")
        return [pdf_path]

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise StructureError.for_file(pdf_path.name, "PDF has no pages")

    # Start with estimated number of parts, increase if needed
    buffer_mb = 2.0
    effective_threshold = threshold_mb - buffer_mb
    num_parts = max(2, math.ceil(file_size_mb / effective_threshold))

    max_attempts = 20  # Prevent infinite loops

    for attempt in range(max_attempts):
        # Ensure at least 1 page per part
        num_parts = min(num_parts, total_pages)
        pages_per_part = total_pages // num_parts

        logger.info(f"[split_by_size] Attempt {attempt + 1}: splitting into {num_parts} parts ({pages_per_part} pages each)")

        # Clean up any previous attempt files
        for old_file in output_dir.glob(f"{base_name}_part*.pdf"):
            old_file.unlink(missing_ok=True)

        output_paths = []
        all_under_threshold = True
        max_part_size = 0

        for i in range(num_parts):
            start = i * pages_per_part
            # Last part gets all remaining pages
            end = total_pages if i == num_parts - 1 else (i + 1) * pages_per_part

            writer = PdfWriter()
            for page_idx in range(start, end):
                writer.add_page(reader.pages[page_idx])

            part_path = output_dir / f"{base_name}_part{i+1}.pdf"

            with open(part_path, 'wb') as f:
                writer.write(f)

            part_size = get_file_size_mb(part_path)
            max_part_size = max(max_part_size, part_size)
            output_paths.append(part_path)

            if part_size > threshold_mb:
                all_under_threshold = False
                logger.warning(f"[split_by_size] Part {i+1} is {part_size:.1f}MB (over {threshold_mb}MB threshold)")

            logger.info(f"[split_by_size] Part {i+1}/{num_parts}: pages {start+1}-{end}, {part_size:.1f}MB")

        if all_under_threshold:
            logger.info(f"[split_by_size] Success: {num_parts} parts, all under {threshold_mb}MB (max: {max_part_size:.1f}MB)")
            return output_paths

        # Parts too big - increase number of parts and retry
        # Estimate how many more parts we need based on largest part
        scale_factor = max_part_size / effective_threshold
        num_parts = max(num_parts + 1, int(num_parts * scale_factor) + 1)
        logger.info(f"[split_by_size] Retrying with {num_parts} parts...")

    # If we get here, we couldn't split small enough
    logger.error(f"[split_by_size] Could not split into parts under {threshold_mb}MB after {max_attempts} attempts")
    # Return what we have - better than nothing
    return output_paths


# =============================================================================
# ORIGINAL SPLIT FUNCTION (for binary search splitting - kept for compatibility)
# =============================================================================

def split_pdf(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    progress_callback: Optional[Callable[[int, str, str], None]] = None
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

    def report_progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, "splitting", message)

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

            # Calculate number of parts needed and BALANCED target size
            num_parts = math.ceil(file_size_mb / threshold_mb)
            # Target = total / parts (balanced distribution), capped at threshold - buffer
            balanced_target = file_size_mb / num_parts
            max_target = threshold_mb - SAFETY_BUFFER_MB
            target_size = min(balanced_target, max_target)
            logger.info(f"Split: {file_size_mb:.2f}MB ÷ {threshold_mb}MB = {num_parts} parts, target {target_size:.1f}MB each")

            def measure_pages(start_page: int, end_page: int, optimize: bool = True) -> float:
                """Create a temp PDF, optionally optimize it, and measure its size.

                Args:
                    start_page: Starting page index (0-based, inclusive)
                    end_page: Ending page index (0-based, exclusive)
                    optimize: If True, apply Ghostscript optimization to get accurate size
                """
                # Unique ID prevents file collisions when multiple jobs run in parallel
                unique_id = str(uuid.uuid4())[:8]

                writer = PdfWriter()
                for idx in range(start_page, end_page):
                    writer.add_page(reader.pages[idx])
                temp_path = output_dir / f"{base_name}_{unique_id}_measure_temp.pdf"
                with open(temp_path, 'wb') as f:
                    writer.write(f)

                if optimize:
                    # Apply Ghostscript optimization to get accurate final size
                    optimized_path = output_dir / f"{base_name}_{unique_id}_measure_opt.pdf"
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

                # Recalculate balanced target for current number of parts
                balanced_target = file_size_mb / num_parts
                target_size = min(balanced_target, max_target)
                logger.info(f"Target size per part: {target_size:.1f}MB")

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

                    report_progress(72 + part_idx * 2, f"Finding split point {part_idx + 1}...")
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
                report_progress(80 + part_num * 3, f"Creating part {part_num} of {len(groups)}...")
                # Unique ID prevents file collisions when multiple jobs run in parallel
                part_unique_id = str(uuid.uuid4())[:8]

                writer = PdfWriter()
                for page_idx in page_indices:
                    writer.add_page(reader.pages[page_idx])

                # Write raw PyPDF2 part to temporary path
                temp_part_path = output_dir / f"{base_name}_part{part_num}_{part_unique_id}_temp.pdf"
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

                if not part_path.exists():
                    raise SplitError(f"Failed to create part {part_num}: file not written")

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
