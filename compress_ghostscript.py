"""Ghostscript PDF compression for scanned legal documents."""

import logging
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from exceptions import SplitError

logger = logging.getLogger(__name__)


def optimize_split_part(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    """
    Lightweight optimization for split PDF parts - removes duplicate resources
    without re-encoding images.

    Use this for split parts where images are already compressed. The /default
    setting preserves image quality while removing PyPDF2's duplicated resources.

    Args:
        input_path: Source PDF (split part with duplicated resources)
        output_path: Destination for optimized PDF

    Returns:
        (success, message) tuple
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use /default - preserves quality, just de-duplicates resources
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/default",  # Preserve quality, don't re-encode
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        # Critical: Don't re-encode images, just pass them through
        "-dPassThroughJPEGImages=true",
        "-dPassThroughJPXImages=true",
        # Remove duplicates
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        f"-sOutputFile={output_path}",
        str(input_path)
    ]

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = max(300, int(file_mb * 5))

        logger.info(f"Optimizing split part {input_path.name} ({file_mb:.1f}MB)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100

        logger.info(f"Optimized: {file_mb:.1f}MB -> {out_mb:.1f}MB ({reduction:.1f}% reduction)")

        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as e:
        return False, str(e)


def get_ghostscript_command() -> Optional[str]:
    """Get Ghostscript binary name for current platform."""
    for name in ["gs", "gswin64c", "gswin32c"]:
        if shutil.which(name):
            return name
    return None


def translate_ghostscript_error(stderr: str, return_code: int) -> str:
    """Translate Ghostscript stderr to a clear, user-friendly error message.

    Also logs the full stderr for debugging purposes.

    Args:
        stderr: Full stderr output from Ghostscript.
        return_code: Process exit code.

    Returns:
        User-friendly error message explaining what went wrong.
    """
    # Log full stderr for debugging (don't truncate!)
    logger.error(f"Ghostscript failed (exit code {return_code}). Full error:\n{stderr}")

    stderr_lower = stderr.lower()

    # Check for encryption/permission errors
    if 'invalidfileaccess' in stderr_lower or 'password' in stderr_lower:
        return "PDF is password-protected or locked. Please remove the password and try again."

    # Check for metadata/type errors
    if 'typecheck' in stderr_lower or 'rangecheck' in stderr_lower:
        return "PDF has corrupted internal data. Try re-saving it from Adobe Acrobat."

    # Check for structure errors
    if any(x in stderr_lower for x in ['undefined', 'ioerror', 'syntaxerror', 'eofread']):
        return "PDF is damaged or corrupted. Please use a different copy of the file."

    # Generic fallback with exit code
    return f"PDF processing failed (Ghostscript exit code {return_code}). The file may be corrupted."


def compress_pdf_with_ghostscript(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    """
    Compress scanned PDF using Ghostscript at 72 DPI with JPEG encoding.

    Uses aggressive compression (72 DPI) which provides significant file size
    reduction while maintaining readable quality for legal documents.
    Converts JPEG2000 to JPEG for maximum compatibility.

    Args:
        input_path: Source PDF
        output_path: Destination for compressed PDF

    Returns:
        (success, message) tuple where message contains reduction % or error
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 72 DPI + DCTEncode: Best compression for scanned legal documents
    # Higher DPI (150, 200) often makes JPEG2000 files BIGGER
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/screen",  # 72 DPI preset - best compression
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        # Force JPEG encoding (converts JPEG2000 to JPEG)
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        # Downsample all images to 72 DPI for maximum compression
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dColorImageResolution=72",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dGrayImageResolution=72",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",  # Faster for 1-bit images
        "-dMonoImageResolution=150",  # Raised for readable scanned text
        "-dMonoImageDownsampleThreshold=1.0",
        # Optimization
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        # Strip metadata that might confuse email clients
        "-dFastWebView=true",
        "-dPrinted=false",
        # Speed optimizations (NO effect on compression ratio)
        "-dNumRenderingThreads=4",
        "-dBandHeight=100",
        "-dBandBufferSpace=500000000",
        f"-sOutputFile={output_path}",
        str(input_path)
    ]

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = max(600, int(file_mb * 10))  # 10 sec/MB, min 10 minutes

        logger.info(f"Compressing {input_path.name} ({file_mb:.1f}MB)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, translate_ghostscript_error(result.stderr, result.returncode)

        if not output_path.exists():
            return False, "Output file not created"

        out_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = ((file_mb - out_mb) / file_mb) * 100

        logger.info(f"Result: {file_mb:.1f}MB -> {out_mb:.1f}MB ({reduction:.1f}% reduction)")

        return True, f"{reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        return False, "Timeout exceeded"
    except Exception as e:
        return False, str(e)


# =============================================================================
# PARALLEL COMPRESSION - Uses multiple CPU cores
# =============================================================================

def compress_parallel(
    input_path: Path,
    working_dir: Path,
    base_name: str,
    split_threshold_mb: float = 25.0,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    max_workers: int = 6
) -> Dict:
    """
    Parallel compression strategy for large PDFs.

    Flow:
    1. Split input PDF into N chunks by page count (fast, no Ghostscript)
    2. Compress each chunk in parallel using ThreadPoolExecutor
    3. Merge compressed chunks back into one PDF
    4. Final split by 25MB threshold for email

    This is 3-4x faster than serial compression for large files because
    each Ghostscript process uses its own CPU core.

    Args:
        input_path: Path to input PDF.
        working_dir: Directory for temp and output files.
        base_name: Base filename for output.
        split_threshold_mb: Max size per final part (default 25MB).
        progress_callback: Optional callback(percent, stage, message).
        max_workers: Max parallel compression workers (default 6).

    Returns:
        Dict with output_path(s), sizes, reduction_percent, etc.
    """
    # Import here to avoid circular imports
    from split_pdf import split_by_pages, merge_pdfs, split_by_size
    from utils import get_file_size_mb

    input_path = Path(input_path)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    def report_progress(percent: int, stage: str, message: str):
        if progress_callback:
            progress_callback(percent, stage, message)

    original_size_mb = get_file_size_mb(input_path)
    original_bytes = input_path.stat().st_size

    logger.info(f"[PARALLEL] Starting parallel compression for {input_path.name} ({original_size_mb:.1f}MB)")

    # Step 1: Calculate optimal number of chunks (~20MB per chunk)
    num_chunks = max(2, min(max_workers, int(original_size_mb / 20)))
    report_progress(5, "splitting", f"Splitting into {num_chunks} chunks for parallel compression...")

    # Step 2: Split by page count (fast, no Ghostscript)
    chunk_paths = split_by_pages(input_path, working_dir, num_chunks, base_name)
    logger.info(f"[PARALLEL] Split into {len(chunk_paths)} chunks")

    # Step 3: Compress each chunk in parallel
    report_progress(15, "compressing", f"Compressing {num_chunks} chunks in parallel...")

    def compress_single_chunk(chunk_path: Path) -> Tuple[Path, bool, str]:
        """Compress a single chunk and return result."""
        unique_id = str(uuid.uuid4())[:8]
        compressed_path = working_dir / f"{chunk_path.stem}_{unique_id}_compressed.pdf"
        success, message = compress_pdf_with_ghostscript(chunk_path, compressed_path)
        return compressed_path, success, message

    compressed_chunks = []
    failed_chunks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all compression jobs
        futures = {
            executor.submit(compress_single_chunk, chunk): (i, chunk)
            for i, chunk in enumerate(chunk_paths)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            chunk_idx, original_chunk = futures[future]
            try:
                compressed_path, success, message = future.result()
                if success and compressed_path.exists():
                    compressed_chunks.append((chunk_idx, compressed_path))
                    logger.info(f"[PARALLEL] Chunk {chunk_idx + 1} compressed successfully")
                else:
                    # Compression failed - use original chunk
                    logger.warning(f"[PARALLEL] Chunk {chunk_idx + 1} compression failed: {message}")
                    failed_chunks.append((chunk_idx, original_chunk))
            except Exception as e:
                logger.error(f"[PARALLEL] Chunk {chunk_idx + 1} error: {e}")
                failed_chunks.append((chunk_idx, original_chunk))

            # Update progress
            done = len(compressed_chunks) + len(failed_chunks)
            pct = 15 + int(55 * done / num_chunks)
            report_progress(pct, "compressing", f"Compressed {done}/{num_chunks} chunks...")

    # Sort chunks by original order and combine with failed chunks
    all_chunks = compressed_chunks + failed_chunks
    all_chunks.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in all_chunks]

    logger.info(f"[PARALLEL] Compression complete: {len(compressed_chunks)} succeeded, {len(failed_chunks)} used original")

    # Step 4: Merge compressed chunks back together
    report_progress(75, "merging", "Merging compressed chunks...")
    merged_path = working_dir / f"{base_name}_compressed.pdf"
    merge_pdfs(ordered_paths, merged_path)

    compressed_size_mb = get_file_size_mb(merged_path)
    compressed_bytes = merged_path.stat().st_size
    reduction_percent = ((original_size_mb - compressed_size_mb) / original_size_mb) * 100

    logger.info(f"[PARALLEL] Merged: {original_size_mb:.1f}MB -> {compressed_size_mb:.1f}MB ({reduction_percent:.1f}% reduction)")

    # Cleanup chunk files
    for chunk in chunk_paths:
        chunk.unlink(missing_ok=True)
    for _, compressed in compressed_chunks:
        compressed.unlink(missing_ok=True)

    # =========================================================================
    # CRITICAL: If parallel compression made file BIGGER, fall back to serial
    # This can happen because PyPDF2 merge duplicates resources across chunks
    # =========================================================================
    if compressed_size_mb >= original_size_mb:
        bloat_bytes = int((compressed_size_mb - original_size_mb) * 1024 * 1024)
        bloat_pct = ((compressed_size_mb - original_size_mb) / original_size_mb) * 100

        logger.warning(f"[PARALLEL] " + "="*50)
        logger.warning(f"[PARALLEL] ⚠️  PARALLEL COMPRESSION CAUSED BLOAT!")
        logger.warning(f"[PARALLEL] " + "="*50)
        logger.warning(f"[PARALLEL]   Original size:     {original_size_mb:.2f}MB")
        logger.warning(f"[PARALLEL]   After parallel:    {compressed_size_mb:.2f}MB")
        logger.warning(f"[PARALLEL]   Bloat:             +{bloat_pct:.1f}% (+{bloat_bytes:,} bytes)")
        logger.warning(f"[PARALLEL]   Chunks processed:  {num_chunks}")
        logger.warning(f"[PARALLEL]   Root cause: PyPDF2 merge duplicated resources across chunks")
        logger.warning(f"[PARALLEL]   Action: Falling back to serial compression (single Ghostscript pass)...")
        logger.warning(f"[PARALLEL] " + "="*50)

        # Delete the inflated merged file
        merged_path.unlink(missing_ok=True)

        # Fall back to serial compression (single Ghostscript pass on entire file)
        report_progress(40, "compressing", "Parallel failed, trying serial compression...")

        serial_output = working_dir / f"{base_name}_serial_compressed.pdf"
        success, message = compress_pdf_with_ghostscript(input_path, serial_output)

        if success and serial_output.exists():
            serial_size_mb = get_file_size_mb(serial_output)

            # If serial also made it bigger, just split the original
            if serial_size_mb >= original_size_mb:
                logger.warning(f"[PARALLEL] Serial also increased size. Using original file.")
                serial_output.unlink(missing_ok=True)

                # Just split the original file without compression
                from split_pdf import split_by_size
                final_parts = split_by_size(input_path, working_dir, base_name, split_threshold_mb)

                return {
                    "output_path": str(final_parts[0]),
                    "output_paths": [str(p) for p in final_parts],
                    "original_size_mb": round(original_size_mb, 2),
                    "compressed_size_mb": round(original_size_mb, 2),
                    "reduction_percent": 0.0,
                    "compression_method": "none",
                    "was_split": len(final_parts) > 1,
                    "total_parts": len(final_parts),
                    "success": True,
                    "note": "File already optimized, split only",
                    "page_count": None,
                    "part_sizes": [p.stat().st_size for p in final_parts],
                    "parallel_chunks": num_chunks
                }

            # Serial worked - use that result
            serial_reduction_pct = ((original_size_mb - serial_size_mb) / original_size_mb) * 100
            serial_reduction_mb = original_size_mb - serial_size_mb

            logger.info(f"[PARALLEL] " + "-"*50)
            logger.info(f"[PARALLEL] ✓ SERIAL FALLBACK SUCCEEDED")
            logger.info(f"[PARALLEL]   Original:    {original_size_mb:.2f}MB")
            logger.info(f"[PARALLEL]   Compressed:  {serial_size_mb:.2f}MB")
            logger.info(f"[PARALLEL]   Reduction:   {serial_reduction_pct:.1f}% (-{serial_reduction_mb:.2f}MB)")
            logger.info(f"[PARALLEL] " + "-"*50)

            # Rename serial output to expected name
            final_compressed = working_dir / f"{base_name}_compressed.pdf"
            serial_output.rename(final_compressed)
            merged_path = final_compressed

            compressed_size_mb = serial_size_mb
            compressed_bytes = merged_path.stat().st_size
            reduction_percent = ((original_size_mb - compressed_size_mb) / original_size_mb) * 100
        else:
            # Serial failed too - just split original
            logger.error(f"[PARALLEL] Serial compression also failed: {message}")

            from split_pdf import split_by_size
            final_parts = split_by_size(input_path, working_dir, base_name, split_threshold_mb)

            return {
                "output_path": str(final_parts[0]),
                "output_paths": [str(p) for p in final_parts],
                "original_size_mb": round(original_size_mb, 2),
                "compressed_size_mb": round(original_size_mb, 2),
                "reduction_percent": 0.0,
                "compression_method": "none",
                "was_split": len(final_parts) > 1,
                "total_parts": len(final_parts),
                "success": True,
                "note": "Compression failed, split original",
                "page_count": None,
                "part_sizes": [p.stat().st_size for p in final_parts],
                "parallel_chunks": num_chunks
            }

    # Step 5: Final split by 25MB threshold if needed
    if compressed_size_mb > split_threshold_mb:
        report_progress(85, "splitting", f"Splitting into parts under {split_threshold_mb}MB...")
        final_parts = split_by_size(merged_path, working_dir, base_name, split_threshold_mb)

        # Only delete merged file if split actually created NEW files
        # (split_by_size returns [merged_path] if already under threshold)
        # Use resolve() for robust path comparison
        merged_resolved = merged_path.resolve()
        parts_are_new_files = (
            len(final_parts) > 1 or
            (len(final_parts) == 1 and final_parts[0].resolve() != merged_resolved)
        )

        if parts_are_new_files:
            merged_path.unlink(missing_ok=True)
            logger.info(f"[PARALLEL] Deleted merged file, keeping {len(final_parts)} split parts")
        else:
            logger.info(f"[PARALLEL] Keeping merged file as final output")

        # Verify all output files exist before returning
        missing_parts: List[str] = []
        for i, part in enumerate(final_parts):
            if not part.exists():
                logger.error(f"[PARALLEL] Part {i+1} missing after split: {part}")
                missing_parts.append(f"missing part {i+1}")
            elif part.stat().st_size == 0:
                logger.error(f"[PARALLEL] Part {i+1} is empty after split: {part}")
                missing_parts.append(f"empty part {i+1}")
            else:
                logger.info(f"[PARALLEL] Part {i+1} verified: {part.name} ({part.stat().st_size / (1024*1024):.1f}MB)")

        if missing_parts:
            raise SplitError(f"Parallel split output incomplete: {', '.join(missing_parts)}")

        # Get page count from original
        from PyPDF2 import PdfReader
        try:
            with open(input_path, 'rb') as f:
                page_count = len(PdfReader(f).pages)
        except Exception:
            page_count = None

        report_progress(100, "complete", f"Complete: {len(final_parts)} parts")

        return {
            "output_path": str(final_parts[0]),
            "output_paths": [str(p) for p in final_parts],
            "original_size_mb": round(original_size_mb, 2),
            "compressed_size_mb": round(compressed_size_mb, 2),
            "reduction_percent": round(reduction_percent, 1),
            "compression_method": "ghostscript_parallel",
            "was_split": len(final_parts) > 1,  # Only true if actually split into multiple parts
            "total_parts": len(final_parts),
            "success": True,
            "page_count": page_count,
            "part_sizes": [p.stat().st_size for p in final_parts],
            "parallel_chunks": num_chunks
        }

    # No split needed - return single file
    report_progress(100, "complete", "Compression complete")

    # Get page count
    from PyPDF2 import PdfReader
    try:
        with open(input_path, 'rb') as f:
            page_count = len(PdfReader(f).pages)
    except Exception:
        page_count = None

    return {
        "output_path": str(merged_path),
        "output_paths": [str(merged_path)],
        "original_size_mb": round(original_size_mb, 2),
        "compressed_size_mb": round(compressed_size_mb, 2),
        "reduction_percent": round(reduction_percent, 1),
        "compression_method": "ghostscript_parallel",
        "was_split": False,
        "total_parts": 1,
        "success": True,
        "page_count": page_count,
        "part_sizes": [merged_path.stat().st_size],
        "parallel_chunks": num_chunks
    }
