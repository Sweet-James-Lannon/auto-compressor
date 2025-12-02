"""Ghostscript PDF compression for scanned legal documents."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional

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
            return False, f"Ghostscript error: {result.stderr[:200]}"

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


def compress_pdf_with_ghostscript(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    """
    Compress scanned PDF using Ghostscript at 150 DPI with JPEG encoding.

    Balances file size reduction with readable quality for legal documents.
    Uses /ebook preset (150 DPI, JPEG quality ~80) with optimized settings.

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

    # /ebook preset: 150 DPI with JPEG compression, good balance for legal docs
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",  # 150 DPI preset with JPEG quality ~80
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        # Force JPEG encoding (converts JPEG2000 to JPEG)
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        # Downsample images to 150 DPI for readable text
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dColorImageResolution=150",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dGrayImageResolution=150",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Bicubic",
        "-dMonoImageResolution=300",  # Higher DPI for text/line art
        "-dMonoImageDownsampleThreshold=1.0",
        # Optimization
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        # Strip metadata that might confuse email clients
        "-dFastWebView=true",
        "-dPrinted=false",
        f"-sOutputFile={output_path}",
        str(input_path)
    ]

    try:
        file_mb = input_path.stat().st_size / (1024 * 1024)
        timeout = max(600, int(file_mb * 10))  # 10 sec/MB, min 10 minutes

        logger.info(f"Compressing {input_path.name} ({file_mb:.1f}MB)")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return False, f"Ghostscript error: {result.stderr[:200]}"

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
