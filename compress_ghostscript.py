"""Ghostscript PDF compression for scanned legal documents."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def get_ghostscript_command() -> Optional[str]:
    """Get Ghostscript binary name for current platform."""
    for name in ["gs", "gswin64c", "gswin32c"]:
        if shutil.which(name):
            return name
    return None


def compress_pdf(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    """
    Compress scanned PDF using Ghostscript at 72 DPI.

    This is the only setting that actually reduces file size for JPEG2000
    scanned documents. Higher DPI settings make files BIGGER.

    Args:
        input_path: Source PDF
        output_path: Destination for compressed PDF

    Returns:
        (success, message) tuple
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return False, "Ghostscript not installed"

    if not input_path.exists():
        return False, f"File not found: {input_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 72 DPI + DCTEncode: tested to give 29% reduction on 115MB scanned PDF
    # Higher DPI (150, 200) makes JPEG2000 files BIGGER
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/screen",  # 72 DPI preset
        "-dNOPAUSE",
        "-dBATCH",
        # Force JPEG encoding (converts JPEG2000 to JPEG)
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        # Downsample all images to 72 DPI
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        "-dColorImageResolution=72",
        "-dColorImageDownsampleThreshold=1.0",
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        "-dGrayImageResolution=72",
        "-dGrayImageDownsampleThreshold=1.0",
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Bicubic",
        "-dMonoImageResolution=72",
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
