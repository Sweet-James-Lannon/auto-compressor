"""
Optional lossy PDF compression using Ghostscript.
Used only when explicitly requested. Never affects default lossless pipeline.
Safe, isolated module with comprehensive fallback handling.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def is_ghostscript_available() -> bool:
    """Check if Ghostscript is installed (supports Linux/Mac/Windows)."""
    # Try common Ghostscript binary names across platforms
    for gs_name in ["gs", "gswin64c", "gswin32c"]:
        gs_path = shutil.which(gs_name)
        if gs_path:
            logger.info(f"Ghostscript found at: {gs_path}")
            return True
    logger.warning("Ghostscript not found in PATH (tried: gs, gswin64c, gswin32c)")
    return False


def get_ghostscript_command() -> Optional[str]:
    """Get the Ghostscript command for the current platform."""
    for gs_name in ["gs", "gswin64c", "gswin32c"]:
        if shutil.which(gs_name):
            return gs_name
    return None


def get_ghostscript_version() -> Optional[str]:
    """Get Ghostscript version if available."""
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return None
    try:
        result = subprocess.run(
            [gs_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def compress_with_ghostscript(
    input_path: Path,
    output_path: Path,
    quality: int = 90,
    image_dpi: int = 150,
    preset: str = "ebook"
) -> Tuple[bool, str]:
    """
    Compress PDF using Ghostscript lossy mode.

    Args:
        input_path: Input PDF path
        output_path: Output PDF path
        quality: JPEG quality 1-100 (85-95 recommended for legal docs)
        image_dpi: Target DPI for images (150 for OCR'd documents)
        preset: PDF preset ("screen", "ebook", "printer", "prepress")

    Returns:
        Tuple of (success: bool, message: str)

    Note:
        - Only use when explicitly requested
        - Text quality should remain high
        - Suitable for scanned legal documents with JPEG images
    """
    if not is_ghostscript_available():
        return False, "Ghostscript not available"

    if not input_path.exists():
        return False, f"Input file not found: {input_path}"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Map preset names to Ghostscript settings
        preset_map = {
            "screen": "/screen",      # 72 DPI, lowest quality
            "ebook": "/ebook",        # 150 DPI, good for reading
            "printer": "/printer",     # 300 DPI, high quality
            "prepress": "/prepress",   # 300 DPI, maximum quality
            "legal": "/ebook"         # Custom for legal docs
        }

        pdf_setting = preset_map.get(preset, "/ebook")

        # Get platform-specific Ghostscript command
        gs_cmd = get_ghostscript_command()
        if not gs_cmd:
            return False, "Ghostscript command not found"

        # Build Ghostscript command with optimized settings for JPEG2000 PDFs
        # Key: Force JPEG encoding (DCTEncode) to convert JPEG2000 which GS handles poorly
        cmd = [
            gs_cmd,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={pdf_setting}",
            "-dNOPAUSE",
            "-dBATCH",
            # Image handling - force JPEG encoding (converts JPEG2000 to JPEG)
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/DCTEncode",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/DCTEncode",
            # Downsampling settings - threshold 1.0 forces downsampling
            "-dDownsampleColorImages=true",
            "-dColorImageDownsampleType=/Bicubic",
            f"-dColorImageResolution={image_dpi}",
            "-dColorImageDownsampleThreshold=1.0",
            "-dDownsampleGrayImages=true",
            "-dGrayImageDownsampleType=/Bicubic",
            f"-dGrayImageResolution={image_dpi}",
            "-dGrayImageDownsampleThreshold=1.0",
            "-dDownsampleMonoImages=true",
            "-dMonoImageDownsampleType=/Bicubic",
            f"-dMonoImageResolution={image_dpi}",
            "-dMonoImageDownsampleThreshold=1.0",
            # Quality and optimization
            "-dDetectDuplicateImages=true",
            "-dCompressFonts=true",
            "-dSubsetFonts=true",
            f"-dJPEGQ={quality}",
            f"-sOutputFile={output_path}",
            str(input_path)
        ]

        logger.info(f"Starting Ghostscript compression:")
        logger.info(f"  - Input: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"  - Quality: {quality}%, DPI: {image_dpi}, Preset: {preset}")

        # Dynamic timeout based on file size (10 seconds per MB, minimum 10 minutes)
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        timeout_seconds = max(600, int(file_size_mb * 10))

        # Run Ghostscript with timeout protection
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            logger.error(f"Ghostscript failed: {error_msg}")
            return False, f"Ghostscript error: {error_msg}"

        # Verify output was created
        if not output_path.exists():
            return False, "Ghostscript did not create output file"

        output_size = output_path.stat().st_size / 1024 / 1024
        input_size = input_path.stat().st_size / 1024 / 1024
        reduction = ((input_size - output_size) / input_size) * 100

        logger.info(f"Ghostscript compression successful:")
        logger.info(f"  - Output: {output_path.name} ({output_size:.2f} MB)")
        logger.info(f"  - Reduction: {reduction:.1f}%")

        return True, f"Compressed successfully: {reduction:.1f}% reduction"

    except subprocess.TimeoutExpired:
        logger.error("Ghostscript timed out (>10 minutes)")
        return False, "Ghostscript timeout - file too large or complex"

    except Exception as e:
        logger.error(f"Unexpected Ghostscript error: {e}")
        return False, f"Unexpected error: {str(e)}"


def compress_legal_document(
    input_path: Path,
    output_path: Path,
    compression_level: str = "recommended"
) -> Tuple[bool, str]:
    """
    Specialized compression for legal documents with medical exhibits.
    Uses optimized settings for scanned documents with OCR text.
    Handles JPEG2000 encoded PDFs by converting to JPEG.

    Args:
        input_path: Input PDF path
        output_path: Output PDF path
        compression_level: "low", "recommended", or "extreme"

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Settings based on compression level (iLovePDF-style)
    level_settings = {
        "low": {
            "quality": 95,    # High quality, minimal compression
            "image_dpi": 200,
            "preset": "ebook"
        },
        "recommended": {
            "quality": 85,    # Good balance of quality and size
            "image_dpi": 150,
            "preset": "ebook"
        },
        "extreme": {
            "quality": 72,    # Aggressive compression for maximum size reduction
            "image_dpi": 72,  # 72 DPI - tested to give best results for large PDFs
            "preset": "ebook"
        }
    }

    settings = level_settings.get(compression_level, level_settings["recommended"])

    return compress_with_ghostscript(
        input_path=input_path,
        output_path=output_path,
        quality=settings["quality"],
        image_dpi=settings["image_dpi"],
        preset=settings["preset"]
    )


def test_ghostscript_compression():
    """
    Test function to verify Ghostscript is working.
    Run this to test the module in isolation.
    """
    if not is_ghostscript_available():
        print("Error: Ghostscript not available")
        return False

    version = get_ghostscript_version()
    print(f"Ghostscript version: {version}")

    # Create a test workflow
    test_input = Path("./test_samples/sample.pdf")
    test_output = Path("./test_samples/sample_compressed.pdf")

    if not test_input.exists():
        print(f"Place a test PDF at: {test_input}")
        return False

    print(f"Testing compression on: {test_input}")
    success, message = compress_legal_document(test_input, test_output)

    if success:
        print(f"Success: {message}")
        print(f"Output saved to: {test_output}")
        return True
    else:
        print(f"Failed: {message}")
        return False


if __name__ == "__main__":
    # Run test when module is executed directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_ghostscript_compression()