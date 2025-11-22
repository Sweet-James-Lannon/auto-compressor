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
    """Check if Ghostscript (gs) is installed."""
    gs_path = shutil.which("gs")
    if gs_path:
        logger.info(f"Ghostscript found at: {gs_path}")
        return True
    logger.warning("Ghostscript not found in PATH")
    return False


def get_ghostscript_version() -> Optional[str]:
    """Get Ghostscript version if available."""
    try:
        result = subprocess.run(
            ["gs", "--version"],
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

        # Build Ghostscript command
        cmd = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={pdf_setting}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            "-dDetectDuplicateImages",  # Remove duplicate images
            "-dCompressFonts=true",      # Compress fonts
            "-dSubsetFonts=true",        # Subset fonts
            f"-dJPEGQ={quality}",        # JPEG quality setting
            f"-r{image_dpi}",            # Resolution for downsampling
            "-dDownsampleColorImages=true",
            f"-dColorImageDownsampleThreshold={1.0 if image_dpi < 300 else 1.5}",
            "-dDownsampleGrayImages=true",
            f"-dGrayImageDownsampleThreshold={1.0 if image_dpi < 300 else 1.5}",
            "-dDownsampleMonoImages=true",
            "-dMonoImageDownsampleThreshold=2.0",
            f"-sOutputFile={output_path}",
            str(input_path)
        ]

        logger.info(f"Starting Ghostscript compression:")
        logger.info(f"  - Input: {input_path.name} ({input_path.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"  - Quality: {quality}%, DPI: {image_dpi}, Preset: {preset}")

        # Run Ghostscript with timeout protection
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for large files
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
    output_path: Path
) -> Tuple[bool, str]:
    """
    Specialized compression for legal documents with medical exhibits.
    Uses optimized settings for scanned documents with OCR text.

    Args:
        input_path: Input PDF path
        output_path: Output PDF path

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Optimized settings for legal documents:
    # - Quality 85: Slight JPEG compression, barely noticeable
    # - DPI 150: Good for screen reading and OCR
    # - Preset "legal": Balanced for text clarity
    return compress_with_ghostscript(
        input_path=input_path,
        output_path=output_path,
        quality=85,      # Slightly lower for better compression
        image_dpi=150,   # Standard for document viewing
        preset="legal"   # Optimized for legal documents
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