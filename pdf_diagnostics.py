"""PDF diagnostics module for analyzing PDF files.

Provides functions to analyze PDF structure, detect compression state,
identify scanned documents, and diagnose potential issues before processing.
"""

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

logger = logging.getLogger(__name__)

CHUNK_TIME_TARGET_SEC = int(os.environ.get("CHUNK_TIME_TARGET_SEC", "90"))
PROFILE_ASSUMED_WORKERS = int(os.environ.get("PROFILE_ASSUMED_WORKERS", "4"))
PROFILE_MIN_PAGES_PER_CHUNK = 50
PROFILE_MAX_PAGES_PER_CHUNK = 400
PROFILE_BASE_SLA_SEC = int(os.environ.get("PARALLEL_JOB_SLA_SEC", "300"))
PROFILE_MAX_SLA_SEC = int(os.environ.get("PARALLEL_JOB_SLA_MAX_SEC", "1800"))
CODEC_PROFILES = {
    "fast": {"base_spp": 0.10, "codecs": {"DCTDecode", "FlateDecode", "None"}},
    "medium": {"base_spp": 0.25, "codecs": {"CCITTFaxDecode"}},
    "slow": {"base_spp": 0.50, "codecs": {"JBIG2Decode", "JPXDecode"}},
}


def analyze_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Analyze a PDF file and return detailed information.

    Args:
        pdf_path: Path to the PDF file to analyze.

    Returns:
        Dictionary containing:
        - file_size_bytes: int
        - file_size_mb: float
        - page_count: int
        - metadata: dict of PDF metadata
        - is_encrypted: bool
        - bytes_per_page: float (average)
        - producer: str (software that created/modified the PDF)
        - creator: str (original creator software)
    """
    result: Dict[str, Any] = {
        "file_size_bytes": 0,
        "file_size_mb": 0.0,
        "page_count": 0,
        "metadata": {},
        "is_encrypted": False,
        "bytes_per_page": 0.0,
        "producer": "",
        "creator": "",
        "error": None,
    }

    try:
        # Get file size
        file_size = os.path.getsize(pdf_path)
        result["file_size_bytes"] = file_size
        result["file_size_mb"] = round(file_size / (1024 * 1024), 2)

        # Read PDF structure
        reader = PdfReader(str(pdf_path), strict=False)
        result["is_encrypted"] = reader.is_encrypted

        if not reader.is_encrypted:
            result["page_count"] = len(reader.pages)

            # Calculate bytes per page
            if result["page_count"] > 0:
                result["bytes_per_page"] = file_size / result["page_count"]

            # Extract metadata
            if reader.metadata:
                metadata = {}
                for key, value in reader.metadata.items():
                    # Clean up key name (remove leading /)
                    clean_key = str(key).lstrip("/")
                    metadata[clean_key] = str(value) if value else ""
                result["metadata"] = metadata

                # Extract common fields
                result["producer"] = metadata.get("Producer", "")
                result["creator"] = metadata.get("Creator", "")

    except PdfReadError as e:
        result["error"] = f"PDF read error: {str(e)}"
        logger.warning(f"[DIAGNOSTICS] PDF read error for {pdf_path}: {e}")
    except Exception as e:
        result["error"] = f"Analysis error: {str(e)}"
        logger.warning(f"[DIAGNOSTICS] Analysis error for {pdf_path}: {e}")

    return result


def detect_already_compressed(pdf_path: Path) -> Tuple[bool, str]:
    """Detect if a PDF has already been heavily compressed.

    Checks for signs of prior compression:
    - Known compression tool signatures in Producer/Creator
    - Very low bytes-per-page ratio

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (is_compressed: bool, reason: str)
    """
    analysis = analyze_pdf(pdf_path)

    if analysis.get("error"):
        return False, ""

    # Known compression tools
    compression_tools = [
        "ghostscript",
        "ilovepdf",
        "smallpdf",

        "pdf compressor",
        "nitro",
        "foxit",
        "pdfoptim",
        "qpdf",
        "cpdf",
        "pdfsizeopt",
    ]

    producer = analysis.get("producer", "").lower()
    creator = analysis.get("creator", "").lower()

    # Check for compression tool signatures
    for tool in compression_tools:
        if tool in producer:
            return True, f"Previously processed by {tool} (Producer)"
        if tool in creator:
            return True, f"Previously processed by {tool} (Creator)"

    # Check bytes per page - very low values indicate heavy compression
    bytes_per_page = analysis.get("bytes_per_page", 0)
    if bytes_per_page > 0:
        # Less than 20KB per page is quite compressed
        if bytes_per_page < 20000:
            return True, f"Low bytes/page ratio ({bytes_per_page/1024:.1f}KB/page)"

    return False, ""


def estimate_processing_profile(
    fingerprint: Dict[str, Any],
    file_size_mb: float,
) -> Dict[str, Any]:
    """Estimate processing cost and safe chunk/SLA settings for compression."""
    image_filters = fingerprint.get("image_filters") or {}
    if not isinstance(image_filters, dict):
        image_filters = {}

    codec_counts: Dict[str, int] = {}
    for codec, count in image_filters.items():
        try:
            codec_counts[str(codec)] = int(count)
        except (TypeError, ValueError):
            codec_counts[str(codec)] = 0

    dominant_codec = "None"
    if codec_counts:
        dominant_codec = max(codec_counts, key=lambda name: codec_counts[name])

    tier = "fast"
    if any(codec_counts.get(codec, 0) > 0 for codec in CODEC_PROFILES["slow"]["codecs"]):
        tier = "slow"
    elif any(codec_counts.get(codec, 0) > 0 for codec in CODEC_PROFILES["medium"]["codecs"]):
        tier = "medium"

    base_spp = float(CODEC_PROFILES[tier]["base_spp"])
    avg_image_dpi = fingerprint.get("avg_image_dpi")
    try:
        dpi_value = float(avg_image_dpi) if avg_image_dpi is not None else 150.0
    except (TypeError, ValueError):
        dpi_value = 150.0
    dpi_factor = max(1.0, dpi_value / 150.0)
    sec_per_page = round(base_spp * dpi_factor, 3)

    try:
        page_count = int(fingerprint.get("page_count") or 0)
    except (TypeError, ValueError):
        page_count = 0

    safe_target = max(30, CHUNK_TIME_TARGET_SEC)
    pages_per_chunk = int(safe_target / max(sec_per_page, 0.01))
    max_pages_per_chunk = max(
        PROFILE_MIN_PAGES_PER_CHUNK,
        min(PROFILE_MAX_PAGES_PER_CHUNK, pages_per_chunk),
    )
    if page_count > 0:
        max_pages_per_chunk = max(1, min(max_pages_per_chunk, page_count))

    chunk_timeout_sec = max(60, int(max_pages_per_chunk * sec_per_page * 2.5))

    if page_count > 0:
        workers = max(1, PROFILE_ASSUMED_WORKERS)
        expected = (page_count * sec_per_page / workers) * 1.5 * 1.2
        recommended_sla_sec = max(PROFILE_BASE_SLA_SEC, int(expected))
    else:
        size_scaled = int(max(1.0, file_size_mb) * 2.5)
        recommended_sla_sec = max(PROFILE_BASE_SLA_SEC, size_scaled)

    recommended_sla_sec = min(recommended_sla_sec, PROFILE_MAX_SLA_SEC)

    return {
        "tier": tier,
        "sec_per_page": sec_per_page,
        "max_pages_per_chunk": int(max_pages_per_chunk),
        "chunk_timeout_sec": int(chunk_timeout_sec),
        "recommended_sla_sec": int(recommended_sla_sec),
        "dominant_codec": dominant_codec,
    }


def fingerprint_pdf(pdf_path: Path, max_pages: int = 5) -> Dict[str, Any]:
    """Generate a lightweight PDF fingerprint for logging and diagnostics."""
    result: Dict[str, Any] = {
        "page_count": 0,
        "sample_pages": 0,
        "file_size_mb": 0.0,
        "bytes_per_page": 0.0,
        "producer": "",
        "creator": "",
        "already_compressed": False,
        "already_reason": "",
        "image_count": 0,
        "image_filters": {},
        "image_megapixels": 0.0,
        "avg_image_dpi": None,
        "text_pages": 0,
        "text_chars": 0,
        "error": None,
    }

    analysis = analyze_pdf(pdf_path)
    if analysis.get("error"):
        result["error"] = analysis["error"]
        return result

    result["file_size_mb"] = analysis.get("file_size_mb", 0.0)
    result["bytes_per_page"] = analysis.get("bytes_per_page", 0.0)
    result["producer"] = analysis.get("producer", "")
    result["creator"] = analysis.get("creator", "")

    # Inline "already compressed" detection from analysis to avoid extra IO.
    compression_tools = [
        "ghostscript",
        "ilovepdf",
        "smallpdf",
        "pdf compressor",
        "nitro",
        "foxit",
        "pdfoptim",
        "qpdf",
        "cpdf",
        "pdfsizeopt",
    ]
    producer = analysis.get("producer", "").lower()
    creator = analysis.get("creator", "").lower()
    for tool in compression_tools:
        if tool in producer:
            result["already_compressed"] = True
            result["already_reason"] = f"Previously processed by {tool} (Producer)"
            break
        if tool in creator:
            result["already_compressed"] = True
            result["already_reason"] = f"Previously processed by {tool} (Creator)"
            break
    if not result["already_compressed"]:
        bytes_per_page = analysis.get("bytes_per_page", 0)
        if bytes_per_page > 0 and bytes_per_page < 20000:
            result["already_compressed"] = True
            result["already_reason"] = f"Low bytes/page ratio ({bytes_per_page/1024:.1f}KB/page)"

    try:
        reader = PdfReader(str(pdf_path), strict=False)
        if reader.is_encrypted:
            result["error"] = "PDF is encrypted"
            return result

        page_count = len(reader.pages)
        result["page_count"] = page_count
        sample_pages = min(max_pages, page_count) if page_count else 0
        result["sample_pages"] = sample_pages

        filters = Counter()
        image_pixels = 0.0
        image_dpis: list[float] = []

        def _resolve(obj: Any) -> Any:
            try:
                return obj.get_object()
            except Exception:
                return obj

        def _clean_filter(name: Any) -> str:
            label = str(name)
            return label[1:] if label.startswith("/") else label

        for idx in range(sample_pages):
            page = reader.pages[idx]
            try:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    result["text_pages"] += 1
                    result["text_chars"] += len(text)
            except Exception:
                pass

            resources = _resolve(page.get("/Resources") or {})
            xobj = resources.get("/XObject")
            if not xobj:
                continue
            xobj = _resolve(xobj)
            for _, obj in xobj.items():
                obj = _resolve(obj)
                if obj.get("/Subtype") != "/Image":
                    continue
                result["image_count"] += 1
                width = obj.get("/Width")
                height = obj.get("/Height")
                if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                    image_pixels += float(width) * float(height)
                    try:
                        mediabox = page.mediabox
                        page_w = float(mediabox.width)
                        page_h = float(mediabox.height)
                        if page_w > 0 and page_h > 0:
                            dpi_x = float(width) / (page_w / 72.0)
                            dpi_y = float(height) / (page_h / 72.0)
                            image_dpis.append((dpi_x + dpi_y) / 2.0)
                    except Exception:
                        pass

                filter_val = obj.get("/Filter")
                if filter_val is None:
                    filters["None"] += 1
                elif isinstance(filter_val, list):
                    for f_item in filter_val:
                        filters[_clean_filter(f_item)] += 1
                else:
                    filters[_clean_filter(filter_val)] += 1

        result["image_filters"] = dict(filters)
        if image_pixels:
            result["image_megapixels"] = round(image_pixels / 1_000_000, 2)
        if image_dpis:
            result["avg_image_dpi"] = round(sum(image_dpis) / len(image_dpis), 1)
    except Exception as e:
        result["error"] = f"Fingerprint error: {str(e)}"
        logger.warning(f"[DIAGNOSTICS] Fingerprint error for {pdf_path}: {e}")

    return result


def detect_scanned_document(pdf_path: Path) -> Tuple[bool, float]:
    """Detect if a PDF is a scanned document (images only, no text).

    Scanned documents typically have:
    - Little to no extractable text
    - High bytes-per-page ratio (large images)
    - Consistent page sizes

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (is_scanned: bool, confidence: float 0-100)
    """
    try:
        reader = PdfReader(str(pdf_path), strict=False)

        if reader.is_encrypted:
            return False, 0.0

        page_count = len(reader.pages)
        if page_count == 0:
            return False, 0.0

        # Sample up to 5 pages for text extraction
        sample_size = min(5, page_count)
        pages_with_no_text = 0
        total_text_length = 0

        for i in range(sample_size):
            try:
                text = reader.pages[i].extract_text() or ""
                text = text.strip()
                total_text_length += len(text)

                # Less than 50 characters is essentially no text
                if len(text) < 50:
                    pages_with_no_text += 1
            except Exception:
                # If text extraction fails, count as no text
                pages_with_no_text += 1

        # Calculate confidence
        no_text_ratio = pages_with_no_text / sample_size
        avg_text_per_page = total_text_length / sample_size

        # High confidence if most pages have no text
        if no_text_ratio >= 0.8 and avg_text_per_page < 100:
            confidence = 90.0
            is_scanned = True
        elif no_text_ratio >= 0.6 and avg_text_per_page < 200:
            confidence = 70.0
            is_scanned = True
        elif no_text_ratio >= 0.4 and avg_text_per_page < 300:
            confidence = 50.0
            is_scanned = True
        else:
            confidence = max(0, no_text_ratio * 50)
            is_scanned = confidence > 40

        return is_scanned, confidence

    except Exception as e:
        logger.warning(f"[DIAGNOSTICS] Scanned detection error for {pdf_path}: {e}")
        return False, 0.0


def get_quality_warnings(pdf_path: Path) -> list:
    """Get quality warnings for a PDF before processing.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of warning dictionaries with type, message, and severity.
    """
    warnings = []

    # Check if already compressed
    is_compressed, reason = detect_already_compressed(pdf_path)
    if is_compressed:
        warnings.append({
            "type": "already_compressed",
            "message": f"This PDF appears already compressed. {reason}. Further compression may be minimal.",
            "severity": "info",
        })

    # Check if scanned document
    is_scanned, confidence = detect_scanned_document(pdf_path)
    if is_scanned:
        warnings.append({
            "type": "scanned_document",
            "message": f"This appears to be a scanned document ({confidence:.0f}% confidence). Compression results may vary.",
            "severity": "info",
        })

    # Check file size
    analysis = analyze_pdf(pdf_path)
    file_size_mb = analysis.get("file_size_mb", 0)

    if file_size_mb > 200:
        warnings.append({
            "type": "large_file",
            "message": f"Large file ({file_size_mb:.1f}MB). Processing may take several minutes.",
            "severity": "warning",
        })

    return warnings


def diagnose_for_job(
    input_path: Optional[Path],
    output_paths: list,
    reported_compressed_mb: float,
) -> Dict[str, Any]:
    """Generate diagnostic report for a compression job.

    Args:
        input_path: Path to input PDF (may be None if cleaned up).
        output_paths: List of output file paths.
        reported_compressed_mb: The compressed size reported by the API.

    Returns:
        Diagnostic report dictionary.
    """
    report = {
        "input_analysis": None,
        "output_analyses": [],
        "size_verification": {
            "reported_mb": reported_compressed_mb,
            "actual_total_mb": 0.0,
            "discrepancy_mb": 0.0,
            "discrepancy_percent": 0.0,
            "all_parts_verified": True,
        },
        "quality_warnings": [],
    }

    # Analyze input if available
    if input_path and input_path.exists():
        report["input_analysis"] = analyze_pdf(input_path)
        report["quality_warnings"] = get_quality_warnings(input_path)

    # Analyze each output file
    total_actual_bytes = 0
    for path_str in output_paths:
        path = Path(path_str)
        if path.exists():
            analysis = analyze_pdf(path)
            analysis["filename"] = path.name
            report["output_analyses"].append(analysis)
            total_actual_bytes += analysis.get("file_size_bytes", 0)
        else:
            report["output_analyses"].append({
                "filename": path.name if hasattr(path, 'name') else str(path),
                "error": "File not found",
            })
            report["size_verification"]["all_parts_verified"] = False

    # Calculate size verification
    actual_mb = total_actual_bytes / (1024 * 1024)
    report["size_verification"]["actual_total_mb"] = round(actual_mb, 2)
    report["size_verification"]["discrepancy_mb"] = round(actual_mb - reported_compressed_mb, 2)

    if reported_compressed_mb > 0:
        discrepancy_pct = ((actual_mb - reported_compressed_mb) / reported_compressed_mb) * 100
        report["size_verification"]["discrepancy_percent"] = round(discrepancy_pct, 1)

    return report
