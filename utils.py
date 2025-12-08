"""Shared utility functions for PDF compression service.

Contains:
- get_file_size_mb: Get file size in MB
- download_pdf: Download PDF from URL with security validation
"""

import ipaddress
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

# Constants for PDF download
DOWNLOAD_TIMEOUT: int = 300  # 5 minute timeout
MAX_DOWNLOAD_SIZE: int = 314572800  # 300 MB max

logger = logging.getLogger(__name__)


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.
    """
    return path.stat().st_size / (1024 * 1024)


def _is_safe_url(url: str) -> bool:
    """Check if URL is safe to fetch (not internal/metadata endpoints).

    Prevents SSRF attacks by blocking:
    - Cloud metadata endpoints (AWS, Azure, GCP)
    - Localhost and loopback addresses
    - Private IP ranges

    Args:
        url: The URL to validate.

    Returns:
        True if URL appears safe, False otherwise.
    """
    try:
        parsed = urlparse(url)

        # Only allow http/https schemes
        if parsed.scheme not in ('http', 'https'):
            return False

        # Block common metadata and internal endpoints
        blocked_hosts = [
            '169.254.169.254',  # Azure/AWS metadata
            'metadata.google.internal',
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
        ]

        hostname = parsed.hostname or ''
        if hostname.lower() in blocked_hosts:
            return False

        # Block private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass  # Not an IP address, hostname is OK

        return True
    except Exception:
        return False


def download_pdf(url: str, output_path: Path) -> None:
    """Download a PDF from a URL.

    Used by /compress-sync to fetch PDFs from Salesforce/Docrio URLs.

    Args:
        url: The URL to download from.
        output_path: Path to save the downloaded file.

    Raises:
        RuntimeError: If download fails, URL is blocked, or file is too large.
    """
    # Security: Validate URL to prevent SSRF
    if not _is_safe_url(url):
        raise RuntimeError("Invalid or blocked URL")

    logger.info(f"Downloading PDF from {url[:100]}...")

    try:
        response = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
            headers={'User-Agent': 'SJ-PDF-Compressor/1.0'},
            allow_redirects=False  # Don't follow redirects to internal URLs
        )
        response.raise_for_status()

        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            raise RuntimeError(f"File too large: {int(content_length) / (1024*1024):.1f}MB")

        # Stream download with size check
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_SIZE:
                    raise RuntimeError("File too large (exceeded 300MB during download)")
                f.write(chunk)

        logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB to {output_path.name}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed: {e}") from e
