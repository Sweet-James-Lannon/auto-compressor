"""Shared utility functions for PDF compression service.

Contains:
- get_file_size_mb: Get file size in MB
- download_pdf: Download PDF from URL with security validation
"""

import ipaddress
import logging
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

from exceptions import DownloadError

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
        logger.info(f"[URL_CHECK] Validating: {url[:200] if url else 'EMPTY/NONE'}")

        if not url or not isinstance(url, str):
            logger.error(f"[URL_CHECK] URL is empty or not a string: {type(url)}")
            raise DownloadError.invalid_url(str(url))

        trimmed = url.strip()
        parsed = urlparse(trimmed)

        # Require scheme and host
        if not parsed.scheme or not parsed.netloc:
            raise DownloadError.invalid_url(trimmed)

        # Only allow http/https schemes
        if parsed.scheme.lower() not in ('http', 'https'):
            raise DownloadError.invalid_url(trimmed)

        # Block credentials in URL (prevents SSRF tricks)
        if parsed.username or parsed.password:
            raise DownloadError.blocked_url(trimmed)

        # Block common metadata and internal endpoints
        blocked_hosts = {
            '169.254.169.254',  # Azure/AWS metadata
            'metadata.google.internal',
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
        }

        hostname = (parsed.hostname or '').strip()
        if not hostname:
            raise DownloadError.invalid_url(trimmed)

        if hostname.lower() in blocked_hosts:
            raise DownloadError.blocked_url(trimmed)

        # Block private IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise DownloadError.blocked_url(trimmed)
        except ValueError:
            # Hostname is not an IP - allow unless on blocked list
            pass

        return True
    except DownloadError:
        # Bubble up specific download errors
        raise
    except Exception as e:
        logger.error(f"[URL_CHECK] Exception during URL validation: {e}")
        raise DownloadError.invalid_url(str(url)) from e


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
    _is_safe_url(url)

    logger.info(f"Downloading PDF from {url[:100]}...")

    def _perform_request(target_url: str) -> requests.Response:
        return requests.get(
            target_url,
            timeout=DOWNLOAD_TIMEOUT,
            stream=True,
            headers={'User-Agent': 'SJ-PDF-Compressor/1.0'},
            allow_redirects=False  # Handle redirects manually to re-validate
        )

    try:
        response = _perform_request(url)

        # Handle a single redirect with re-validation
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            redirect_target = response.headers.get("location")
            if not redirect_target:
                raise DownloadError.invalid_url(url)
            redirect_url = urljoin(url, redirect_target)
            _is_safe_url(redirect_url)  # Re-validate redirect destination
            logger.info(f"[URL_REDIRECT] Following redirect to {redirect_url[:200]}")
            response = _perform_request(redirect_url)

        status = response.status_code
        if status == 404:
            raise DownloadError.not_found(url)
        if status in (401, 403):
            raise DownloadError.expired_or_forbidden(url, status_code=status)
        if status >= 400:
            raise DownloadError(f"Download failed (HTTP {status})", status_code=status)

        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            size_mb = int(content_length) / (1024 * 1024)
            raise DownloadError.too_large(size_mb)

        # Stream download with size check
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_SIZE:
                    raise DownloadError.too_large(downloaded / (1024 * 1024))
                f.write(chunk)

        logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB to {output_path.name}")

        # Basic sanity check: ensure we actually downloaded something
        if downloaded == 0 or output_path.stat().st_size == 0:
            raise DownloadError("Downloaded file is empty", status_code=400)

    except DownloadError:
        # Re-raise structured download errors
        raise
    except requests.exceptions.Timeout as e:
        raise DownloadError("Download timed out after 5 minutes", status_code=504) from e
    except requests.exceptions.RequestException as e:
        raise DownloadError(f"Download failed: {e}", status_code=502) from e
