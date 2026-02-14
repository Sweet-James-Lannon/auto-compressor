"""Shared utility functions for PDF compression service.

Contains:
- get_file_size_mb: Get file size in MB
- download_pdf: Download PDF from URL with security validation
"""

import ipaddress
import logging
import os
import socket
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests

from pdf_compressor.core.exceptions import DownloadError

# Constants for PDF download
DOWNLOAD_TIMEOUT: int = 300  # 5 minute timeout
MAX_DOWNLOAD_SIZE: int = 524288000  # 500 MB max

logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.environ.get(name) or "").strip()
    return raw if raw in allowed else default


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.
    """
    return path.stat().st_size / (1024 * 1024)


def redact_url_for_log(url: str, max_len: int = 200) -> str:
    """Return a URL safe for logs (no query/fragment/userinfo)."""
    if not url or not isinstance(url, str):
        return "EMPTY/NONE"

    trimmed = url.strip()
    try:
        parsed = urlparse(trimmed)
        if not parsed.scheme or not parsed.netloc:
            safe = trimmed
        else:
            host = parsed.hostname or ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            if parsed.port:
                netloc = f"{host}:{parsed.port}"
            else:
                netloc = host
            safe = parsed._replace(netloc=netloc, query="", fragment="", params="").geturl()
    except Exception:
        safe = trimmed

    if len(safe) > max_len:
        return safe[:max_len] + "..."
    return safe


def _parse_cpu_list(cpu_list: str) -> int:
    """Parse cpuset list format like '0-3,5' into a CPU count."""
    count = 0
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                count += int(end) - int(start) + 1
            except ValueError:
                return 0
        else:
            try:
                int(part)
            except ValueError:
                return 0
            count += 1
    return count


def get_effective_cpu_count(default: int = 1) -> int:
    """Return effective CPU count, respecting cgroup quotas when present."""
    host_count = os.cpu_count() or default
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            host_count = min(host_count, len(affinity))
    except (AttributeError, OSError):
        pass
    quota_count = None

    # cgroup v2
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        try:
            quota_str, period_str = cpu_max.read_text().strip().split()[:2]
            if quota_str != "max":
                quota = int(quota_str)
                period = int(period_str)
                if quota > 0 and period > 0:
                    quota_count = max(1, int(quota / period))
        except Exception:
            quota_count = None

    # cgroup v1
    if quota_count is None:
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.exists() and period_path.exists():
            try:
                quota = int(quota_path.read_text().strip())
                period = int(period_path.read_text().strip())
                if quota > 0 and period > 0:
                    quota_count = max(1, int(quota / period))
            except Exception:
                quota_count = None

    cpuset_count = None
    for cpuset_path in (
        Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        Path("/sys/fs/cgroup/cpuset.cpus"),
    ):
        if cpuset_path.exists():
            cpu_list = cpuset_path.read_text().strip()
            if cpu_list:
                count = _parse_cpu_list(cpu_list)
                if count:
                    cpuset_count = count
                    break

    effective = host_count
    if quota_count:
        effective = min(effective, quota_count)
    if cpuset_count:
        effective = min(effective, cpuset_count)

    return max(default, effective)


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_hostname_ips(hostname: str) -> list:
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return []

    ips = []
    for info in addrinfos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip not in ips:
            ips.append(ip)
    return ips


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
        logger.info("[URL_CHECK] Validating: %s", redact_url_for_log(url))

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
            if _is_disallowed_ip(ip):
                raise DownloadError.blocked_url(trimmed)
        except ValueError:
            resolved_ips = _resolve_hostname_ips(hostname)
            if not resolved_ips:
                raise DownloadError.invalid_url(trimmed)
            for ip in resolved_ips:
                if _is_disallowed_ip(ip):
                    raise DownloadError.blocked_url(trimmed)

        return True
    except DownloadError:
        # Bubble up specific download errors
        raise
    except Exception as e:
        logger.error(f"[URL_CHECK] Exception during URL validation: {e}")
        raise DownloadError.invalid_url(str(url)) from e


def validate_external_url(url: str) -> None:
    """Validate URL safety for outbound HTTP requests (download/callback)."""
    _is_safe_url(url)


def download_pdf(url: str, output_path: Path, max_download_size_bytes: int = MAX_DOWNLOAD_SIZE) -> None:
    """Download a PDF from a URL.

    Used by async endpoints to fetch PDFs from Salesforce/Docrio URLs.

    Args:
        url: The URL to download from.
        output_path: Path to save the downloaded file.
        max_download_size_bytes: Maximum allowed download size in bytes.

    Raises:
        DownloadError: If download fails, URL is blocked, or file is too large.
    """
    # Security: Validate URL to prevent SSRF
    validate_external_url(url)

    logger.info("Downloading PDF from %s...", redact_url_for_log(url, max_len=120))

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
            validate_external_url(redirect_url)  # Re-validate redirect destination
            logger.info("[URL_REDIRECT] Following redirect to %s", redact_url_for_log(redirect_url))
            try:
                response.close()
            except Exception:
                pass
            response = _perform_request(redirect_url)

        status = response.status_code
        if status == 404:
            raise DownloadError.not_found(url)
        if status in (401, 403):
            raise DownloadError.expired_or_forbidden(url, status_code=status)
        if status >= 400:
            raise DownloadError(f"Download failed (HTTP {status})", status_code=status)

        # Check Content-Length if available
        expected_bytes = None
        content_length = response.headers.get('content-length')
        if content_length:
            try:
                expected_bytes = int(content_length)
            except Exception:
                expected_bytes = None

        if expected_bytes and expected_bytes > max_download_size_bytes:
            size_mb = expected_bytes / (1024 * 1024)
            limit_mb = max_download_size_bytes / (1024 * 1024)
            raise DownloadError.too_large(size_mb, limit_mb=limit_mb)

        # If the server applies Content-Encoding, requests may transparently decompress the body,
        # making `downloaded` larger than the on-the-wire Content-Length. Only enforce strict
        # byte-for-byte matching when Content-Encoding is identity/absent.
        content_encoding = (response.headers.get("content-encoding") or "").strip().lower()
        strict_length = expected_bytes is not None and content_encoding in ("", "identity")
        if expected_bytes is not None and not strict_length:
            logger.info(
                f"[DOWNLOAD_SIZE] Content-Encoding='{content_encoding}' detected; "
                f"skipping strict Content-Length verification"
            )

        # Stream download with size check
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_download_size_bytes:
                    limit_mb = max_download_size_bytes / (1024 * 1024)
                    raise DownloadError.too_large(downloaded / (1024 * 1024), limit_mb=limit_mb)
                if strict_length and expected_bytes is not None and downloaded > expected_bytes:
                    raise DownloadError(
                        f"Download exceeded Content-Length ({downloaded:,} > {expected_bytes:,} bytes). "
                        f"The source may be misreporting file size.",
                        status_code=502,
                    )
                f.write(chunk)

        if strict_length and expected_bytes is not None and downloaded != expected_bytes:
            raise DownloadError(
                f"Download size mismatch (expected {expected_bytes:,} bytes, got {downloaded:,} bytes). "
                f"The download may be incomplete.",
                status_code=502,
            )

        logger.info(
            f"Downloaded {downloaded:,} bytes "
            f"({downloaded / (1024 * 1024):.1f}MB) "
            f"to {output_path.name}"
        )
        if strict_length and expected_bytes is not None:
            logger.info(f"[DOWNLOAD_SIZE] Verified Content-Length: {expected_bytes:,} bytes")

        # Basic sanity check: ensure we actually downloaded something
        if downloaded == 0 or output_path.stat().st_size == 0:
            raise DownloadError("Downloaded file is empty", status_code=400)

    except DownloadError:
        output_path.unlink(missing_ok=True)
        # Re-raise structured download errors
        raise
    except requests.exceptions.Timeout as e:
        output_path.unlink(missing_ok=True)
        raise DownloadError("Download timed out after 5 minutes", status_code=504) from e
    except requests.exceptions.RequestException as e:
        output_path.unlink(missing_ok=True)
        raise DownloadError(f"Download failed: {e}", status_code=502) from e
