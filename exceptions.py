"""Custom exceptions for PDF compression operations.

All error messages are written in plain English so users know exactly
what went wrong and how to fix it.
"""

from typing import Optional


class PDFCompressionError(Exception):
    """Base exception for all PDF compression errors."""

    error_type: str = "PDFCompressionError"
    status_code: int = 400

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        # Allow subclasses to override status codes
        if hasattr(self, "status_code_override"):
            try:
                self.status_code = int(getattr(self, "status_code_override"))  # type: ignore[attr-defined]
            except Exception:
                pass


class EncryptionError(PDFCompressionError):
    """PDF is password-protected or locked.

    User-friendly message examples:
    - "This PDF is password-protected. Please remove the password and try again."
    - "This PDF is locked. Please unlock it before uploading."
    """

    error_type: str = "EncryptionError"

    @staticmethod
    def for_file(filename: str) -> "EncryptionError":
        """Create error with simple message for a specific file."""
        return EncryptionError(
            f"'{filename}' is password-protected or locked. "
            f"Please remove the password and try again."
        )


class MetadataCorruptionError(PDFCompressionError):
    """PDF has corrupted internal data.

    User-friendly message examples:
    - "This PDF has corrupted data. Try re-saving it from the original program."
    - "This PDF's internal data is damaged. Please use a different copy."
    """

    error_type: str = "MetadataCorruptionError"

    @staticmethod
    def for_file(filename: str) -> "MetadataCorruptionError":
        """Create error with simple message for a specific file."""
        return MetadataCorruptionError(
            f"'{filename}' has corrupted internal data. "
            f"Try opening it in Adobe Acrobat and re-saving it, or use a different copy."
        )


class StructureError(PDFCompressionError):
    """PDF file is damaged or malformed.

    User-friendly message examples:
    - "This PDF is damaged and cannot be processed."
    - "This PDF file appears to be corrupted. Please use a different copy."
    """

    error_type: str = "StructureError"

    @staticmethod
    def for_file(filename: str, detail: str = "") -> "StructureError":
        """Create error with simple message for a specific file."""
        base_msg = f"'{filename}' is damaged and cannot be processed."
        if detail:
            return StructureError(f"{base_msg} Issue: {detail}")
        return StructureError(
            f"{base_msg} Try opening it in Adobe Acrobat and re-saving it, "
            f"or use a different copy of the file."
        )


class SplitError(PDFCompressionError):
    """PDF cannot be split into small enough parts for email.

    User-friendly message examples:
    - "This PDF cannot be split into parts under 25MB for email."
    - "Even after splitting, some parts are still too large for email attachments."
    """

    error_type: str = "SplitError"

    @staticmethod
    def for_file(filename: str, threshold_mb: float, attempts: int) -> "SplitError":
        """Create error with simple message for a specific file."""
        return SplitError(
            f"'{filename}' cannot be split into parts under {threshold_mb:.0f}MB. "
            f"Tried {attempts} different ways to split it. "
            f"The PDF likely contains high-resolution images that can't be compressed further. "
            f"Try reducing image quality in the original document before uploading."
        )


class DownloadError(PDFCompressionError):
    """Errors related to downloading source PDFs (invalid, blocked, expired, not found)."""

    error_type: str = "DownloadError"

    def __init__(self, message: str, status_code: int = 400) -> None:
        self.status_code_override = status_code
        super().__init__(message)

    @staticmethod
    def invalid_url(url: str) -> "DownloadError":
        return DownloadError(f"Invalid URL: '{url}'", status_code=400)

    @staticmethod
    def blocked_url(url: str) -> "DownloadError":
        return DownloadError(f"URL is blocked for security reasons: '{url}'", status_code=400)

    @staticmethod
    def expired_or_forbidden(url: str, status_code: int) -> "DownloadError":
        return DownloadError(
            f"Download failed (HTTP {status_code}). The link may be expired or access-restricted.",
            status_code=404 if status_code == 404 else 403
        )

    @staticmethod
    def not_found(url: str) -> "DownloadError":
        return DownloadError(f"Download failed (404). File not found or link expired: '{url}'", status_code=404)

    @staticmethod
    def too_large(mb: float) -> "DownloadError":
        return DownloadError(f"File too large: {mb:.1f}MB (limit 300MB)", status_code=413)
