"""PDF compressor package."""

__all__ = ["create_app"]


def create_app():
    """Lazily import app factory to avoid import-time side effects."""
    from pdf_compressor.factory import create_app as _create_app

    return _create_app()
