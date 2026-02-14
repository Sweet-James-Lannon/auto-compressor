"""Route blueprints."""

from pdf_compressor.routes.api_routes import api_bp
from pdf_compressor.routes.web_routes import web_bp

__all__ = ["api_bp", "web_bp"]
