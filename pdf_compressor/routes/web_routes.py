"""Web and health routes."""

from flask import Blueprint

from pdf_compressor.services import compression_service

web_bp = Blueprint("web", __name__)


@web_bp.get("/")
def dashboard():
    return compression_service.dashboard()


@web_bp.get("/health")
def health():
    return compression_service.health()


@web_bp.get("/favicon.ico")
def favicon():
    return compression_service.favicon()
