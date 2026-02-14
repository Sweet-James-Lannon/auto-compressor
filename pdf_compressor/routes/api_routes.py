"""API routes."""

from flask import Blueprint

from pdf_compressor.services import compression_service, status_service

api_bp = Blueprint("api", __name__)

api_bp.add_url_rule(
    "/compress-async",
    endpoint="compress_async",
    view_func=compression_service.compress_async,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/compress-sync",
    endpoint="compress_sync",
    view_func=compression_service.compress_async,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/status/<job_id>",
    endpoint="get_status",
    view_func=status_service.get_status,
    methods=["GET"],
)
api_bp.add_url_rule(
    "/job/check",
    endpoint="job_check_pdfco",
    view_func=status_service.job_check_pdfco,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/download/<filename>",
    endpoint="download",
    view_func=compression_service.download,
    methods=["GET"],
)
api_bp.add_url_rule(
    "/diagnose/<job_id>",
    endpoint="diagnose",
    view_func=status_service.diagnose,
    methods=["GET"],
)
