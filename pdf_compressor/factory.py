"""Flask app factory."""

from __future__ import annotations

from flask import Flask

from pdf_compressor import bootstrap
from pdf_compressor.config import load_runtime_config
from pdf_compressor.routes.api_routes import api_bp
from pdf_compressor.routes.web_routes import web_bp
from pdf_compressor.services import compression_service


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates")

    runtime_config = load_runtime_config()
    app.config["MAX_CONTENT_LENGTH"] = runtime_config.max_content_length
    compression_service.configure_app(app)

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    compression_service.register_error_handlers(app)

    bootstrap.bootstrap_runtime()
    return app
