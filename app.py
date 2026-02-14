"""Compatibility entrypoint for Gunicorn and local runs."""

import os

from pdf_compressor.factory import create_app
from pdf_compressor.services import compression_service

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    compression_service.logger.info("Starting server on port %s", port)
    compression_service.logger.info("File retention: %ss", compression_service.FILE_RETENTION_SECONDS)
    app.run(host="0.0.0.0", port=port, debug=False)
