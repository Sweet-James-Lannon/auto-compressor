#!/bin/bash
# Azure App Service startup script
# Installs Ghostscript and qpdf for PDF compression, then starts gunicorn

set -e

echo "=== PDF Compressor Startup Script ==="

# Install system dependencies for PDF compression
echo "Installing system dependencies..."
apt-get update
apt-get install -y ghostscript qpdf

# Verify installations
echo "Verifying installations..."
gs --version && echo "Ghostscript installed successfully" || echo "WARNING: Ghostscript installation failed"
qpdf --version && echo "qpdf installed successfully" || echo "WARNING: qpdf installation failed"

# Start gunicorn with production settings
echo "Starting gunicorn..."
gunicorn --bind=0.0.0.0:8000 --timeout 300 --workers 2 app:app
