#!/bin/bash
# Azure App Service startup script
# Installs Ghostscript for PDF compression, then starts gunicorn

set -e

echo "=== PDF Compressor Startup Script ==="

# Install system dependencies for PDF compression
echo "Installing system dependencies..."
apt-get update
apt-get install -y ghostscript

# Verify installations
echo "Verifying installations..."
gs --version && echo "Ghostscript installed successfully" || echo "WARNING: Ghostscript installation failed"

# Start gunicorn with production settings
# Using single worker with threads to maintain shared memory for job queue
echo "Starting gunicorn..."
gunicorn --bind=0.0.0.0:8000 --timeout 300 --workers 1 --threads 4 app:app
