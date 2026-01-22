#!/bin/bash
# Azure App Service startup script
# Installs Ghostscript if needed, then starts gunicorn

set -euo pipefail

echo "=== PDF Compressor Startup Script ==="

if ! command -v gs >/dev/null 2>&1; then
  echo "Installing system dependencies..."
  apt-get update -qq
  apt-get install -y --no-install-recommends ghostscript
  rm -rf /var/lib/apt/lists/*
fi

echo "Verifying installations..."
gs --version && echo "Ghostscript installed successfully" || echo "WARNING: Ghostscript installation failed"

PORT="${PORT:-8000}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-900}"
GUNICORN_THREADS="${GUNICORN_THREADS:-6}"

# Start gunicorn with production settings.
# Use a single worker to keep the in-memory job queue consistent.
echo "Starting gunicorn on port ${PORT}..."
exec gunicorn --bind "0.0.0.0:${PORT}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --workers 1 \
  --threads "${GUNICORN_THREADS}" \
  app:app
