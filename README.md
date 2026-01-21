# auto-compressor (internal)

## Overview
This service accepts PDFs, compresses them with Ghostscript, and optionally splits output
when the caller provides a split threshold. It exposes an HTTP API used by Salesforce and
a lightweight dashboard for manual testing.

## Core behavior
- Inputs: URL, upload, or base64 (async).
- Compression: serial for smaller files, parallel for large files or high page counts.
- Splitting: only when `split_threshold_mb` (alias `splitSizeMB`) is provided; otherwise output is a single PDF.
- Outputs: download links; async requests can callback to Salesforce.

## Endpoints (summary)
- `POST /compress` (async): JSON with `file_download_link` or `file_content_base64`.
  Optional: `name`, `pages`, `password`, `callbackUrl`, `split_threshold_mb` (`splitSizeMB`), `matterId`.
- `POST /compress-sync` (blocking): JSON with `file_download_link` or multipart upload.
  Optional: `name`, `split_threshold_mb` (`splitSizeMB`). If `matterId` is present it switches to async.
- `GET /status/<job_id>`
- `GET /download/<filename>`
- `POST /job/check`
- `GET /diagnose/<job_id>`

## Run locally
- Python 3.11
- Ghostscript installed (`gs` on PATH)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Key configuration (Azure App Settings)
- `API_TOKEN`
- `BASE_URL`
- `SALESFORCE_CALLBACK_URL`
- `COMPRESSION_MODE` (aggressive/lossless/adaptive)
- `ALLOW_LOSSY_COMPRESSION` (1/0)
- `PARALLEL_MAX_WORKERS`
- `TARGET_CHUNK_MB`, `MAX_CHUNK_MB`, `MAX_PAGES_PER_CHUNK`
- `FILE_RETENTION_SECONDS`
- `SYNC_MAX_MB`, `SYNC_AUTO_ASYNC_MB` (only if using `/compress-sync` for large files)

## Before running MB tests
1) Decide split behavior per test: omit `split_threshold_mb` to keep a single output; provide it to force splitting.
2) Confirm the compression/parallel settings you want (`COMPRESSION_MODE`, `ALLOW_LOSSY_COMPRESSION`,
   `PARALLEL_MAX_WORKERS`, chunk sizes).
3) Restart the service to pick up any config changes.
4) Clear `uploads/` if you want clean disk usage and cleaner timing comparisons.
5) Verify logs show the expected worker cap and that each job logs `split=off` or the threshold you set.
