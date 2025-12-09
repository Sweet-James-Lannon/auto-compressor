# SJ PDF Compressor

Compresses PDFs and auto-splits into ≤25MB parts for email/Salesforce. Parallel Ghostscript is used for large files; sync limit is 300MB.

## Usage

```bash
POST /compress-sync
Authorization: Bearer <token>
Content-Type: application/json

{"file_download_link": "https://source-url/document.pdf"}
```

Response:
```json
{
  "success": true,
  "files": ["/download/part1.pdf", "/download/part2.pdf"],
  "original_mb": 115.2,
  "compressed_mb": 45.8
}
```

Errors return `error_type`/`error_message` (e.g., `DownloadError`, `FileTooLarge`, `Timeout`, `InvalidPDF`).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_TOKEN` | - | Required for authentication |
| `SPLIT_THRESHOLD_MB` | 25 | Max size per output file |
| `FILE_RETENTION_SECONDS` | 86400 | Auto-delete after 24h |
| `PARALLEL_MAX_WORKERS` | 2 | Ghostscript workers per request (parallel path) |
| `SYNC_TIMEOUT_SECONDS` | 540 | Timeout for `/compress-sync` before 504 |
| `DISABLE_ASYNC_WORKERS` | unset | Set to `1` to skip async worker startup |

## Deployment

Azure App Service startup command:
```bash
apt-get update && apt-get install -y ghostscript && gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=1 --threads=16 app:app
```

Instance count must be 1. Multiple instances cause download failures due to non-shared local storage.

## Architecture

- Files < ~30MB: Serial Ghostscript compression
- Files > ~30MB: Parallel chunk compression (workers configurable via `PARALLEL_MAX_WORKERS`)
- If compression increases size: return original and split if above threshold
- Final split uses `ceil(size/threshold)` so outputs stay under `SPLIT_THRESHOLD_MB`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /compress-sync | Synchronous compression |
| POST | /compress | Async compression (returns job_id) |
| GET | /status/{job_id} | Poll async job status |
| GET | /download/{filename} | Download compressed file |
| GET | /health | Service health check |

## Local Development

```bash
brew install ghostscript
pip install -r requirements.txt
python app.py
```

## License

MIT
