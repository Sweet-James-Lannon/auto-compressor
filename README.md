# SJ PDF Compressor

Compresses scanned PDFs for email attachment compliance. Auto-splits files exceeding 25MB.

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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_TOKEN` | - | Required for authentication |
| `SPLIT_THRESHOLD_MB` | 25 | Max size per output file |
| `FILE_RETENTION_SECONDS` | 86400 | Auto-delete after 24h |

## Deployment

Azure App Service startup command:
```bash
apt-get update && apt-get install -y ghostscript && gunicorn --bind=0.0.0.0:8000 --timeout=600 --workers=1 --threads=16 app:app
```

Instance count must be 1. Multiple instances cause download failures due to non-shared local storage.

## Architecture

- Files < 60MB: Serial Ghostscript compression
- Files > 60MB: Parallel chunk compression (6 workers)
- If compression increases size: Falls back to split-only

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
