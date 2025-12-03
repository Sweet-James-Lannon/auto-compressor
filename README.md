# SJ PDF Compressor

PDF compression service for Salesforce integration. Compresses scanned documents using Ghostscript with automatic splitting for large files.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Requires Ghostscript: `brew install ghostscript` (Mac) or `apt-get install ghostscript` (Linux)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_TOKEN` | Bearer token for auth | None (open) |
| `PORT` | Server port | 5005 |
| `FILE_RETENTION_SECONDS` | Auto-cleanup interval | 86400 (24h) |
| `SPLIT_THRESHOLD_MB` | Max file size before auto-split | 25 |
| `MAX_SPLIT_PARTS` | Maximum parts when splitting | 50 |

## Features

### Automatic PDF Splitting
Files exceeding 25MB are automatically split into email-safe parts:
- Smart page distribution based on content size
- Each part optimized with Ghostscript
- Configure threshold via `SPLIT_THRESHOLD_MB`

### Web Dashboard
Access `/` for a web UI with drag-and-drop upload and real-time status.

### Error Handling
Structured errors with user-friendly messages:

| Error Type | HTTP | Cause |
|------------|------|-------|
| EncryptionError | 422 | Password-protected PDF |
| StructureError | 422 | Corrupted/damaged PDF |
| CompressionFailureError | 422 | Already optimized |
| SplitError | 422 | Cannot split small enough |

## API

### POST /compress

Compresses a PDF asynchronously. Returns a job ID for polling.

**Headers:**
```
Authorization: Bearer <your-token>
```

**Request Options:**

1. **Form data upload:**
```
Content-Type: multipart/form-data
pdf: <file>
```

2. **Base64 JSON:**
```json
{
  "file_content_base64": "<base64-encoded-pdf>"
}
```

3. **URL download:**
```json
{
  "file_download_link": "https://example.com/document.pdf"
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "job_id": "a1b2c3d4e5f67890",
  "status_url": "/status/a1b2c3d4e5f67890"
}
```

**Error Codes:**
- `400` - Invalid PDF or missing file
- `401` - Missing Authorization header
- `403` - Invalid token
- `413` - File too large (max 300MB)

### GET /status/<job_id>

Poll for job completion. Requires auth if `API_TOKEN` is set.

**Processing:**
```json
{
  "success": true,
  "status": "processing"
}
```

**Completed (single file):**
```json
{
  "success": true,
  "status": "completed",
  "download_links": ["/download/doc_compressed.pdf"],
  "original_mb": 50.0,
  "compressed_mb": 5.2,
  "reduction_percent": 89.6,
  "was_split": false,
  "total_parts": 1
}
```

**Completed (split into parts):**
```json
{
  "success": true,
  "status": "completed",
  "was_split": true,
  "total_parts": 3,
  "download_links": [
    "/download/doc_part1.pdf",
    "/download/doc_part2.pdf",
    "/download/doc_part3.pdf"
  ],
  "original_mb": 80.0,
  "compressed_mb": 45.0,
  "reduction_percent": 43.8
}
```

**Failed:**
```json
{
  "success": false,
  "status": "failed",
  "error": "PDF is password-protected",
  "error_type": "EncryptionError"
}
```

### GET /health

Returns service status and Ghostscript availability.

### GET /download/<filename>

Direct file download. Files auto-delete after retention period.

## Integration Guide

For n8n, Salesforce, or any custom application:

| Item | Value |
|------|-------|
| **Endpoint URL** | `https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress` |
| **Payload** | Form data, base64 JSON, or URL JSON |
| **Auth** | Header: `Authorization: Bearer <your-token>` |

### cURL

```bash
# Submit for compression
JOB=$(curl -s -X POST https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress \
  -H "Authorization: Bearer your-token" \
  -F "pdf=@document.pdf" | jq -r '.job_id')

# Poll for completion
curl -s "https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/status/$JOB" \
  -H "Authorization: Bearer your-token"

# Download result (filename from download_links)
curl -O "https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/download/document_compressed.pdf"
```

### Python

```python
import requests
import time

BASE_URL = "https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net"
HEADERS = {"Authorization": "Bearer your-token"}

# Submit
with open("document.pdf", "rb") as f:
    resp = requests.post(f"{BASE_URL}/compress", headers=HEADERS, files={"pdf": f})
job_id = resp.json()["job_id"]

# Poll until complete
while True:
    status = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS).json()
    if status["status"] != "processing":
        break
    time.sleep(2)

# Download all parts
if status["status"] == "completed":
    for link in status["download_links"]:
        r = requests.get(f"{BASE_URL}{link}")
        filename = link.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(r.content)
```

### Salesforce Apex

```apex
// Submit compression job
HttpRequest req = new HttpRequest();
req.setEndpoint('https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress');
req.setMethod('POST');
req.setHeader('Authorization', 'Bearer ' + API_TOKEN);
req.setHeader('Content-Type', 'application/json');
req.setBody('{"file_content_base64": "' + EncodingUtil.base64Encode(pdfBlob) + '"}');

Http http = new Http();
HttpResponse res = http.send(req);
Map<String, Object> result = (Map<String, Object>)JSON.deserializeUntyped(res.getBody());
String jobId = (String)result.get('job_id');

// Poll /status/{jobId} for completion, then download from download_links
```

## Project Structure

```
app.py                  # Flask routes, auth, async job handling
compress.py             # Compression orchestration with splitting
compress_ghostscript.py # Ghostscript wrapper (72 DPI compression)
split_pdf.py            # PDF splitting for large files
job_queue.py            # Async job queue and URL download
exceptions.py           # Custom error types
startup.sh              # Azure startup (installs Ghostscript)
dashboard/              # Web UI
```

### Why startup.sh?

Azure App Service runs on a Linux container that doesn't include Ghostscript by default. The `startup.sh` script runs on each deployment to:
1. Install Ghostscript via `apt-get`
2. Verify the installation
3. Start the Flask app with gunicorn

Without this, the `/health` endpoint would return `"ghostscript": false` and compression would fail.

## Technical Notes

- Uses 72 DPI for scanned docs (higher DPI increases size for JPEG2000)
- Returns original file if compression makes it larger
- 300MB max file size
- Files auto-split if >25MB after compression
- HTTPS handled by Azure App Service

## Deployment

Deployed via GitHub Actions to Azure App Service. Push to `main` triggers auto-deploy.

Set `API_TOKEN` in Azure Portal > Configuration > Application settings.

## License

MIT
