# SJ PDF Compressor

PDF compression service for Salesforce integration. Compresses scanned documents using Ghostscript.

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
| `FILE_RETENTION_SECONDS` | Auto-cleanup interval | 600 |

## API

### POST /compress

Compresses a PDF and returns the result as base64.

**Headers:**
```
Authorization: Bearer <your-token>
Content-Type: multipart/form-data
```

**Request (Form Data):**
```
pdf: <file>
```

**Request (JSON alternative):**
```json
{
  "file_content_base64": "<base64-encoded-pdf>"
}
```

**Response:**
```json
{
  "success": true,
  "original_mb": 50.0,
  "compressed_mb": 5.2,
  "reduction_percent": 89.6,
  "compressed_pdf_b64": "JVBERi0xLjQK...",
  "compression_method": "ghostscript",
  "request_id": "a1b2c3d4"
}
```

**Error Codes:**
- `400` - Invalid PDF or missing file
- `401` - Missing Authorization header
- `403` - Invalid token
- `413` - File too large (max 300MB)
- `500` - Compression failed

### GET /health

Returns service status and Ghostscript availability.

### GET /download/<filename>

Direct file download. Files auto-delete after retention period.

## Integration Guide

For n8n, Salesforce, or any custom application, you need three things:

| Item | Value |
|------|-------|
| **Endpoint URL** | `https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress` |
| **Payload** | Form data with `pdf` field, or JSON with `file_content_base64` |
| **Auth** | Header: `Authorization: Bearer <your-token>` |

### cURL

```bash
curl -X POST https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress \
  -H "Authorization: Bearer your-token" \
  -F "pdf=@document.pdf"
```

### Python

```python
import requests
response = requests.post(
    "https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress",
    headers={"Authorization": "Bearer your-token"},
    files={"pdf": open("document.pdf", "rb")}
)
compressed = base64.b64decode(response.json()["compressed_pdf_b64"])
```

### Salesforce Apex

```apex
HttpRequest req = new HttpRequest();
req.setEndpoint('https://sj-doc-compressor-g4esbvbgd5fdesee.westus3-01.azurewebsites.net/compress');
req.setMethod('POST');
req.setHeader('Authorization', 'Bearer ' + API_TOKEN);
req.setHeader('Content-Type', 'application/json');
req.setBody('{"file_content_base64": "' + EncodingUtil.base64Encode(pdfBlob) + '"}');

Http http = new Http();
HttpResponse res = http.send(req);
String compressedBase64 = (String)JSON.deserializeUntyped(res.getBody()).get('compressed_pdf_b64');
```

## Project Structure

```
app.py                  # Flask routes, auth, file handling
compress.py             # Compression orchestration
compress_ghostscript.py # Ghostscript wrapper
startup.sh              # Azure startup script (installs Ghostscript)
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
- HTTPS handled by Azure App Service

## Deployment

Deployed via GitHub Actions to Azure App Service. Push to `main` triggers auto-deploy.

Set `API_TOKEN` in Azure Portal > Configuration > Application settings.

## License

MIT
