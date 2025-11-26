# SJ PDF Compressor

**Reduce PDF file sizes by up to 90% for faster uploads and lower storage costs.**

---

## What It Does (The Simple Version)

This service takes large PDF files (like scanned legal documents) and makes them smaller - much smaller. A 100MB file can become 10MB while keeping the document perfectly readable.

**Why it matters:**
- Faster uploads to Salesforce
- Lower storage costs
- Faster email attachments
- Better performance for case workers

---

## Quick Start

### Run Locally

```bash
# Clone and setup
git clone https://github.com/Sweet-James-Lannon/auto-compressor.git
cd auto-compressor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
python app.py
```

Open http://localhost:5005 for the web dashboard.

### Environment Variables

Create a `.env` file (optional):

```env
API_TOKEN=your-secret-token-here  # Required for Salesforce integration
PORT=5005                         # Default: 5005
FILE_RETENTION_SECONDS=600        # Auto-delete files after 10 minutes
```

---

## API Reference

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "ghostscript": true,
  "timestamp": "2024-11-25T14:30:00.000Z"
}
```

### Compress PDF

```http
POST /compress
Authorization: Bearer your-api-key
Content-Type: multipart/form-data
```

**Request:** Upload a PDF file with field name `pdf`

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
- `403` - Invalid API key
- `413` - File too large (max 300MB)
- `500` - Compression failed

### Download File

```http
GET /download/<filename>
```

Direct file download. Files auto-delete after 10 minutes.

---

## Integration Examples

### cURL

```bash
curl -X POST https://your-service.azurewebsites.net/compress \
  -H "Authorization: Bearer your-api-key" \
  -F "pdf=@document.pdf" \
  -o response.json

# Extract the compressed PDF
cat response.json | jq -r '.compressed_pdf_b64' | base64 -d > compressed.pdf
```

### Salesforce (Apex)

```apex
HttpRequest req = new HttpRequest();
req.setEndpoint('https://your-service.azurewebsites.net/compress');
req.setMethod('POST');
req.setHeader('Authorization', 'Bearer ' + apiKey);
// ... attach PDF and send
```

---

## Technical Architecture

```
┌─────────────────────────────────────────────────┐
│                  Web Dashboard                   │
│              (dashboard/dashboard.html)          │
└─────────────────────┬───────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────┐
│                   app.py                         │
│  • Flask routes (/health, /compress, /download) │
│  • API key authentication                        │
│  • File upload/cleanup management                │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                 compress.py                      │
│  • Orchestrates compression                      │
│  • Calculates metrics                            │
│  • Smart fallback (returns original if bigger)   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│            compress_ghostscript.py               │
│  • Ghostscript subprocess wrapper                │
│  • 72 DPI optimization for scanned docs          │
│  • JPEG2000 → JPEG conversion                    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              Ghostscript Binary                  │
│        (Industry-standard PDF processor)         │
└─────────────────────────────────────────────────┘
```

### Why Ghostscript?

We evaluated multiple PDF compression tools:

| Tool | Result |
|------|--------|
| **Ghostscript** | Best results for scanned legal docs. Industry standard. |
| pikepdf | Good for lossless, struggled with JPEG2000 images |
| qpdf | Great optimizer, but Ghostscript alone performed better |

Ghostscript is the standard tool used by PDF software worldwide. It handles our scanned legal documents better than alternatives.

### Why 72 DPI?

Through testing, we found that 72 DPI gives the best compression for scanned documents:

- **72 DPI**: 29% reduction on 115MB test file ✓
- **150 DPI**: Made files BIGGER (JPEG2000 issue)
- **200 DPI**: Made files BIGGER

This happens because our documents use JPEG2000 encoding. Higher DPI settings try to preserve too much detail, actually increasing file size.

### Smart Fallback

If compression would make a file larger (already-optimized PDFs), we return the original file instead. See `compress.py:56-69`.

---

## Azure Deployment

### Prerequisites

- Azure CLI installed
- Resource group created

### Deploy

```bash
RESOURCE_GROUP="your-resource-group"
APP_NAME="sj-pdf-compressor"

# Create App Service
az appservice plan create \
  --name "${APP_NAME}-plan" \
  --resource-group $RESOURCE_GROUP \
  --sku P1V2 --is-linux

az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan "${APP_NAME}-plan" \
  --name $APP_NAME \
  --runtime "PYTHON:3.11"

# Set API token
az webapp config appsettings set \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --settings API_TOKEN="your-secure-token"

# Deploy
zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*"
az webapp deployment source config-zip \
  --resource-group $RESOURCE_GROUP \
  --name $APP_NAME \
  --src deploy.zip
```

---

## Troubleshooting

### "Ghostscript not installed"

The health endpoint returns `"ghostscript": false`. Install Ghostscript:
- **Mac**: `brew install ghostscript`
- **Ubuntu**: `apt-get install ghostscript`
- **Azure**: Already handled by `startup.sh`

### "Invalid token" (403)

Check that:
1. `API_TOKEN` environment variable is set
2. Request includes `Authorization: Bearer your-token` header
3. Token matches exactly (no extra spaces)

### Low compression ratio

Some PDFs don't compress well:
- Already-compressed PDFs may not shrink further
- Text-only PDFs compress less than scanned images
- The service returns the original if compression would make it larger

---

## Security

- **Authentication**: Bearer token via `API_TOKEN` environment variable
- **Validation**: PDF magic bytes checked (`%PDF-` signature)
- **File limits**: 300MB maximum upload size
- **Auto-cleanup**: Files deleted after 10 minutes (configurable)
- **Request tracking**: Unique ID per request for debugging

---

## License

MIT License - see [LICENSE](LICENSE) file.
