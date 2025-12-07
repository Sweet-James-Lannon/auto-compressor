# SJ PDF Compressor

PDF compression API for Salesforce. Compresses scanned medical demand documents using Ghostscript, with automatic splitting for files over 25MB (Outlook's attachment limit).

---

## How It Works (The Big Picture)

```
Case Manager uploads 115MB demand PDF in Salesforce
                    ↓
Salesforce sends URL to our API → POST /compress-sync
                    ↓
API downloads the PDF from Docreo/AWS
                    ↓
Ghostscript compresses it (115MB → 45MB)
                    ↓
Still over 25MB? Auto-split into parts (45MB → 2 parts × ~22MB each)
                    ↓
Return download URLs → { "files": ["/download/part1.pdf", "/download/part2.pdf"] }
                    ↓
Salesforce displays files, user attaches to Outlook email
```

### Why These Choices?

| Decision | Why |
|----------|-----|
| **25MB split threshold** | Outlook's attachment limit is 25MB. Parts must be under this to email. |
| **Ghostscript at 72 DPI** | Scanned medical docs are images. Lower DPI = smaller files. 72 DPI is readable but compact. |
| **Synchronous endpoint** | Salesforce can't easily poll async jobs. One request, one response is simpler for Jai's integration. |
| **Async endpoint kept** | Dashboard and large files (300MB) still use async for progress tracking. |

---

## Two Ways to Compress

### 1. Synchronous (for Salesforce) - `/compress-sync`

**Best for:** Salesforce integration, simple scripts, files under ~100MB

```
POST /compress-sync
  ↓ (blocks until done)
Response: { files: ["/download/part1.pdf", ...] }
```

One request. One response. No polling. Jai's Salesforce component shows a spinner while waiting.

**Limitation:** Salesforce has a 120-second HTTP timeout. Files over ~100MB may take longer to compress.

### 2. Asynchronous (for Dashboard/Large Files) - `/compress`

**Best for:** Web dashboard, very large files, when you need progress updates

```
POST /compress → { job_id: "abc123" }
GET /status/abc123 → { status: "processing", progress: { percent: 45 } }
GET /status/abc123 → { status: "completed", download_links: [...] }
```

Three steps, but you get progress updates and it handles huge files without timeout.

---

## API Reference

### POST /compress-sync

Synchronous compression for Salesforce. Blocks until complete.

**Request:**
```bash
curl -X POST https://your-domain/compress-sync \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_download_link": "https://docreo-aws-url/document.pdf"}'
```

**Success Response:**
```json
{
  "success": true,
  "files": ["/download/doc_part1.pdf", "/download/doc_part2.pdf"],
  "original_mb": 115.2,
  "compressed_mb": 45.8,
  "was_split": true,
  "total_parts": 2
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "PDF is password-protected",
  "error_type": "EncryptionError"
}
```

**How Jai Uses This:**
1. User clicks "Send to AI" in Salesforce
2. Jai's component extracts the Docreo download URL
3. Sends POST to `/compress-sync` with that URL
4. Shows spinner while waiting
5. Receives array of file URLs
6. Displays download links to user

---

### POST /compress

Async compression. Returns job ID, poll `/status/<job_id>` for results.

**Request (URL method):**
```json
{ "file_download_link": "https://example.com/document.pdf" }
```

**Request (upload method):**
```bash
curl -X POST https://your-domain/compress \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "pdf=@document.pdf"
```

**Response:**
```json
{
  "success": true,
  "job_id": "a1b2c3d4e5f67890",
  "status_url": "/status/a1b2c3d4e5f67890"
}
```

---

### GET /status/<job_id>

Poll for job completion.

**Processing:**
```json
{
  "success": true,
  "status": "processing",
  "progress": { "percent": 72, "stage": "splitting", "message": "Finding split point 2..." }
}
```

**Completed:**
```json
{
  "success": true,
  "status": "completed",
  "download_links": ["/download/doc_part1.pdf", "/download/doc_part2.pdf"],
  "original_mb": 80.0,
  "compressed_mb": 45.0,
  "was_split": true,
  "total_parts": 2
}
```

---

### GET /download/<filename>

Download a compressed file. Files auto-delete after 24 hours.

```bash
curl -O https://your-domain/download/doc_part1.pdf
```

---

### GET /health

Check if the service is running.

```json
{
  "status": "healthy",
  "ghostscript": true
}
```

---

## Error Types

| Error | HTTP | What It Means | User Message |
|-------|------|---------------|--------------|
| `EncryptionError` | 422 | PDF is password-protected | "This PDF is locked. Please unlock it first." |
| `StructureError` | 422 | PDF is corrupted/damaged | "This PDF appears to be damaged." |
| `SplitError` | 422 | Can't split small enough (single page too big) | "Individual pages are too large to split." |

---

## Setup

### Local Development

```bash
# Install Ghostscript
brew install ghostscript  # Mac
apt-get install ghostscript  # Linux

# Setup Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python app.py
```

### Environment Variables

| Variable | What It Does | Default |
|----------|--------------|---------|
| `API_TOKEN` | Required for auth. If not set, API is open. | None |
| `PORT` | Server port | 5005 |
| `FILE_RETENTION_SECONDS` | How long to keep files before auto-delete | 86400 (24h) |
| `SPLIT_THRESHOLD_MB` | Split files larger than this | 25 |

---

## Architecture

```
app.py                  # Flask routes, handles HTTP requests
   ↓
compress.py             # Orchestrates compression + splitting
   ↓
├── compress_ghostscript.py  # Runs Ghostscript (the actual compression)
└── split_pdf.py             # Splits large PDFs into parts
   ↓
job_queue.py            # Background workers for async jobs
```

### Why Each File Exists

| File | Purpose |
|------|---------|
| `app.py` | HTTP layer. Handles requests, auth, routes. |
| `compress.py` | Business logic. "Compress this PDF, split if needed." |
| `compress_ghostscript.py` | Ghostscript wrapper. Translates to `gs` commands. |
| `split_pdf.py` | Binary search algorithm to find optimal split points. |
| `job_queue.py` | Thread pool for async jobs. Downloads URLs. |
| `exceptions.py` | Custom error types (Encryption, Structure, Split). |
| `startup.sh` | Azure runs this on deploy to install Ghostscript. |

---

## Deployment (Azure)

Push to `main` → GitHub Actions → Azure App Service

**Important Azure Settings:**

1. **Application settings** → Set `API_TOKEN`
2. **General settings** → Startup Command should point to `startup.sh`
3. **Scale out** → Instance count = 1 (for shared memory in job queue)

### Why startup.sh?

Azure's Linux container doesn't have Ghostscript. The startup script:
1. Installs Ghostscript via `apt-get`
2. Starts the Flask app with Gunicorn

Without this, compression fails.

---

## Technical Details

### Gunicorn Configuration

```bash
gunicorn --bind=0.0.0.0:8000 --timeout=1800 --workers=1 --threads=8 app:app
```

| Setting | Value | Why |
|---------|-------|-----|
| `timeout` | 1800 (30 min) | Large files (300MB) take up to 35 min to compress |
| `workers` | 1 | Single process = shared memory for job queue. Multiple workers would cause 404 errors. |
| `threads` | 8 | Handle 8 concurrent HTTP requests |

### Split Algorithm

When a compressed file is still over 25MB, we split it:

1. Calculate parts needed: `ceil(file_size / 25MB)`
2. Binary search for split points (measuring actual compressed size)
3. Create each part, verify it's under 25MB
4. If a part is still too big, add more parts and retry

This ensures **every part is under 25MB**, not just estimated.

### Why 72 DPI?

Medical demands are scanned images. Higher DPI = more pixels = larger files.

- 300 DPI: Archival quality, huge files
- 150 DPI: Good balance, but still large
- **72 DPI: Screen-readable, much smaller files**

For email attachments, 72 DPI is sufficient.

---

## License

MIT
