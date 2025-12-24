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

## How it works (backend flow)

- Two modes:
  - `/compress-sync` (blocking) for dashboard/manual tests or legacy callers.
  - `/compress-sync` with `matterId` (new Salesforce flow): returns `202` fast, does the work in background, then POSTs results to `SALESFORCE_CALLBACK_URL`.
- Compression pipeline:
  - Downloads the PDF (or accepts upload/base64), validates header, tracks files for cleanup.
  - Uses Ghostscript; small files compress serially, big files can use parallel chunks.
  - If output is still large, auto-splits into ≤ `SPLIT_THRESHOLD_MB` parts and logs part sizes.
- Outputs are served from `/download/{file}`; `BASE_URL` makes links absolute and HTTPS for callbacks.
- Cleanup daemon deletes old files after `FILE_RETENTION_SECONDS`.

## Salesforce integration

- Entry point: Salesforce calls `POST /compress-sync` with `file_download_link` (presigned URL) and, for async, a `matterId`.
- Async behavior: when `matterId` is present, we enqueue the job and immediately return `202 + job_id`. A worker downloads the PDF, compresses/splits, then posts results to the prod callback URL in `SALESFORCE_CALLBACK_URL`. Download links in the callback are absolute, built with `BASE_URL`.
- Error callbacks: if download or processing fails, we still send a callback with `compressedLinks: []` and an `error` message. Download failures are usually expired/invalid presigned URLs (404/401/403).
- Auth: if `API_TOKEN` is set, Salesforce must send `Authorization: Bearer <token>`.
- Large files: use the async path (`matterId`) to avoid sync timeouts on big uploads.

## Why this is better than iLovePDF (for our use case)

- Self-hosted and API-driven: no external throttling or account limits; predictable SLAs.
- Tuned for Salesforce: async callback avoids Apex timeouts; includes `matterId` correlation and part sizes for verification.
- Auto-splitting built in: guarantees parts stay under our configured threshold without manual steps.
- Observability: structured errors, progress updates (async), and detailed logging for download/compression/splitting.
- Security/control: bearer auth option, no third-party data sharing, short-lived on-disk storage with cleanup.

## Quick testing (dashboard)

- Navigate to `/` to open the dashboard.
- Upload a PDF (or supply a URL) and submit; the dashboard uses `/compress-sync` (blocking) and returns download links directly.
- For Salesforce-style async: send JSON to `/compress-sync` with `file_download_link` and `matterId`; you will get `202` + `job_id`, and the service will callback with absolute download URLs.

## Testing

Run unit tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Configuration (essentials)

| Variable | Default | Description |
|----------|---------|-------------|
| `API_TOKEN` | - | Bearer token for auth (set to lock down endpoints) |
| `BASE_URL` | empty | Public HTTPS host used to build absolute download links |
| `SALESFORCE_CALLBACK_URL` | empty | Where to POST async results when `matterId` is present |
| `SPLIT_THRESHOLD_MB` | 25 | Max size per output file after splitting |
| `FILE_RETENTION_SECONDS` | 86400 | Auto-delete files after this many seconds |
| `UPLOAD_FOLDER` | empty | Absolute path for storing PDFs; on Azure use a persistent path under `/home` |
| `SYNC_TIMEOUT_SECONDS` | 540 | Timeout for `/compress-sync` work before 504 |
| `ASYNC_MAX_MB` | 450 | Max PDF size allowed for async jobs (downloaded via signed URL) |
| `DISABLE_ASYNC_WORKERS` | unset | Set to `1` to disable background workers |

## Deployment notes

- Azure App Service: install Ghostscript, run gunicorn (see `startup.sh`).
- Use a persistent upload folder (`UPLOAD_FOLDER` under `/home`) or a shared store; `/tmp` is not persisted across restarts.
- For multiple instances, store outputs in shared storage (Azure Files/Blob) or keep a single instance.
- Ensure `BASE_URL` and `SALESFORCE_CALLBACK_URL` are set; restart after config changes.
