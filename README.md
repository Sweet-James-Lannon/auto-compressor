# SJ PDF Compressor

Compresses PDFs and auto-splits into ≤25MB parts for email/Salesforce when output exceeds the split trigger. Parallel Ghostscript is used for large files; sync limit is 300MB.

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
  - If output exceeds `SPLIT_TRIGGER_MB`, auto-splits into ≤ `SPLIT_THRESHOLD_MB` parts and logs part sizes.
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

By default the service uses aggressive compression (downsampling) to minimize
output size. Set `ALLOW_LOSSY_COMPRESSION=0` and `COMPRESSION_MODE=lossless` to
preserve original image quality.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_TOKEN` | - | Bearer token for auth (set to lock down endpoints) |
| `BASE_URL` | empty | Public HTTPS host used to build absolute download links |
| `SALESFORCE_CALLBACK_URL` | empty | Where to POST async results when `matterId` is present |
| `SPLIT_THRESHOLD_MB` | 25 | Max size per output file after splitting |
| `SPLIT_TRIGGER_MB` | 30 | Split only when output exceeds this size (parts still use `SPLIT_THRESHOLD_MB`) |
| `SPLIT_SAFETY_BUFFER_MB` | 0 | Buffer to keep split parts under the limit (email overhead) |
| `COMPRESSION_MODE` | aggressive | `lossless`, `aggressive`, or `adaptive` (quality vs size tradeoff) |
| `ALLOW_LOSSY_COMPRESSION` | 1 | Set to `1` to allow downsampling/quality-reducing compression |
| `SCANNED_CONFIDENCE_FOR_AGGRESSIVE` | 70 | Min scanned-doc confidence for adaptive aggressive mode |
| `TARGET_CHUNK_MB` | 40 | Target chunk size for parallel compression |
| `MAX_CHUNK_MB` | 60 | Max chunk size before re-splitting (prevents long-running chunks) |
| `MAX_PARALLEL_CHUNKS` | 16 | Upper bound on chunk count per file |
| `MAX_PAGES_PER_CHUNK` | 200 | Cap pages per chunk to avoid very large per-chunk workloads |
| `GS_NUM_RENDERING_THREADS` | unset | Override Ghostscript rendering threads (serial/optimize passes) |
| `SPLIT_OPTIMIZE_MAX_OVERAGE_MB` | 1.0 | Skip optimizing page-split parts that exceed the limit by more than this |
| `FILE_RETENTION_SECONDS` | 86400 | Auto-delete files after this many seconds |
| `MIN_FILE_RETENTION_SECONDS` | 3600 | Minimum retention enforced even if `FILE_RETENTION_SECONDS` is lower |
| `UPLOAD_FOLDER` | empty | Absolute path for storing PDFs; on Azure use a persistent path under `/home` |
| `SYNC_TIMEOUT_SECONDS` | 540 | Timeout for `/compress-sync` work before 504 |
| `SYNC_AUTO_ASYNC_MB` | 120 | Auto-queue sync requests above this size and return `202` + job_id |
| `ASYNC_MAX_MB` | 450 | Max PDF size allowed for async jobs (downloaded via signed URL) |
| `DISABLE_ASYNC_WORKERS` | unset | Set to `1` to disable background workers |

## Ghostscript CPU usage (notes)

Based on the Ghostscript docs (https://www.ghostscript.com/doc/Use.html and
https://www.ghostscript.com/doc/Language.html):

- `-dNumRenderingThreads` can use multiple CPU cores when banding/clist rendering;
  docs recommend setting it to the number of available cores for best throughput.
- `BandHeight` controls band size, and `BandBufferSpace` limits band buffer memory.
  The docs note that if you only want to allocate more memory for banding, use
  `BufferSpace` instead of `BandBufferSpace`.
- Some vector devices (including `pdfwrite`) only write output on exit; changing
  `OutputFile` mid-run flushes the pages received so far and then starts a new file.

In this service:
- We run multiple Ghostscript processes in parallel for large PDFs, so per-process
  rendering threads are scaled down to avoid oversubscribing CPU cores.
- You can override per-process thread count with `GS_NUM_RENDERING_THREADS` if needed.

## Deployment notes

- Azure App Service: install Ghostscript, run gunicorn (see `startup.sh`).
- Use a persistent upload folder (`UPLOAD_FOLDER` under `/home`) or a shared store; `/tmp` is not persisted across restarts.
- For multiple instances, store outputs in shared storage (Azure Files/Blob) or keep a single instance.
- Ensure `BASE_URL` and `SALESFORCE_CALLBACK_URL` are set; restart after config changes.
