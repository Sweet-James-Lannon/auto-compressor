# SJ PDF Compressor (Sweet James)

Plain-English summary: this service takes a PDF, compresses it with Ghostscript,
and splits it into email-safe parts when needed. It exposes a small HTTP API for
Salesforce and a simple dashboard for manual testing.

---

## 1) What this repo does

- Accepts PDFs from a URL, an upload, or base64.
- Compresses them with Ghostscript (fast + tuned for scanned legal docs).
- Splits output to <= 25MB parts when the output is too big for email/Salesforce.
- Returns download links and (for Salesforce) can callback asynchronously.
- Uses job IDs and a cleanup thread so files are temporary and safe.

---

## 2) How to run it

### Local (dev)

- Python 3.11
- Ghostscript installed (gs)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Azure App Service (prod)

- Uses `startup.sh` and gunicorn
- Requires Ghostscript installed in the container

---

## 3) Main API endpoints (what to call)

### POST /compress (async)

- Input: JSON with one of:
  - `file_download_link` (URL)
  - `file_content_base64`
- Optional: `name`, `pages`, `password`, `callbackUrl`
- Returns: `job_id`
- Poll: `GET /status/<job_id>`

### POST /compress-sync (blocking)

- Input: JSON with `file_download_link` or a multipart upload
- Optional: `name`
- Used by: dashboard/manual testing and a few legacy callers that expect a single request/response
- Not used by: Salesforce long-running flows (timeouts). For Salesforce, use async with `matterId`.
- If `matterId` is present, it switches to async callback flow (returns 202)
- This endpoint is retained for backward compatibility and local testing

### GET /download/<filename>

- Serves the file
- `?name=...` controls the download filename shown to the user

### GET /status/<job_id>

- Returns current job state and links when finished

### POST /job/check

- Lightweight PDF.co-style status endpoint

### GET /diagnose/<job_id>

- Diagnostic report for completed jobs

---

## 4) Naming behavior (important for user downloads)

- The internal filenames are job-id based (safe for storage).
- The download filename shown to users is based on the original input name:
  - `OriginalName_compressed.pdf`
  - `OriginalName_compressed_part1.pdf`, `..._part2.pdf`
- Source order for name:
  1) request field `name`
  2) upload filename
  3) URL filename
  4) fallback `document`

---

## 5) Compression logic (simple version)

1) Validate PDF header
2) Decide compression mode (lossless/aggressive/adaptive)
3) Choose path:
   - **Serial** for smaller files
   - **Parallel** for large files
4) Compress with Ghostscript
5) If output > SPLIT_TRIGGER_MB, split into parts <= SPLIT_THRESHOLD_MB
6) Verify outputs, return links

### Ghostscript specifics

- Aggressive mode uses `/screen` plus JPEG encoding for maximum size reduction
- Lossless mode uses `/default` and de-duplication only
- Parallel mode splits by pages, compresses chunks concurrently, then splits
  each chunk to <= threshold

---

## 6) File map (where the logic lives)

- `app.py`:
  - Flask API, request parsing, async queue, callbacks, download naming
- `compress.py`:
  - Top-level compression flow and serial/parallel selection
- `compress_ghostscript.py`:
  - Ghostscript command building + parallel chunk compression
- `split_pdf.py`:
  - Splitting logic (binary search + size-based)
- `job_queue.py`:
  - Background workers and job state
- `utils.py`:
  - Safe download, CPU detection, helpers
- `pdf_diagnostics.py`:
  - Diagnostics + quality warnings
- `exceptions.py`:
  - Typed errors for clean responses

---

## 7) Salesforce flow (how it’s used in production)

- Salesforce calls `POST /compress-sync` with:
  - `file_download_link`
  - `matterId`
- Service responds `202 + job_id` quickly
- Worker downloads, compresses, splits, then POSTs results to
  `SALESFORCE_CALLBACK_URL`
- Callback includes absolute download links built from `BASE_URL`

---

## 8) Configuration (the important env vars)

These are the same settings you see in Azure App Settings.

### Core

| Variable | Meaning |
|----------|---------|
| API_TOKEN | Bearer token for auth |
| BASE_URL | Public HTTPS base used to build absolute links |
| SALESFORCE_CALLBACK_URL | Callback target for async flow |
| COMPRESSION_MODE | aggressive / lossless / adaptive |
| ALLOW_LOSSY_COMPRESSION | 1/0 allow downsampling |
| SPLIT_THRESHOLD_MB | Max size per part (default 25) |
| SPLIT_TRIGGER_MB | Only split when output exceeds this |
| FILE_RETENTION_SECONDS | How long files stay before cleanup |

### Parallel/Chunking

| Variable | Meaning |
|----------|---------|
| TARGET_CHUNK_MB | Target chunk size |
| MAX_CHUNK_MB | Max chunk size before re-splitting |
| MAX_PARALLEL_CHUNKS | Upper bound on chunk count |
| MAX_PAGES_PER_CHUNK | Advisory page cap |
| PARALLEL_MAX_WORKERS | Cap parallel workers |
| GS_NUM_RENDERING_THREADS | Threads per Ghostscript process |

### Ghostscript tuning

| Variable | Meaning |
|----------|---------|
| GS_FAST_WEB_VIEW | PDF linearization on/off |
| GS_BAND_HEIGHT | Banding height |
| GS_BAND_BUFFER_SPACE_MB | Band buffer size |
| GS_COLOR_DOWNSAMPLE_TYPE | /Subsample /Average /Bicubic |
| GS_GRAY_DOWNSAMPLE_TYPE | /Subsample /Average /Bicubic |

### Split tuning

| Variable | Meaning |
|----------|---------|
| SPLIT_SAFETY_BUFFER_MB | Buffer under the limit |
| SPLIT_OPTIMIZE_MAX_OVERAGE_MB | Skip optimization if too far over |
| SPLIT_MINIMIZE_PARTS | Try extra pass to reduce part count |
| SPLIT_ULTRA_JPEGQ | JPEG quality for extra pass |
| SPLIT_ULTRA_GAP_PCT | Only run extra pass if close to dropping a part |

---

## 9) Quick dev test

```bash
curl -X POST /compress \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"file_download_link":"https://source/file.pdf","name":"Case_File.pdf"}'
```

Then poll:

```bash
GET /status/<job_id>
```

| `API_TOKEN` | - | Bearer token for auth (set to lock down endpoints) |
---

## 10) Known tradeoffs (intentional)

- Parallel mode may produce more parts than serial. This is expected and faster.
- Large files prioritize size-based chunking for throughput.
- Internal filenames are job-based; user-facing names are set at download time.
- These tradeoffs are intentional and tied to our real user inputs: large legal PDFs
  that must be compressed quickly and safely within email/Salesforce size limits.
  Any change to compression, chunking, or splitting should be evaluated against
  real-world PDF sizes and user workflows before shipping.
- Sync does not change compression logic. It uses the same compress/split pipeline
  as async; the only sync-only behavior is size gating, timeouts, and auto-async
  handoff for large files.

---

## 11) Deployment notes (Azure)

- App Service Linux + Python 3.11
- Install Ghostscript in startup
- Use `/home` storage for uploads (persistent)
- Set `BASE_URL` and `SALESFORCE_CALLBACK_URL`
