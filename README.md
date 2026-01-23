# auto-compressor (internal)

## PDF Compressor - Simple explanation
This is an in-house PDF compression API for Salesforce. A user selects a file from Docrio in a
Lightning Web Component (LWC), optionally sets a split size, and the API compresses it. If
splitting is enabled, the API slices the output into parts under the requested size.

Requests are always async. The API returns a job ID immediately so the LWC can show progress
and poll `/status/<job_id>` for results.

## Main files (the cast)
| File | Role | What it does |
| --- | --- | --- |
| `app.py` | Front desk | HTTP API, request parsing, job queue, callbacks, dashboard |
| `job_queue.py` | Waiting room | Holds jobs until workers are available |
| `compress.py` | Traffic controller | Chooses serial vs parallel compression paths |
| `compress_ghostscript.py` | Compressor | Runs Ghostscript and parallel chunk compression |
| `split_pdf.py` | Paper cutter | Splits PDFs into parts for delivery constraints |
| `utils.py` | Toolbox | Shared helpers (SSRF checks, CPU detection, sizes) |
| `exceptions.py` | Error translator | Friendly error types for callers |
| `pdf_diagnostics.py` | Quality inspector | Detects quality risks and warnings |

## How it works (step by step)
1) User submits a Docrio file URL (or upload/base64).
2) API returns a job ID immediately.
3) Job goes into the queue; workers pick it up when free.
4) `compress.py` picks the compression path (serial or parallel).
5) Ghostscript compresses the PDF.
6) If a split size was requested, `split_pdf.py` slices the output into parts.
7) The LWC polls `/status/<job_id>` and gets download links + quality warnings.

## Compression decision logic (current defaults)
- < 1 MB: skip compression (already tiny).
- 1 to 30 MB: serial Ghostscript.
- 30 to 100 MB: serial Ghostscript (unless page-heavy rule triggers).
- > 100 MB: parallel compression (split into chunks, compress in parallel).
- >= 600 pages: parallel compression even if the file is smaller.

Notes:
- Size-based parallel uses `PARALLEL_THRESHOLD_MB=30` and `PARALLEL_SERIAL_CUTOFF_MB=100`.
- Page-based parallel uses `PARALLEL_PAGE_THRESHOLD=600` and `PARALLEL_PAGE_MIN_MB` (default 0).

## Parallel strategy (under the hood)
- Splits the input into page-based chunks near `TARGET_CHUNK_MB` (default 40 MB).
- Caps chunk size at `MAX_CHUNK_MB` (default 60 MB) and chunk count at `MAX_PARALLEL_CHUNKS` (default 16).
- Caps pages per chunk at `MAX_PAGES_PER_CHUNK` (default 200).
- Files >= 200 MB use larger chunks (60 MB target, 90 MB max) and fewer chunks (max 12) unless env overrides.
- Chunks under 20 MB are skipped to avoid overhead.
- Chunks are merged with Ghostscript to dedupe resources.
- If chunked output is >8% larger than input and splitting is enabled, a merge + re-split pass runs to dedupe.
- If parallel output is larger than input and splitting is enabled, the system falls back to splitting the original.

## Ghostscript defaults (aggressive mode)
- `/screen` preset plus explicit downsampling for size.
- Color DPI: 50, Gray DPI: 50, Mono DPI: 100.
- JPEG re-encode; JPEG2000 is converted to JPEG.
- Duplicate images are detected and deduped; fonts are subset.
- Fast web view enabled by default.

Lossless mode keeps images and focuses on deduplication only.

## Splitting behavior
- Splitting happens only when `split_threshold_mb` (or `splitSizeMB`) is provided.
- If `split_trigger_mb` is set, splitting only occurs when output exceeds the trigger.
- The splitter uses a size-based binary search to minimize part count.
- If the output is close to fitting into fewer parts (default 12% gap), it can try an
  ultra pass (JPEG quality 50) to reduce parts when lossy is allowed.

## API summary
- `POST /compress-async` (or `/compress-sync` alias)
  - Accepts JSON: `file_download_link`, `file_content_base64`, or `file_url`.
  - Accepts multipart: `pdf` or `file`.
  - Optional: `split_threshold_mb` or `splitSizeMB`, `pages`, `password`, `callbackUrl`.
- `GET /status/<job_id>`
- `GET /download/<filename>`
- `GET /health`
- `GET /diagnose/<job_id>`

## Key config settings
- `SPLIT_THRESHOLD_MB` (default 25)
- `SPLIT_TRIGGER_MB` (default = threshold)
- `SPLIT_SAFETY_BUFFER_MB` (default 0)
- `SPLIT_ULTRA_JPEGQ` (default 50)
- `SPLIT_ULTRA_GAP_PCT` (default 0.12)
- `COMPRESSION_MODE` (aggressive, lossless, adaptive)
- `ALLOW_LOSSY_COMPRESSION` (1/0)
- `PARALLEL_THRESHOLD_MB` (default 30)
- `PARALLEL_SERIAL_CUTOFF_MB` (default 100)
- `PARALLEL_PAGE_THRESHOLD` (default 600)
- `PARALLEL_PAGE_MIN_MB` (default 0)
- `PARALLEL_MAX_WORKERS` (default auto)
- `TARGET_CHUNK_MB` (default 40)
- `MAX_CHUNK_MB` (default 60)
- `MAX_PARALLEL_CHUNKS` (default 16)
- `MAX_PAGES_PER_CHUNK` (default 200)
- `GS_COLOR_IMAGE_RESOLUTION` (default 50)
- `GS_GRAY_IMAGE_RESOLUTION` (default 50)
- `GS_MONO_IMAGE_RESOLUTION` (default 100)
- `ASYNC_WORKERS` (default min(2, cpu))
- `FILE_RETENTION_SECONDS` (default 86400, minimum 3600)
- `ASYNC_MAX_MB` (default 450, hard cap 600)
- `MAX_QUEUE_SIZE` (default 50)

## Safety and limits
- SSRF protection blocks private IPs and cloud metadata endpoints.
- Upload limit: 500 MB (Flask `MAX_CONTENT_LENGTH`).
- Async download limit: `ASYNC_MAX_MB` (default 450 MB, capped at 600 MB).
- Queue size cap: 50 jobs by default.
- File cleanup: automatic deletion after retention window (min 1 hour).
- Ghostscript timeouts:
  - Aggressive mode: ~10 seconds per MB, minimum 600 seconds.
  - Lossless mode: ~5 seconds per MB, minimum 300 seconds.
- Quality warnings are included when `ENABLE_QUALITY_WARNINGS=1`.

## Error messages (friendly translations)
- `EncryptionError`: PDF is password-protected or locked.
- `StructureError`: PDF is damaged or malformed.
- `MetadataCorruptionError`: PDF has corrupted internal data.
- `SplitError`: Cannot split into small enough parts.
- `DownloadError`: Could not fetch the PDF from the provided URL.

## Run locally
- Python 3.11
- Ghostscript installed (`gs` on PATH)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Benchmark script
Use `scripts/benchmark_compress.py` to measure wall time, CPU time, throughput, and optional page-count checks.

Example (single file, no split):
```bash
python scripts/benchmark_compress.py /path/to/file.pdf --working-dir ./benchmarks --verify-pages
```

Example (force split + write summaries):
```bash
python scripts/benchmark_compress.py /path/to/pdfs --recursive \
  --split-threshold-mb 25 --split-trigger-mb 25 \
  --working-dir ./benchmarks --verify-pages \
  --json-out bench.json --csv-out bench.csv
```

Notes:
- Omit `--split-threshold-mb` to keep a single output and avoid chunk/merge overhead.
- Use `--cleanup` to delete outputs after each run.
