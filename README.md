# auto-compressor (internal)

## PDF Compressor - Simple explanation
This is an in-house PDF compression API for Salesforce. A user selects a file from Docrio in a
Lightning Web Component (LWC), optionally sets a split size, and the API compresses it. If
splitting is enabled, the API slices the output into parts under the requested size.

Requests are always async. The API returns a job ID immediately so the LWC can show progress
and poll `/status/<job_id>` for results.

## Main files (the cast)
```text
+--------------------------+--------------------+------------------------------------------------------+
| File                     | Role               | What it does                                         |
+--------------------------+--------------------+------------------------------------------------------+
| app.py                   | Front desk         | HTTP API, request parsing, job queue, callbacks, UI  |
| job_queue.py             | Waiting room       | Holds jobs until workers are available               |
| compress.py              | Traffic controller | Chooses serial vs parallel compression paths         |
| compress_ghostscript.py  | Compressor         | Runs Ghostscript and parallel chunk compression      |
| split_pdf.py             | Paper cutter       | Splits PDFs into parts for delivery constraints      |
| utils.py                 | Toolbox            | Shared helpers (SSRF checks, CPU detection, sizes)   |
| exceptions.py            | Error translator   | Friendly error types for callers                     |
| pdf_diagnostics.py       | Quality inspector  | Detects quality risks and warnings                   |
+--------------------------+--------------------+------------------------------------------------------+
```

## How it works (step by step)
1) User submits a Docrio file URL (or upload/base64).
2) API returns a job ID immediately.
3) Job goes into the queue; workers pick it up when free.
4) `compress.py` picks the compression path (serial or parallel).
5) Ghostscript compresses the PDF.
6) If a split size was requested, `split_pdf.py` slices the output into parts.
7) The LWC polls `/status/<job_id>` and gets download links + quality warnings.

```text
[User] -> [API] -> [Job queue] -> [compress.py] -> [Ghostscript] -> [split_pdf] -> [Links]
```

## Compression decision logic (current defaults)
- < 1 MB: skip compression (already tiny).
- 1 to ~200 MB: serial Ghostscript unless page-heavy rule forces parallel.
- >= 200 MB (no split) and estimated chunks > 3: parallel compression for throughput.
- Split requests: size-based threshold still applies but prefer serial first when feasible.
- Page heavy: >= 600 pages and meets `PARALLEL_PAGE_MIN_MB` (default 0) routes to parallel to avoid serial timeouts.
- If serial times out and parallel is allowed, we fall back to parallel.

## Parallel strategy (under the hood)
- Splits the input into page-based chunks near `TARGET_CHUNK_MB` (default 40 MB; 60 MB for 200 MB+ aggressive files).
- Caps chunk size at `MAX_CHUNK_MB` (default 60 MB / 90 MB for large) and chunk count at `SLA_MAX_PARALLEL_CHUNKS` (default 5 for aggressive paths).
- Caps pages per chunk at `MAX_PAGES_PER_CHUNK` (default 200; env can raise).
- Pre-dedupe is **skipped in aggressive mode** unless split inflation exceeds 8% (then forced).
- Micro-probe: a 3-page sample compression predicts inflation/throughput; if delta >5% or >12s, we avoid parallel unless page-count forces it.
- Probe (parallel): run the smallest chunk first; if it inflates >5% or takes >12s, bail early (after an optional quick lossless attempt) to protect SLA/quality.
- Chunks are merged with Ghostscript (45s timeout). PyPDF2 fallback has a 60s timeout; bloat is logged and caught downstream.
- If chunked output is >8% larger than input and splitting is enabled, a merge + re-split pass runs to dedupe.
- If parallel output is larger than input, we return the original (or split original if splitting was requested).

## Ghostscript defaults (aggressive mode)
- `/screen` preset plus explicit downsampling for size.
- Quality floors: Color ≥150 DPI, Gray ≥200 DPI, Mono ≥300 DPI (env values are floored to these mins).
- JPEG re-encode; JPEG2000 is converted to JPEG.
- Duplicate images are detected and deduped; fonts are subset.
- Fast web view enabled by default.

Lossless mode keeps images and focuses on deduplication only.

## Splitting behavior
- Splitting happens only when `split_threshold_mb` (or `splitSizeMB`) is provided.
- If `split_trigger_mb` is set, splitting only occurs when output exceeds the trigger.
- The splitter uses a size-based binary search to minimize part count.
- Very large, page-heavy PDFs first try a fast page split (+1 part) to avoid slow binary search.
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

### Response fields of interest
- `quality_mode`: `aggressive_150dpi` or `lossless`.
- `analysis`: `{ page_count, bytes_per_page, probe: {pages, delta_pct, elapsed, success}, probe_bad }`.
- `probe_bailout`: true when we returned the original because a quick safety test showed no safe win; `probe_bailout_reason` is a short human string.
- `lossless_fallback_used`: true when a fast, safe lossless attempt improved size after a probe bailout.

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
- `MAX_CHUNK_MB` (default 60; 90 for large aggressive)
- `MAX_PARALLEL_CHUNKS` (default 16), `SLA_MAX_PARALLEL_CHUNKS` (default 5 for aggressive paths)
- `MAX_PAGES_PER_CHUNK` (default 200)
- (Fixed) `PROBE_TIME_BUDGET_SEC` = 12s, `PROBE_INFLATION_ABORT_PCT` = 0.05, `PROBE_SAMPLE_PAGES` = 3
- (Fixed) `PARALLEL_JOB_SLA_SEC` = 300s, `SLA_MAX_PARALLEL_CHUNKS` = 5
- (Fixed) `MERGE_TIMEOUT_SEC` = 45s, `MERGE_FALLBACK_TIMEOUT_SEC` = 60s
- (Fixed) `MERGE_BLOAT_ABORT_PCT` = 0.02
- `GS_COLOR_IMAGE_RESOLUTION` (floored at 150)
- `GS_GRAY_IMAGE_RESOLUTION` (floored at 200)
- `GS_MONO_IMAGE_RESOLUTION` (floored at 300)
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
- Time/SLA guardrails (fixed):
  - Parallel job SLA: 300s (hard stop for post-processing).
  - Probe bailout: >5% inflation or >12s runtime returns the original.
  - Per-chunk Ghostscript budget: 40–90s (size-scaled, clamped).
  - Merge timeout: 45s for Ghostscript; PyPDF2 fallback capped at 60s.
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
