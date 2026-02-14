# auto-compressor (internal)

Reliable PDF compression API for Salesforce/LWC.

## What This Service Does
- Accepts PDF input (URL, upload, or base64).
- Processes jobs asynchronously.
- Compresses with Ghostscript.
- Optionally splits output into delivery-sized parts.
- Returns stable status + download links via `/status/<job_id>`.

## Current Compression Model (Reliability Cutover)
The engine now uses a simple, deterministic flow.

### 1) Two routing paths only
- **Serial path**: files `<= SERIAL_THRESHOLD_MB` (default `50`).
- **Parallel path**: files `> SERIAL_THRESHOLD_MB`.

### 2) Profile-driven execution
`pdf_diagnostics.estimate_processing_profile(...)` estimates:
- `tier`
- `sec_per_page`
- `max_pages_per_chunk`
- `chunk_timeout_sec`
- `recommended_sla_sec`
- `dominant_codec`

This profile feeds chunk sizing and time budgets.

### 3) No timeout cascade
Removed old probe/rebalance/extreme retry chains.
Parallel compression is linear:
1. split chunks
2. compress each chunk once
3. keep original chunk if compressed chunk is larger
4. merge
5. if merged output is worse, return original (or split original if split requested)

### 4) Size-safety guarantee
- Compressed output is never intentionally returned worse than input.
- Split delivery prioritizes honoring requested split thresholds, with tightening/binary fallback when needed.

## What Is and Is Not Guaranteed
### Guaranteed
- Job completes with either:
  - meaningful compression, or
  - honest original return (`compression_method: "none"`).
- No long fallback loop explosion.
- API endpoints and response shape stay compatible.

### Not guaranteed
- Every PDF gets a large reduction.
- PDFs already highly optimized/scanned in certain codecs may get little/no gain.

## API (unchanged)
- `POST /compress-async` (alias: `/compress-sync`)
- `GET /status/<job_id>`
- `GET /download/<filename>`
- `GET /health`
- `GET /diagnose/<job_id>`

## Important Defaults
- `SERIAL_THRESHOLD_MB=50`
- `PARALLEL_JOB_SLA_SEC=300`
- `PARALLEL_JOB_SLA_MAX_SEC=1800`
- `TARGET_CHUNK_MB=30`
- `MAX_CHUNK_MB=50`
- `MAX_PAGES_PER_CHUNK=600`
- `SPLIT_THRESHOLD_MB=25` (request-level override still supported)
- Split controls support validated env overrides:
  - `SPLIT_MINIMIZE_PARTS` (default `False`)
  - `SPLIT_ADAPTIVE_MAX_ATTEMPTS` (default `3`)
  - `SPLIT_ENABLE_BINARY_FALLBACK` (default `False`)
- Request split threshold clamp in app: `5MB..50MB`

## Split Behavior
- Split is **opt-in**.
- Request fields: `split_threshold_mb` or `splitSizeMB`.
- Effective threshold is clamped to `5..50 MB`.
- Ghostscript page-range splitting is preferred; fallback splitters remain available for edge cases.

## Benchmarking
Use `scripts/benchmark_compress.py`.

Example:
```bash
python3 scripts/benchmark_compress.py uploads/your_input.pdf \
  --working-dir ./benchmarks \
  --split-threshold-mb 30 --split-trigger-mb 30 \
  --verify-pages
```

Useful flags:
- `--cleanup` deletes benchmark outputs after each run.
- `--json-out` and `--csv-out` write summaries.

Note:
- `benchmarks/` is git-ignored.

## Pre-Push Validation (Recommended)
Before merging to `main`, run:
```bash
python3 -m py_compile app.py $(find pdf_compressor -name '*.py')
python3 -m pytest -q
```
Then benchmark your real PDFs (same files you use in production) and compare:
- wall time
- reduction percent
- part counts
- page match

## Code Map
- `app.py` - compatibility entrypoint (`gunicorn app:app`)
- `pdf_compressor/factory.py` - Flask app factory + blueprint registration
- `pdf_compressor/bootstrap.py` - one-time startup of cleanup daemon + async workers
- `pdf_compressor/routes/web_routes.py` - `/`, `/health`, `/favicon.ico`
- `pdf_compressor/routes/api_routes.py` - API endpoints (`/compress-async`, `/status/<job_id>`, etc.)
- `pdf_compressor/services/compression_service.py` - request handlers + orchestration helpers
- `pdf_compressor/services/status_service.py` - status/check/diagnose wrappers
- `pdf_compressor/services/file_service.py` - file tracking/cleanup wrappers
- `pdf_compressor/workers/job_queue.py` - async queue + worker threads
- `pdf_compressor/engine/compress.py` - serial/parallel routing and result shaping
- `pdf_compressor/engine/ghostscript.py` - Ghostscript compression + parallel pipeline
- `pdf_compressor/engine/split.py` - split/merge utilities for delivery constraints
- `pdf_compressor/engine/pdf_diagnostics.py` - fingerprinting + processing profile estimation
- `pdf_compressor/core/utils.py` - shared helpers + download safeguards
- `pdf_compressor/core/settings.py` - runtime setting resolution and guardrails
- `pdf_compressor/core/exceptions.py` - typed exceptions across layers
- `scripts/benchmark_compress.py` - reproducible benchmark harness

## Package Layout
```text
pdf_compressor/
  factory.py
  config.py
  bootstrap.py
  routes/
  services/
  workers/
  engine/
  core/
  templates/
```

Note:
- Root modules like `compress.py` and `split_pdf.py` are temporary compatibility wrappers that re-export from `pdf_compressor/*`.
