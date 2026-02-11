"""PDF splitting module for large file handling.

This module provides functionality to split PDFs that exceed
the size threshold into smaller parts for delivery constraints.
"""

import logging
import math
import multiprocessing
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from PyPDF2.errors import PdfReadError

from compress_ghostscript import (
    optimize_split_part,
    compress_pdf_with_ghostscript,
    compress_ultra_aggressive,
    get_ghostscript_command,
)
from exceptions import EncryptionError, StructureError, SplitError
from settings import get_split_runtime_settings
from utils import get_file_size_mb, env_int

# Constants - Size threshold defaults
# Default kept for backward compatibility; override via SPLIT_THRESHOLD_MB.
DEFAULT_THRESHOLD_MB: float = 25.0
_safety_buffer = float(os.environ.get("SPLIT_SAFETY_BUFFER_MB", "0"))
SAFETY_BUFFER_MB: float = max(0.0, _safety_buffer)
ALLOW_LOSSY_COMPRESSION: bool = os.environ.get("ALLOW_LOSSY_COMPRESSION", "1").lower() in ("1", "true", "yes")
_opt_overage = float(os.environ.get("SPLIT_OPTIMIZE_MAX_OVERAGE_MB", "1.0"))
SPLIT_OPTIMIZE_MAX_OVERAGE_MB: float = max(0.0, _opt_overage)
_split_settings = get_split_runtime_settings()
SPLIT_MINIMIZE_PARTS: bool = _split_settings.split_minimize_parts
SPLIT_ENABLE_BINARY_FALLBACK: bool = _split_settings.split_enable_binary_fallback
SPLIT_ADAPTIVE_MAX_ATTEMPTS: int = _split_settings.split_adaptive_max_attempts
SPLIT_ULTRA_JPEGQ: int = _split_settings.split_ultra_jpegq
MERGE_TIMEOUT_SEC = env_int("MERGE_TIMEOUT_SEC", 120)
MERGE_FALLBACK_TIMEOUT_SEC = env_int("MERGE_FALLBACK_TIMEOUT_SEC", 120)
MERGE_TIMEOUT_SEC_PER_MB = 0.8
MERGE_TIMEOUT_MAX_SEC = 600
MERGE_FALLBACK_TIMEOUT_MAX_SEC = 600
MERGE_BLOAT_ABORT_PCT = 0.02
SPLIT_ULTRA_GAP_PCT: float = _split_settings.split_ultra_gap_pct
FAST_SPLIT_ENABLED = True
FAST_SPLIT_MIN_MB = 120.0
FAST_SPLIT_MIN_PAGES = 2000
FAST_SPLIT_EXTRA_PARTS = 1
FAST_SPLIT_MAX_PARTS = 20

logger = logging.getLogger(__name__)


# =============================================================================
# NEW HELPER FUNCTIONS FOR PARALLEL COMPRESSION
# =============================================================================

def _cleanup_paths(paths: List[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _merge_with_pypdf2_worker(paths: List[str], out_path: str, result_queue) -> None:
    try:
        merger = PdfMerger()
        for path in paths:
            merger.append(path)
        merger.write(out_path)
        merger.close()
        result_queue.put({"ok": True})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def _rename_split_parts(parts: List[Path], base_name: str, output_dir: Path) -> List[Path]:
    renamed: List[Path] = []
    for idx, part in enumerate(parts, 1):
        dest = output_dir / f"{base_name}_part{idx}.pdf"
        if part.resolve() == dest.resolve():
            renamed.append(dest)
            continue
        dest.unlink(missing_ok=True)
        part.rename(dest)
        renamed.append(dest)
    return renamed


def _should_try_ultra(file_size_mb: float, threshold_mb: float, part_count: int, gap_pct: float) -> bool:
    if part_count <= 1:
        return False
    target_size = threshold_mb * (part_count - 1)
    if target_size <= 0:
        return False
    gap = max(0.0, file_size_mb - target_size)
    return (gap / max(file_size_mb, 0.1)) <= gap_pct


def _should_use_fast_split(file_size_mb: float, total_pages: int) -> bool:
    if not FAST_SPLIT_ENABLED:
        return False
    return file_size_mb >= FAST_SPLIT_MIN_MB and total_pages >= FAST_SPLIT_MIN_PAGES


def _fast_split_by_pages(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float,
    num_parts: int,
    total_pages: int,
    skip_optimization_under_threshold: bool,
) -> Optional[List[Path]]:
    extra_parts = FAST_SPLIT_EXTRA_PARTS
    desired_parts = max(2, num_parts + extra_parts)
    desired_parts = min(desired_parts, total_pages)
    if FAST_SPLIT_MAX_PARTS > 0:
        desired_parts = min(desired_parts, FAST_SPLIT_MAX_PARTS)
    if desired_parts <= 1:
        return None

    logger.info(
        "[split_by_size] Fast split fallback: parts=%s (base=%s extra=%s)",
        desired_parts,
        num_parts,
        extra_parts,
    )

    parts = split_by_pages(pdf_path, output_dir, desired_parts, base_name)
    final_parts: List[Path] = []
    oversize = False

    for idx, part in enumerate(parts, start=1):
        part_size = get_file_size_mb(part)
        dest = output_dir / f"{base_name}_part{idx}.pdf"
        dest.unlink(missing_ok=True)

        if part_size <= threshold_mb and skip_optimization_under_threshold:
            part.rename(dest)
            final_parts.append(dest)
        else:
            success, opt_message = optimize_split_part(part, dest)
            if success and dest.exists():
                optimized_size = get_file_size_mb(dest)
                if optimized_size > part_size:
                    logger.warning(
                        "[split_by_size] Fast split optimization increased size for part %s (%.2fMB -> %.2fMB); using raw",
                        idx,
                        part_size,
                        optimized_size,
                    )
                    dest.unlink(missing_ok=True)
                    part.rename(dest)
                else:
                    part.unlink(missing_ok=True)
            else:
                logger.warning(
                    "[split_by_size] Fast split optimization failed for part %s: %s; using raw",
                    idx,
                    opt_message,
                )
                dest.unlink(missing_ok=True)
                part.rename(dest)
            final_parts.append(dest)

        final_size = get_file_size_mb(dest)
        if final_size > threshold_mb:
            oversize = True

    if oversize:
        logger.warning("[split_by_size] Fast split produced oversize parts; falling back to size-based splitter...")
        _cleanup_paths(final_parts)
        return None

    return final_parts


def _maybe_fast_split(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float,
    num_parts: int,
    total_pages: int,
    skip_optimization_under_threshold: bool,
    reason: str,
) -> Optional[List[Path]]:
    if not _should_use_fast_split(get_file_size_mb(pdf_path), total_pages):
        return None

    fast_parts = _fast_split_by_pages(
        pdf_path,
        output_dir,
        base_name,
        threshold_mb,
        num_parts,
        total_pages,
        skip_optimization_under_threshold,
    )
    if fast_parts:
        logger.info("[split_by_size] Fast split succeeded (%s)", reason)
        return fast_parts

    logger.warning("[split_by_size] Fast split failed (%s); falling back to size-based splitter...", reason)
    return None


def gs_split_by_pages(pdf_path: Path, output_dir: Path, num_parts: int, base_name: str) -> Optional[List[Path]]:
    """Split PDF by page ranges using Ghostscript.

    Ghostscript page-range extraction avoids resource duplication seen in
    PyPDF2 page-copy splits on some documents.
    """
    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        return None

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        reader = PdfReader(pdf_path, strict=False)
        total_pages = len(reader.pages)
    except Exception as exc:
        logger.warning("[gs_split_by_pages] Could not read pages for %s: %s", pdf_path.name, exc)
        return None

    if total_pages == 0:
        raise StructureError.for_file(pdf_path.name, "PDF has no pages")

    num_parts = max(1, min(num_parts, total_pages))
    pages_per_part = total_pages // num_parts
    input_mb = get_file_size_mb(pdf_path)
    timeout = max(60, min(600, int(max(1.0, input_mb) * 2)))

    created: List[Path] = []
    for i in range(num_parts):
        start = i * pages_per_part + 1
        end = total_pages if i == num_parts - 1 else (i + 1) * pages_per_part
        unique_id = str(uuid.uuid4())[:8]
        chunk_path = output_dir / f"{base_name}_chunk{i+1}_{unique_id}.pdf"

        cmd = [
            gs_cmd,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/default",
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            "-dPassThroughJPEGImages=true",
            "-dPassThroughJPXImages=true",
            "-dDownsampleColorImages=false",
            "-dDownsampleGrayImages=false",
            "-dDownsampleMonoImages=false",
            f"-dFirstPage={start}",
            f"-dLastPage={end}",
            f"-sOutputFile={chunk_path}",
            str(pdf_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0 or not chunk_path.exists() or chunk_path.stat().st_size == 0:
                logger.warning(
                    "[gs_split_by_pages] Failed chunk %s/%s pages %s-%s (code=%s), falling back to PyPDF2",
                    i + 1,
                    num_parts,
                    start,
                    end,
                    result.returncode,
                )
                _cleanup_paths(created)
                chunk_path.unlink(missing_ok=True)
                return None
        except Exception as exc:
            logger.warning(
                "[gs_split_by_pages] Exception chunk %s/%s pages %s-%s: %s. Falling back to PyPDF2",
                i + 1,
                num_parts,
                start,
                end,
                exc,
            )
            _cleanup_paths(created)
            chunk_path.unlink(missing_ok=True)
            return None

        created.append(chunk_path)
        logger.info("Created chunk %s/%s: pages %s-%s -> %s (GS)", i + 1, num_parts, start, end, chunk_path.name)

    return created


def split_by_pages(pdf_path: Path, output_dir: Path, num_parts: int, base_name: str) -> List[Path]:
    """Split PDF into N parts by page count. Fast - no Ghostscript.

    Used for parallel compression: split first, compress each chunk in parallel.

    Args:
        pdf_path: Path to source PDF.
        output_dir: Directory for output chunks.
        num_parts: Number of parts to create.
        base_name: Base filename for chunks.

    Returns:
        List of paths to created chunk files.

    Example: 300 pages / 6 parts = 50 pages each
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gs_chunks = gs_split_by_pages(pdf_path, output_dir, num_parts, base_name)
    if gs_chunks:
        return gs_chunks

    reader = PdfReader(pdf_path, strict=False)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise StructureError.for_file(pdf_path.name, "PDF has no pages")

    num_parts = max(1, min(num_parts, total_pages))

    # Calculate pages per part (last part gets remainder)
    pages_per_part = total_pages // num_parts

    chunk_paths = []
    for i in range(num_parts):
        start = i * pages_per_part
        # Last part gets all remaining pages
        end = total_pages if i == num_parts - 1 else (i + 1) * pages_per_part

        writer = PdfWriter()
        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        # Use unique ID to prevent collisions in parallel operations
        unique_id = str(uuid.uuid4())[:8]
        chunk_path = output_dir / f"{base_name}_chunk{i+1}_{unique_id}.pdf"

        with open(chunk_path, 'wb') as f:
            writer.write(f)

        chunk_paths.append(chunk_path)
        logger.info(f"Created chunk {i+1}/{num_parts}: pages {start+1}-{end} -> {chunk_path.name}")

    return chunk_paths


def merge_pdfs(pdf_paths: List[Path], output_path: Path) -> None:
    """Merge multiple PDFs into one using Ghostscript (deduplicates resources).

    Uses Ghostscript to merge PDFs which properly deduplicates shared resources
    like fonts and images. Falls back to PyPDF2 if Ghostscript is not available.

    Args:
        pdf_paths: List of PDF paths to merge (in order).
        output_path: Path for merged output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_total_bytes = sum(p.stat().st_size for p in pdf_paths)
    input_total_mb = input_total_bytes / (1024 * 1024)

    def _scaled_timeout(base_sec: int, total_mb: float, max_sec: int) -> int:
        scaled = int(total_mb * MERGE_TIMEOUT_SEC_PER_MB)
        return max(60, min(max_sec, max(base_sec, scaled)))

    gs_timeout = _scaled_timeout(MERGE_TIMEOUT_SEC, input_total_mb, MERGE_TIMEOUT_MAX_SEC)
    fallback_timeout = _scaled_timeout(
        MERGE_FALLBACK_TIMEOUT_SEC,
        input_total_mb,
        MERGE_FALLBACK_TIMEOUT_MAX_SEC,
    )

    logger.info(
        "[merge_pdfs] Merge timeouts: gs=%ss fallback=%ss (inputs=%.1fMB, parts=%s)",
        gs_timeout,
        fallback_timeout,
        input_total_mb,
        len(pdf_paths),
    )

    def _merge_with_pypdf2(paths: List[Path], out: Path, timeout_sec: int) -> None:
        """Run PyPDF2 merge in a killable subprocess with strict timeout cleanup."""
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_merge_with_pypdf2_worker,
            args=([str(p) for p in paths], str(out), result_queue),
            daemon=True,
        )
        proc.start()
        proc.join(timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(1)
            out.unlink(missing_ok=True)
            raise TimeoutError(f"PyPDF2 merge timed out after {timeout_sec}s")

        status = None
        try:
            status = result_queue.get_nowait()
        except Exception:
            status = None

        if proc.exitcode not in (0, None):
            out.unlink(missing_ok=True)
            raise RuntimeError(f"PyPDF2 merge process exited with code {proc.exitcode}")

        if not status or not status.get("ok"):
            out.unlink(missing_ok=True)
            detail = status.get("error") if isinstance(status, dict) else "unknown merge failure"
            raise RuntimeError(f"PyPDF2 merge failed: {detail}")

        if not out.exists() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            raise RuntimeError("PyPDF2 merge produced no output")

    gs_cmd = get_ghostscript_command()
    if not gs_cmd:
        # Fallback to PyPDF2 if Ghostscript not available
        logger.warning("Ghostscript not found, falling back to PyPDF2 merge (may inflate file size)")
        _merge_with_pypdf2(pdf_paths, output_path, fallback_timeout)
        logger.info(f"Merged {len(pdf_paths)} PDFs into {output_path.name} (PyPDF2)")
        return

    # Use Ghostscript - properly deduplicates resources
    cmd = [
        gs_cmd,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/default",  # Preserve quality, just merge
        "-dNOPAUSE", "-dBATCH", "-dQUIET",
        "-dDetectDuplicateImages=true",  # Key: deduplicate shared images
        "-dPassThroughJPEGImages=true",  # Don't re-encode JPEGs
        "-dPassThroughJPXImages=true",   # Don't re-encode JPEG2000
        f"-sOutputFile={output_path}",
        *[str(p) for p in pdf_paths]
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=gs_timeout)
        if result.returncode != 0:
            logger.warning(f"Ghostscript merge failed (code {result.returncode}), falling back to PyPDF2")
            # Fallback to PyPDF2
            _merge_with_pypdf2(pdf_paths, output_path, fallback_timeout)
            logger.info(f"Merged {len(pdf_paths)} PDFs into {output_path.name} (PyPDF2 fallback)")
        else:
            logger.info(f"Merged {len(pdf_paths)} PDFs into {output_path.name} (Ghostscript)")
    except subprocess.TimeoutExpired:
        logger.warning("Ghostscript merge timed out, falling back to PyPDF2")
        _merge_with_pypdf2(pdf_paths, output_path, fallback_timeout)
        logger.info(f"Merged {len(pdf_paths)} PDFs into {output_path.name} (PyPDF2 fallback)")

    # === BLOAT DETECTION ===
    # PyPDF2 fallback can duplicate resources, causing merged file to be larger
    # than the sum of inputs. Detect and warn loudly.
    if output_path.exists():
        output_bytes = output_path.stat().st_size
        output_mb = output_bytes / (1024 * 1024)

        if output_bytes > input_total_bytes:
            bloat_bytes = output_bytes - input_total_bytes
            bloat_mb = bloat_bytes / (1024 * 1024)
            bloat_pct = (bloat_bytes / input_total_bytes) * 100 if input_total_bytes > 0 else 0
            output_mb = output_bytes / (1024 * 1024)

            logger.warning(f"[merge_pdfs] ⚠️  BLOAT DETECTED - Output larger than inputs!")
            logger.warning(f"[merge_pdfs]    Input total:  {input_total_mb:.2f}MB ({input_total_bytes:,} bytes)")
            logger.warning(f"[merge_pdfs]    Output size:  {output_mb:.2f}MB ({output_bytes:,} bytes)")
            logger.warning(f"[merge_pdfs]    Bloat: +{bloat_mb:.2f}MB (+{bloat_pct:.1f}%)")
            logger.warning(f"[merge_pdfs]    Likely cause: PyPDF2 fallback duplicated shared resources")
            # Attempt a lossless Ghostscript pass only if bloat is meaningful (>1%) and size is reasonable (<100MB).
            min_bloat_for_dedup = max(1.0, MERGE_BLOAT_ABORT_PCT * 100)
            if gs_cmd and bloat_pct > min_bloat_for_dedup and output_mb < 100:
                dedup_path = output_path.with_name(f"{output_path.stem}_dedup.pdf")
                success, _ = optimize_split_part(output_path, dedup_path)
                if success and dedup_path.exists() and dedup_path.stat().st_size < output_bytes:
                    output_path.unlink(missing_ok=True)
                    dedup_path.rename(output_path)
                    new_bytes = output_path.stat().st_size
                    new_mb = new_bytes / (1024 * 1024)
                    saved_mb = (output_bytes - new_bytes) / (1024 * 1024)
                    logger.info(
                        f"[merge_pdfs] ✅ Deduplicated merged PDF: {output_mb:.2f}MB -> {new_mb:.2f}MB "
                        f"(saved {saved_mb:.2f}MB)"
                    )
                else:
                    dedup_path.unlink(missing_ok=True)
        else:
            saved_bytes = input_total_bytes - output_bytes
            saved_mb = saved_bytes / (1024 * 1024)
            logger.info(f"[merge_pdfs] ✓ Merge efficient: {input_total_mb:.2f}MB -> {output_mb:.2f}MB (saved {saved_mb:.2f}MB)")


def split_by_size(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    skip_optimization_under_threshold: bool = False,
) -> List[Path]:
    """Split PDF into parts under threshold_mb each.

    Speed-first strategy:
    1) Try Ghostscript page-range splitting with adaptive part counts.
    2) Fall back to fast equal-page PyPDF2 splitting with adaptive part counts.
    3) Optionally (env) fall back to slow binary-search splitter.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = get_file_size_mb(pdf_path)

    # If already under threshold, just return the original
    if file_size_mb <= threshold_mb:
        logger.info(f"PDF {file_size_mb:.1f}MB already under {threshold_mb}MB, no split needed")
        return [pdf_path]

    reader = PdfReader(pdf_path, strict=False)
    total_pages = len(reader.pages)

    if total_pages == 0:
        raise StructureError.for_file(pdf_path.name, "PDF has no pages")

    # Calculate initial part count and run a few adaptive attempts.
    attempt_parts = min(total_pages, max(2, math.ceil(file_size_mb / threshold_mb)))
    max_attempts = SPLIT_ADAPTIVE_MAX_ATTEMPTS

    def _validate(parts: List[Path]) -> tuple[bool, float]:
        if not parts:
            return False, 0.0
        max_part = 0.0
        oversize = 0
        for part in parts:
            size_mb = get_file_size_mb(part)
            max_part = max(max_part, size_mb)
            if size_mb > threshold_mb:
                oversize += 1
        if oversize == 0:
            return True, max_part
        logger.warning(
            "[split_by_size] %s/%s parts exceed %.1fMB (max=%.1fMB)",
            oversize,
            len(parts),
            threshold_mb,
            max_part,
        )
        return False, max_part

    for attempt in range(1, max_attempts + 1):
        logger.info(
            "[split_by_size] Adaptive attempt %s/%s with %s parts (target %.1fMB)",
            attempt,
            max_attempts,
            attempt_parts,
            threshold_mb,
        )

        # Preferred path: Ghostscript page-range split.
        gs_parts = gs_split_by_pages(pdf_path, output_dir, attempt_parts, base_name)
        if gs_parts:
            renamed = _rename_split_parts(gs_parts, base_name, output_dir) if len(gs_parts) > 1 else gs_parts
            ok, max_part = _validate(renamed)
            if ok:
                logger.info("[split_by_size] ✓ SUCCESS via GS split (%s parts)", len(renamed))
                return renamed
            _cleanup_paths(renamed)
        else:
            # Fast fallback: equal-page PyPDF2 split (no optimization).
            pages_per_part = max(1, total_pages // attempt_parts)
            output_paths: List[Path] = []
            for i in range(attempt_parts):
                start = i * pages_per_part
                end = total_pages if i == attempt_parts - 1 else min(total_pages, (i + 1) * pages_per_part)
                if start >= end:
                    break
                writer = PdfWriter()
                for page_idx in range(start, end):
                    writer.add_page(reader.pages[page_idx])
                part_path = output_dir / f"{base_name}_part{i+1}.pdf"
                part_path.unlink(missing_ok=True)
                with open(part_path, "wb") as handle:
                    writer.write(handle)
                output_paths.append(part_path)

            ok, max_part = _validate(output_paths)
            if ok and output_paths:
                logger.info("[split_by_size] ✓ SUCCESS via fast PyPDF2 split (%s parts)", len(output_paths))
                return output_paths
            _cleanup_paths(output_paths)

        # Increase part count aggressively to avoid many retry rounds.
        ratio = max(1.05, max_part / max(0.1, threshold_mb))
        proposed = max(attempt_parts + 1, int(math.ceil(attempt_parts * ratio * 1.15)))
        proposed = min(total_pages, proposed)
        if proposed <= attempt_parts:
            break
        logger.info("[split_by_size] Increasing part count: %s -> %s", attempt_parts, proposed)
        attempt_parts = proposed

    # Best-effort quick split before any slow fallback decision.
    if attempt_parts >= 2:
        logger.warning("[split_by_size] Strict threshold not met; returning best-effort %s-part split", attempt_parts)
        gs_parts = gs_split_by_pages(pdf_path, output_dir, attempt_parts, base_name)
        if gs_parts:
            return _rename_split_parts(gs_parts, base_name, output_dir) if len(gs_parts) > 1 else gs_parts

        pages_per_part = max(1, total_pages // attempt_parts)
        best_effort_parts: List[Path] = []
        for i in range(attempt_parts):
            start = i * pages_per_part
            end = total_pages if i == attempt_parts - 1 else min(total_pages, (i + 1) * pages_per_part)
            if start >= end:
                break
            writer = PdfWriter()
            for page_idx in range(start, end):
                writer.add_page(reader.pages[page_idx])
            part_path = output_dir / f"{base_name}_part{i+1}.pdf"
            part_path.unlink(missing_ok=True)
            with open(part_path, "wb") as handle:
                writer.write(handle)
            best_effort_parts.append(part_path)
        if best_effort_parts:
            return best_effort_parts

    if SPLIT_ENABLE_BINARY_FALLBACK:
        logger.warning("[split_by_size] Adaptive split exhausted; using slow binary fallback")
        return split_pdf(
            pdf_path,
            output_dir,
            base_name,
            threshold_mb,
            skip_optimization_under_threshold=skip_optimization_under_threshold,
        )

    logger.warning(
        "[split_by_size] Adaptive split exhausted after %s attempts; returning unsplit file to preserve responsiveness",
        max_attempts,
    )
    return [pdf_path]


# =============================================================================
# DELIVERY SPLIT WITH OPTIONAL ULTRA PASS
# =============================================================================

def split_for_delivery(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    prefer_binary: bool = False,
    skip_optimization_under_threshold: bool = False,
) -> List[Path]:
    """Split with optional ultra compression to minimize part count when close to a lower split."""
    source_path = Path(pdf_path)
    source_bytes = source_path.stat().st_size
    split_required = source_bytes > int(max(0.1, threshold_mb) * 1024 * 1024)

    def _collect_valid_parts(candidate_parts: List[Path]) -> tuple[List[Path], int, float]:
        valid_parts: List[Path] = []
        total_bytes = 0
        max_part_mb = 0.0
        for part in candidate_parts:
            part_path = Path(part)
            if not part_path.exists():
                continue
            size_bytes = part_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            valid_parts.append(part_path)
            total_bytes += size_bytes
            max_part_mb = max(max_part_mb, size_mb)
        return valid_parts, total_bytes, max_part_mb

    def _cleanup_non_source(parts: List[Path]) -> None:
        for part in parts:
            try:
                if part.resolve() != source_path.resolve():
                    part.unlink(missing_ok=True)
            except OSError:
                continue

    def _tighten_parts(parts: List[Path], reason: str, force_all: bool = False) -> None:
        if not parts:
            return

        for idx, part in enumerate(parts, start=1):
            part_path = Path(part)
            if not part_path.exists():
                continue

            current_mb = get_file_size_mb(part_path)
            if not force_all and current_mb <= threshold_mb:
                continue
            if force_all and current_mb < max(1.0, threshold_mb * 0.4):
                continue

            def try_replace(label: str, fn, tag: str) -> None:
                nonlocal current_mb
                candidate = output_dir / f"{part_path.stem}_{tag}_{uuid.uuid4().hex[:6]}.pdf"
                candidate.unlink(missing_ok=True)
                ok, msg = fn(part_path, candidate)
                if not ok or not candidate.exists():
                    candidate.unlink(missing_ok=True)
                    logger.debug(
                        "[split_for_delivery] %s failed for part %s (%s): %s",
                        label,
                        idx,
                        reason,
                        msg,
                    )
                    return

                new_mb = get_file_size_mb(candidate)
                if new_mb + 0.01 < current_mb:
                    part_path.unlink(missing_ok=True)
                    candidate.rename(part_path)
                    logger.info(
                        "[split_for_delivery] %s improved part %s (%s): %.1fMB -> %.1fMB",
                        label,
                        idx,
                        reason,
                        current_mb,
                        new_mb,
                    )
                    current_mb = new_mb
                else:
                    candidate.unlink(missing_ok=True)

            try_replace("Lossless optimize", optimize_split_part, "opt")

            if ALLOW_LOSSY_COMPRESSION and (current_mb > threshold_mb or force_all):
                try_replace("Aggressive recompress", compress_pdf_with_ghostscript, "tight")

                def ultra_fn(inp: Path, out: Path):
                    return compress_ultra_aggressive(inp, out, jpeg_quality=SPLIT_ULTRA_JPEGQ)

                try_replace("Ultra recompress", ultra_fn, "ultra")

    def _finalize(candidate_parts: List[Path]) -> List[Path]:
        if len(candidate_parts) <= 1:
            return candidate_parts

        valid_parts, total_bytes, max_part_mb = _collect_valid_parts(candidate_parts)
        if not valid_parts:
            return [source_path]

        if split_required and max_part_mb > threshold_mb:
            logger.warning(
                "[split_for_delivery] Parts exceed threshold %.1fMB (max=%.1fMB); tightening parts",
                threshold_mb,
                max_part_mb,
            )
            _tighten_parts(valid_parts, reason="threshold", force_all=False)
            valid_parts, total_bytes, max_part_mb = _collect_valid_parts(valid_parts)

        if split_required and max_part_mb > threshold_mb:
            logger.warning(
                "[split_for_delivery] Still above threshold after tightening (max=%.1fMB); forcing binary split",
                max_part_mb,
            )
            _cleanup_non_source(valid_parts)
            strict_parts = split_pdf(
                source_path,
                output_dir,
                base_name,
                threshold_mb=threshold_mb,
                progress_callback=progress_callback,
                skip_optimization_under_threshold=skip_optimization_under_threshold,
            )
            valid_parts, total_bytes, max_part_mb = _collect_valid_parts(strict_parts)
            if not valid_parts:
                return [source_path]

            if split_required and max_part_mb > threshold_mb:
                logger.warning(
                    "[split_for_delivery] Threshold cannot be fully met after binary split (max=%.1fMB); returning best effort parts",
                    max_part_mb,
                )

        allowed_overhead = max(256 * 1024, int(source_bytes * 0.01))
        max_allowed = source_bytes + allowed_overhead
        if total_bytes > max_allowed:
            logger.warning(
                "[split_for_delivery] Split output exceeds overhead cap (%s -> %s bytes, cap=%s); tightening parts for smaller delivery",
                source_bytes,
                total_bytes,
                max_allowed,
            )
            _tighten_parts(valid_parts, reason="overhead", force_all=True)
            valid_parts, total_bytes, _ = _collect_valid_parts(valid_parts)
            if total_bytes > max_allowed:
                logger.warning(
                    "[split_for_delivery] Split output still above overhead cap after tightening (%s bytes > %s); honoring split anyway",
                    total_bytes,
                    max_allowed,
                )

        return valid_parts

    if prefer_binary:
        parts = split_pdf(
            pdf_path,
            output_dir,
            base_name,
            threshold_mb=threshold_mb,
            progress_callback=progress_callback,
            skip_optimization_under_threshold=skip_optimization_under_threshold,
        )
    else:
        parts = split_by_size(
            pdf_path,
            output_dir,
            base_name,
            threshold_mb,
            skip_optimization_under_threshold=skip_optimization_under_threshold,
        )

    if split_required and len(parts) <= 1:
        logger.warning(
            "[split_for_delivery] Primary split returned %s part; forcing binary split to honor threshold %.1fMB",
            len(parts),
            threshold_mb,
        )
        parts = split_pdf(
            pdf_path,
            output_dir,
            base_name,
            threshold_mb=threshold_mb,
            progress_callback=progress_callback,
            skip_optimization_under_threshold=skip_optimization_under_threshold,
        )

    if not SPLIT_MINIMIZE_PARTS or not ALLOW_LOSSY_COMPRESSION:
        return _finalize(parts)

    file_size_mb = get_file_size_mb(pdf_path)
    min_parts = max(1, math.ceil(file_size_mb / threshold_mb))
    should_try_ultra = len(parts) > min_parts and _should_try_ultra(
        file_size_mb,
        threshold_mb,
        len(parts),
        SPLIT_ULTRA_GAP_PCT,
    )

    if not should_try_ultra:
        return _finalize(parts)

    if progress_callback:
        progress_callback(90, "splitting", "Trying extra compression to reduce part count...")

    ultra_path = output_dir / f"{base_name}_ultra.pdf"
    ok, msg = compress_ultra_aggressive(pdf_path, ultra_path, jpeg_quality=SPLIT_ULTRA_JPEGQ)
    if not ok or not ultra_path.exists():
        logger.warning("[split_for_delivery] Ultra compression skipped: %s", msg)
        ultra_path.unlink(missing_ok=True)
        return _finalize(parts)

    ultra_base = f"{base_name}_ultra"
    ultra_parts = split_by_size(
        ultra_path,
        output_dir,
        ultra_base,
        threshold_mb,
        skip_optimization_under_threshold=skip_optimization_under_threshold,
    )

    if len(ultra_parts) < len(parts):
        logger.info(
            "[split_for_delivery] Ultra reduced parts: %s -> %s",
            len(parts),
            len(ultra_parts),
        )
        _cleanup_paths(parts)
        if len(ultra_parts) > 1:
            renamed = _rename_split_parts(ultra_parts, base_name, output_dir)
            ultra_path.unlink(missing_ok=True)
            return _finalize(renamed)
        return _finalize(ultra_parts)

    logger.info(
        "[split_for_delivery] Ultra did not reduce part count (%s parts); keeping original split",
        len(parts),
    )
    _cleanup_paths(ultra_parts)
    ultra_path.unlink(missing_ok=True)
    return _finalize(parts)


# =============================================================================
# ORIGINAL SPLIT FUNCTION (for binary search splitting - kept for compatibility)
# =============================================================================

def split_pdf(
    pdf_path: Path,
    output_dir: Path,
    base_name: str,
    threshold_mb: float = DEFAULT_THRESHOLD_MB,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    skip_optimization_under_threshold: bool = False,
) -> List[Path]:
    """Split a PDF into multiple parts under a size threshold.

    Uses a greedy, size-based splitter: for each part, binary-search the largest
    page range that stays under a conservative target (`threshold_mb - buffer`),
    then repeat for the remainder until all pages are assigned.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory to write output files.
        base_name: Base filename for output parts (e.g., "doc" -> "doc_part1.pdf").
        threshold_mb: Maximum size per part in MB. Defaults to SPLIT_THRESHOLD_MB.
        skip_optimization_under_threshold: Skip Ghostscript optimization when the raw part
            is already under the threshold.

    Returns:
        List of paths to the created PDF parts, all verified under threshold_mb.

    Raises:
        FileNotFoundError: If source PDF doesn't exist.
        EncryptionError: If PDF is password-protected.
        StructureError: If PDF is corrupted or malformed.
        SplitError: If PDF cannot be split due to write or structural errors.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    def report_progress(percent: int, message: str):
        if progress_callback:
            progress_callback(percent, "splitting", message)

    file_size_mb = get_file_size_mb(pdf_path)

    # If already under threshold, just return the original
    if file_size_mb <= threshold_mb:
        logger.info(f"PDF already under {threshold_mb}MB, no split needed")
        return [pdf_path]

    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f, strict=False)

            # Do not block on "encrypted" flag; attempt empty password and continue.
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                    logger.info(f"{pdf_path.name} flagged encrypted; attempted empty password and continuing split.")
                except Exception as de:
                    logger.warning(f"{pdf_path.name} flagged encrypted; decrypt attempt failed ({de}), continuing anyway.")

            total_pages = len(reader.pages)

            if total_pages == 0:
                raise StructureError.for_file(pdf_path.name, "PDF has no pages")

            def measure_pages_raw_and_optimized(start_page: int, end_page: int) -> tuple[float, float]:
                """Measure raw and optimized sizes for a page range.

                Uses Ghostscript optimization once to estimate how much PyPDF2
                output shrinks after de-duplication. This helps prevent
                over-splitting when PyPDF2 inflates sizes.
                """
                unique_id = str(uuid.uuid4())[:8]
                writer = PdfWriter()
                for idx in range(start_page, end_page):
                    writer.add_page(reader.pages[idx])
                temp_path = output_dir / f"{base_name}_{unique_id}_sample_temp.pdf"
                with open(temp_path, 'wb') as f:
                    writer.write(f)

                raw_size = get_file_size_mb(temp_path)
                optimized_path = output_dir / f"{base_name}_{unique_id}_sample_opt.pdf"
                success, _ = optimize_split_part(temp_path, optimized_path)
                if success and optimized_path.exists():
                    opt_size = get_file_size_mb(optimized_path)
                else:
                    opt_size = raw_size

                optimized_path.unlink(missing_ok=True)
                temp_path.unlink(missing_ok=True)
                return raw_size, opt_size

            # Estimate how much optimization shrinks PyPDF2 output to avoid over-splitting.
            size_ratio = 1.0
            ratio_samples: List[float] = []
            if total_pages >= 2 and file_size_mb > threshold_mb and not skip_optimization_under_threshold:
                sample_pages = min(20, total_pages)
                sample_starts = [0]
                if total_pages > sample_pages * 2:
                    sample_starts.append(total_pages // 2)
                if total_pages > sample_pages * 3:
                    sample_starts.append(max(total_pages - sample_pages, 0))

                for start in sample_starts:
                    raw_size, opt_size = measure_pages_raw_and_optimized(
                        start,
                        min(start + sample_pages, total_pages),
                    )
                    if raw_size > 0 and opt_size > 0:
                        ratio_samples.append(opt_size / raw_size)

                if ratio_samples:
                    worst_ratio = max(ratio_samples)
                    if worst_ratio < 0.98:
                        # Add 10% headroom so we don't overshoot the threshold.
                        size_ratio = min(1.0, worst_ratio * 1.1)
                        logger.info(
                            f"[split_pdf] Estimated optimized size ratio: {size_ratio:.2f} "
                            f"(from {len(ratio_samples)} sample range(s))"
                        )

            max_target = threshold_mb - SAFETY_BUFFER_MB
            # Use a greedy "fill to max" strategy: pack as many pages as possible into each part
            # while staying under a conservative per-part target (threshold - buffer).
            target_size = max_target if max_target > 0 else threshold_mb
            logger.info(
                f"Split: {file_size_mb:.2f}MB target {target_size:.1f}MB parts "
                f"(limit {threshold_mb:.1f}MB, buffer {SAFETY_BUFFER_MB:.1f}MB)"
            )

            def measure_pages(start_page: int, end_page: int, optimize: bool = False) -> float:
                """Create a temp PDF, optionally optimize it, and measure its size.

                Args:
                    start_page: Starting page index (0-based, inclusive)
                    end_page: Ending page index (0-based, exclusive)
                    optimize: If True, apply Ghostscript optimization to get accurate size
                """
                # Unique ID prevents file collisions when multiple jobs run in parallel
                unique_id = str(uuid.uuid4())[:8]

                writer = PdfWriter()
                for idx in range(start_page, end_page):
                    writer.add_page(reader.pages[idx])
                temp_path = output_dir / f"{base_name}_{unique_id}_measure_temp.pdf"
                with open(temp_path, 'wb') as f:
                    writer.write(f)

                if optimize:
                    # Apply Ghostscript optimization to get accurate final size
                    optimized_path = output_dir / f"{base_name}_{unique_id}_measure_opt.pdf"
                    success, _ = optimize_split_part(temp_path, optimized_path)
                    if success and optimized_path.exists():
                        size = get_file_size_mb(optimized_path)
                        optimized_path.unlink()
                    else:
                        size = get_file_size_mb(temp_path)
                else:
                    size = get_file_size_mb(temp_path)

                temp_path.unlink(missing_ok=True)
                if not optimize and size_ratio < 0.98:
                    # Adjust raw PyPDF2 size using estimated optimization ratio.
                    size *= size_ratio
                return size

            # Greedy split: repeatedly find the largest end page that keeps this part under target_size.
            report_progress(70, "Calculating split points...")
            page_ranges: List[tuple[int, int]] = []
            start_page = 0

            while start_page < total_pages:
                # Binary search for the maximum end_page where size(start_page, end_page) <= target_size.
                low = start_page + 1
                high = total_pages
                best_end: Optional[int] = None

                # Progress based on how many pages have been assigned to parts so far.
                progress_pct = 70 + int(10 * start_page / max(total_pages, 1))
                report_progress(progress_pct, f"Finding end of part {len(page_ranges) + 1}...")
                logger.info(
                    f"Finding part {len(page_ranges) + 1}: searching pages {start_page + 1}-{total_pages}"
                )

                while low <= high:
                    mid = (low + high) // 2
                    size = measure_pages(start_page, mid)
                    logger.info(f"  Pages {start_page + 1}-{mid}: {size:.1f}MB")

                    if size <= target_size:
                        best_end = mid
                        low = mid + 1  # Try more pages
                    else:
                        high = mid - 1  # Too big, try fewer pages

                if best_end is None:
                    # Even a single page is above the target_size. Accept it if it's still under the hard limit,
                    # otherwise this PDF can't be split to meet the threshold.
                    single_page_size = measure_pages(start_page, start_page + 1)
                    if single_page_size <= threshold_mb:
                        best_end = start_page + 1
                        logger.warning(
                            f"Page {start_page + 1} alone is {single_page_size:.1f}MB "
                            f"(over target {target_size:.1f}MB, under limit {threshold_mb:.1f}MB)."
                        )
                    else:
                        logger.warning(
                            "Page %s alone is %.1fMB (> limit %.1fMB). Keeping oversize part.",
                            start_page + 1,
                            single_page_size,
                            threshold_mb,
                        )
                        best_end = start_page + 1

                page_ranges.append((start_page, best_end))
                logger.info(f"Split point {len(page_ranges)}: pages {start_page + 1}-{best_end}")
                start_page = best_end

            # Now create the actual split parts - ALWAYS optimize to match measured sizes
            output_paths: List[Path] = []

            total_parts = len(page_ranges)
            for part_num, (range_start, range_end) in enumerate(page_ranges, 1):
                report_progress(80 + int(15 * part_num / max(total_parts, 1)), f"Creating part {part_num} of {total_parts}...")
                # Unique ID prevents file collisions when multiple jobs run in parallel
                part_unique_id = str(uuid.uuid4())[:8]

                writer = PdfWriter()
                for page_idx in range(range_start, range_end):
                    writer.add_page(reader.pages[page_idx])

                # Write raw PyPDF2 part to temporary path
                temp_part_path = output_dir / f"{base_name}_part{part_num}_{part_unique_id}_temp.pdf"
                with open(temp_part_path, 'wb') as out_f:
                    writer.write(out_f)

                raw_size_mb = get_file_size_mb(temp_part_path)
                pages_in_part = range_end - range_start
                logger.info(f"Created {temp_part_path.name}: {pages_in_part} pages, {raw_size_mb:.1f}MB (raw)")

                part_path = output_dir / f"{base_name}_part{part_num}.pdf"

                if skip_optimization_under_threshold and raw_size_mb <= threshold_mb:
                    part_path.unlink(missing_ok=True)
                    temp_part_path.rename(part_path)
                    current_size_mb = raw_size_mb
                    logger.info(
                        "Part %s under threshold (%.1fMB); skipping optimization",
                        part_num,
                        raw_size_mb,
                    )
                else:
                    # Optimize to match what we measured during binary search.
                    success, message = optimize_split_part(temp_part_path, part_path)
                    if success and part_path.exists():
                        optimized_size_mb = get_file_size_mb(part_path)
                        logger.info(f"Optimized {part_path.name}: {raw_size_mb:.1f}MB -> {optimized_size_mb:.1f}MB")

                        # If optimization made it bigger, keep the raw output instead.
                        if optimized_size_mb > raw_size_mb:
                            logger.warning(
                                f"Optimization increased size for {part_path.name}: "
                                f"{raw_size_mb:.1f}MB -> {optimized_size_mb:.1f}MB. Keeping raw output."
                            )
                            part_path.unlink(missing_ok=True)
                            temp_part_path.rename(part_path)
                            current_size_mb = raw_size_mb
                        else:
                            temp_part_path.unlink(missing_ok=True)
                            current_size_mb = optimized_size_mb
                    else:
                        # Optimization failed - just rename
                        logger.warning(f"Optimization failed: {message}")
                        part_path.unlink(missing_ok=True)
                        temp_part_path.rename(part_path)
                        current_size_mb = raw_size_mb

                def try_replace_with_smaller(label: str, suffix: str, compress_fn) -> None:
                    nonlocal current_size_mb
                    candidate_path = part_path.with_suffix(suffix)
                    candidate_path.unlink(missing_ok=True)
                    ok, msg = compress_fn(part_path, candidate_path)
                    if not ok or not candidate_path.exists():
                        logger.warning(f"{label} failed for {part_path.name}: {msg}")
                        candidate_path.unlink(missing_ok=True)
                        return

                    new_size_mb = get_file_size_mb(candidate_path)
                    if new_size_mb < current_size_mb:
                        part_path.unlink(missing_ok=True)
                        candidate_path.rename(part_path)
                        logger.info(f"{label}: {current_size_mb:.1f}MB -> {new_size_mb:.1f}MB")
                        current_size_mb = new_size_mb
                    else:
                        candidate_path.unlink(missing_ok=True)

                # If still over the hard limit, try progressively more aggressive compression.
                if current_size_mb > threshold_mb and ALLOW_LOSSY_COMPRESSION:
                    logger.info(
                        f"Part {part_num} is {current_size_mb:.1f}MB (> {threshold_mb:.1f}MB), "
                        f"trying full compression..."
                    )
                    try_replace_with_smaller("Full compression", ".compressed.pdf", compress_pdf_with_ghostscript)

                if current_size_mb > threshold_mb and ALLOW_LOSSY_COMPRESSION:
                    logger.warning(
                        f"Part {part_num} still {current_size_mb:.1f}MB (> {threshold_mb:.1f}MB), "
                        f"trying ultra compression (quality reduction)..."
                    )

                    def ultra_fn(inp: Path, out: Path):
                        return compress_ultra_aggressive(inp, out)

                    try_replace_with_smaller("Ultra compression", ".ultra.pdf", ultra_fn)
                elif current_size_mb > threshold_mb and not ALLOW_LOSSY_COMPRESSION:
                    logger.warning(
                        "Part %s remains %.1fMB (> %.1fMB). Lossy compression disabled; keeping lossless output.",
                        part_num,
                        current_size_mb,
                        threshold_mb,
                    )

                if not part_path.exists():
                    raise SplitError(f"Failed to create part {part_num}: file not written")

                output_paths.append(part_path)

            # Final verification: ensure every part is under the requested threshold.
            oversize: List[tuple[int, Path, float]] = []
            for i, p in enumerate(output_paths, 1):
                size_mb = get_file_size_mb(p)
                logger.info(f"  Part {i}: {size_mb:.1f}MB")
                if size_mb > threshold_mb:
                    oversize.append((i, p, size_mb))

            if oversize:
                details = ", ".join([f"part {i}={s:.1f}MB" for i, _, s in oversize])
                logger.warning(
                    "Split produced oversized parts for %s: %s (limit %.1fMB). Keeping output.",
                    pdf_path.name,
                    details,
                    threshold_mb,
                )

            # Return the parts
            logger.info(f"Split complete: {len(output_paths)} parts")
            report_progress(95, "Split complete")
            return output_paths

    except (EncryptionError, StructureError, SplitError):
        # Re-raise our custom exceptions as-is (already have good messages)
        raise
    except PdfReadError as e:
        # Translate PyPDF2 errors to our custom exceptions
        error_str = str(e).lower()
        if 'encrypt' in error_str or 'password' in error_str:
            raise EncryptionError.for_file(pdf_path.name) from e
        raise StructureError.for_file(pdf_path.name, str(e)) from e
    except Exception as e:
        logger.error(f"Failed to split PDF: {e}")
        raise StructureError.for_file(pdf_path.name, str(e)) from e
