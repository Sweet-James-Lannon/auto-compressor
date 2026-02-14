#!/usr/bin/env python3
"""Benchmark compress_pdf with structured logs and summaries."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pdf_compressor.engine.compress import compress_pdf  # noqa: E402

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


LINE_WIDTH = 78
BAR_WIDTH = 28


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _human_mb(value: float) -> str:
    return f"{value:.1f}MB"


def _human_sec(value: float) -> str:
    return f"{value:.2f}s"


def _divider(char: str = "-") -> str:
    return char * LINE_WIDTH


def _print_banner() -> None:
    print("=" * LINE_WIDTH)
    print("PDF COMPRESSION BENCHMARK".center(LINE_WIDTH))
    print("=" * LINE_WIDTH)


def _print_kv(label: str, value: str) -> None:
    print(f"{label:<16}: {value}")


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _collect_inputs(inputs: List[str], recursive: bool) -> List[Path]:
    seen: set[Path] = set()
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            iterator = path.rglob("*.pdf") if recursive else path.glob("*.pdf")
            for candidate in iterator:
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    files.append(candidate)
        elif path.is_file():
            if path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files)


class ProgressPrinter:
    def __init__(self, job_label: str, quiet: bool) -> None:
        self.job_label = job_label
        self.quiet = quiet
        self.last_percent = -1
        self.last_stage = ""

    def __call__(self, percent: int, stage: str, message: str) -> None:
        if self.quiet:
            return
        if percent == self.last_percent and stage == self.last_stage:
            return
        self.last_percent = percent
        self.last_stage = stage
        filled = max(0, min(BAR_WIDTH, int((percent / 100) * BAR_WIDTH)))
        bar = "#" * filled + "." * (BAR_WIDTH - filled)
        print(
            f"[bench] {self.job_label} [{bar}] {percent:3d}% {stage:<12} {message}"
        )


def _count_pages(path: Path) -> Optional[int]:
    if PdfReader is None:
        return None
    try:
        with open(path, "rb") as handle:
            reader = PdfReader(handle, strict=False)
            return len(reader.pages)
    except Exception:
        return None


def _write_json(path: Path, payload: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, payload: List[Dict[str, Any]]) -> None:
    if not payload:
        return
    fieldnames = [
        "file",
        "input_mb",
        "output_mb",
        "reduction_pct",
        "parts",
        "wall_seconds",
        "cpu_seconds",
        "throughput_mb_s",
        "compression_method",
        "compression_mode",
        "split_threshold_mb",
        "split_trigger_mb",
        "page_match",
        "status",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PDF compression.")
    parser.add_argument("inputs", nargs="+", help="PDF file(s) or directories")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument("--working-dir", default="./benchmarks", help="Directory for outputs")
    parser.add_argument("--split-threshold-mb", type=float, default=None, help="Split parts under this size")
    parser.add_argument("--split-trigger-mb", type=float, default=None, help="Only split if output exceeds this")
    parser.add_argument("--verify-pages", action="store_true", help="Verify input/output page counts")
    parser.add_argument("--cleanup", action="store_true", help="Delete output files after each run")
    parser.add_argument("--json-out", default=None, help="Write JSON summary to path")
    parser.add_argument("--csv-out", default=None, help="Write CSV summary to path")
    parser.add_argument("--quiet", action="store_true", help="Disable per-stage progress logs")
    args = parser.parse_args()

    inputs = _collect_inputs(args.inputs, args.recursive)
    if not inputs:
        print("No input PDFs found.")
        return 2

    working_dir = Path(args.working_dir).expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)

    _print_banner()
    _print_kv("Run", _now_stamp())
    _print_kv("Host", platform.node() or "unknown")
    _print_kv("Python", sys.version.split()[0])
    _print_kv("Inputs", f"{len(inputs)} file(s)")
    _print_kv("Working dir", _safe_relpath(working_dir))
    _print_kv("Split", "on" if args.split_threshold_mb else "off")
    _print_kv("Verify pages", "on" if args.verify_pages else "off")
    print(_divider())

    results: List[Dict[str, Any]] = []

    for idx, input_path in enumerate(inputs, start=1):
        input_mb = input_path.stat().st_size / (1024 * 1024)
        label = f"{idx:02d}/{len(inputs):02d}"
        print()
        print(_divider())
        print(f"[{label}] FILE: {_safe_relpath(input_path)} ({_human_mb(input_mb)})")
        print(f"        SPLIT: {args.split_threshold_mb or 'off'}  TRIGGER: {args.split_trigger_mb or 'off'}")
        print(f"        START: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

        progress = ProgressPrinter(label, quiet=args.quiet)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()

        record: Dict[str, Any] = {
            "file": str(input_path),
            "input_mb": round(input_mb, 2),
            "split_threshold_mb": args.split_threshold_mb,
            "split_trigger_mb": args.split_trigger_mb,
            "status": "ok",
        }

        output_paths: List[Path] = []
        try:
            result = compress_pdf(
                str(input_path),
                working_dir=working_dir,
                split_threshold_mb=args.split_threshold_mb,
                split_trigger_mb=args.split_trigger_mb,
                progress_callback=progress,
            )
            output_paths = [Path(p) for p in result.get("output_paths", [])]
            if not output_paths and result.get("output_path"):
                output_paths = [Path(result["output_path"])]

            output_mb = result.get("compressed_size_mb") or 0.0
            reduction = result.get("reduction_percent") or 0.0
            record.update({
                "output_mb": round(output_mb, 2),
                "reduction_pct": round(reduction, 1),
                "parts": result.get("total_parts") or len(output_paths),
                "compression_method": result.get("compression_method"),
                "compression_mode": result.get("compression_mode"),
                "output_paths": [str(p) for p in output_paths],
                "part_sizes_bytes": [p.stat().st_size for p in output_paths if p.exists()],
            })
        except Exception as exc:
            record.update({
                "status": "error",
                "error": str(exc),
                "output_mb": 0.0,
                "reduction_pct": 0.0,
                "parts": 0,
            })

        end_cpu = time.process_time()
        end_wall = time.perf_counter()
        wall_s = end_wall - start_wall
        cpu_s = end_cpu - start_cpu
        throughput = (input_mb / wall_s) if wall_s > 0 else 0.0

        record.update({
            "wall_seconds": round(wall_s, 2),
            "cpu_seconds": round(cpu_s, 2),
            "throughput_mb_s": round(throughput, 2),
        })

        page_match: Optional[bool] = None
        if args.verify_pages and record.get("status") == "ok":
            input_pages = _count_pages(input_path)
            output_pages = 0
            for out_path in output_paths:
                count = _count_pages(out_path)
                if count is None:
                    output_pages = None
                    break
                output_pages += count
            if input_pages is None or output_pages is None:
                page_match = None
            else:
                page_match = input_pages == output_pages
            record.update({
                "input_pages": input_pages,
                "output_pages": output_pages,
            })

        record["page_match"] = page_match
        results.append(record)

        status_label = "OK" if record.get("status") == "ok" else "FAIL"
        page_label = "n/a"
        if args.verify_pages:
            if page_match is True:
                page_label = "OK"
            elif page_match is False:
                page_label = "MISMATCH"

        print(f"        DONE: {status_label}  WALL={_human_sec(wall_s)} CPU={_human_sec(cpu_s)}")
        print(
            f"        OUT:  {_human_mb(record.get('output_mb', 0.0))}  "
            f"RED={record.get('reduction_pct', 0.0):.1f}%  "
            f"PARTS={record.get('parts', 0)}  "
            f"PAGES={page_label}"
        )

        if record.get("status") != "ok":
            print(f"        ERROR: {record.get('error')}")

        if args.cleanup and output_paths:
            for path in output_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    continue

    print()
    print(_divider("="))
    print("SUMMARY".center(LINE_WIDTH))
    print(_divider("="))

    header = (
        f"{'#':<3} {'file':<32} {'in':>7} {'out':>7} {'red':>6} "
        f"{'parts':>5} {'wall':>7} {'mb/s':>6} {'ok':>4}"
    )
    print(header)
    print(_divider("-"))

    for idx, row in enumerate(results, start=1):
        file_label = Path(row.get("file", "")).name
        if len(file_label) > 32:
            file_label = file_label[:29] + "..."
        status = "OK" if row.get("status") == "ok" else "FAIL"
        print(
            f"{idx:<3} {file_label:<32} "
            f"{row.get('input_mb', 0):>6.1f} "
            f"{row.get('output_mb', 0):>6.1f} "
            f"{row.get('reduction_pct', 0):>5.1f}% "
            f"{row.get('parts', 0):>5} "
            f"{row.get('wall_seconds', 0):>6.1f}s "
            f"{row.get('throughput_mb_s', 0):>6.1f} "
            f"{status:>4}"
        )

    if args.json_out:
        _write_json(Path(args.json_out), results)
        print(_divider("-"))
        print(f"Wrote JSON: {args.json_out}")

    if args.csv_out:
        _write_csv(Path(args.csv_out), results)
        print(f"Wrote CSV: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
