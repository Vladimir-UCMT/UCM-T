#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal adapter to plug Ringdown (RD) engine into UCM-T calibration pipeline.

Responsibilities (no more):
- accept --outdir
- run RD engine benchmark (RD_BENCH_3_NEW) from repo root
- publish results to:
    <outdir>/results/results_global.json
    <outdir>/results/results_items.csv

Robustness:
- never raises to the caller; on failure writes status="error" in results_global.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_BENCH = "RD_BENCH_3_NEW"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root_from_this_file() -> Path:
    # modules/ringdown/pilot_rd.py -> parents:
    # [0]=ringdown, [1]=modules, [2]=repo_root
    return Path(__file__).resolve().parents[2]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_min_items_csv(path: Path, status: str, value: Optional[float] = None, note: str = "") -> None:
    _safe_mkdir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "status", "value", "note"])
        w.writerow(["GLOBAL", status, "" if value is None else value, note])


def _list_subdirs(p: Path) -> set[Path]:
    if not p.exists():
        return set()
    return {x for x in p.iterdir() if x.is_dir()}


def _pick_newest_dir(dirs: set[Path]) -> Optional[Path]:
    if not dirs:
        return None
    # Pick newest by modification time
    return max(dirs, key=lambda x: x.stat().st_mtime)


def _run_rd_engine(repo_root: Path, ringdown_root: Path, bench: str, tag: str, score: str, B: int) -> Tuple[int, str]:
    """
    Runs RD engine via subprocess from repo_root (important for relative paths).
    Returns (returncode, combined_output_snippet).
    """
    pilot = ringdown_root / "engine" / "core" / "pilot_cvn_rd.py"
    if not pilot.exists():
        raise FileNotFoundError(f"RD pilot script not found: {pilot}")

    cmd = [
        sys.executable,
        "-X", "utf8",
        str(pilot),
        "--bench", bench,
        "--tag", tag,
        "--score", score,
        "--B", str(B),
        "--root", str(ringdown_root),
    ]

    # Keep output (can be huge). We capture but truncate.
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = proc.stdout or ""
    if len(out) > 8000:
        out = out[-8000:]
    return proc.returncode, out


def _publish_results(run_dir: Path, out_results_dir: Path) -> Tuple[Path, Path]:
    """
    Copies run_dir/results_global.json -> out_results_dir/results_global.json
    Copies run_dir/results_event.csv   -> out_results_dir/results_items.csv
    If results_event.csv missing, creates minimal results_items.csv.
    Returns (global_path, items_path).
    """
    src_global = run_dir / "results_global.json"
    src_event = run_dir / "results_event.csv"

    dst_global = out_results_dir / "results_global.json"
    dst_items = out_results_dir / "results_items.csv"

    if not src_global.exists():
        raise FileNotFoundError(f"Missing results_global.json in run dir: {src_global}")

    shutil.copy2(src_global, dst_global)

    if src_event.exists():
        shutil.copy2(src_event, dst_items)
    else:
        # Fallback: minimal items.csv if event-level CSV is absent
        _write_min_items_csv(dst_items, status="ok", note="fallback: results_event.csv not found")

    return dst_global, dst_items


def _make_error_global(error: str, stdout_tail: str = "") -> dict:
    return {
        "schema": "ucm.results_global.v1",
        "status": "error",
        "timestamp_utc": _now_iso(),
        "error": error,
        "stdout_tail": stdout_tail,
    }


def _make_ok_global(published_from: str, returncode: int, stdout_tail: str = "") -> dict:
    return {
        "schema": "ucm.results_global.v1",
        "status": "ok",
        "timestamp_utc": _now_iso(),
        "published_from": published_from,
        "engine_returncode": returncode,
        "stdout_tail": stdout_tail,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T Ringdown adapter (pilot_rd.py)")
    ap.add_argument("--outdir", required=True, help="Output directory for CALIB (module subdir).")
    ap.add_argument("--bench", default=DEFAULT_BENCH, help=f"RD bench name (default: {DEFAULT_BENCH}).")
    ap.add_argument("--tag", default="NEW", help="RD engine tag (default: NEW).")
    ap.add_argument("--score", default="model_nll", help="RD score metric (default: model_nll).")
    ap.add_argument("--B", type=int, default=200, help="RD engine B parameter (default: 200).")
    ap.add_argument("--no-run", action="store_true", help="Do not run engine; only publish newest existing run dir.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    out_results_dir = outdir / "results"
    _safe_mkdir(out_results_dir)

    repo_root = _repo_root_from_this_file()
    ringdown_root = repo_root / "modules" / "ringdown"
    runs_root = ringdown_root / "RUNS" / args.bench

    stdout_tail = ""
    returncode = 0

    try:
        before = _list_subdirs(runs_root)

        if not args.no_run:
            returncode, stdout_tail = _run_rd_engine(
                repo_root=repo_root,
                ringdown_root=ringdown_root,
                bench=args.bench,
                tag=args.tag,
                score=args.score,
                B=args.B,
            )

        after = _list_subdirs(runs_root)
        new_dirs = after - before if not args.no_run else after

        run_dir = _pick_newest_dir(new_dirs) or _pick_newest_dir(after)
        if run_dir is None:
            raise FileNotFoundError(f"No run directory found under: {runs_root}")

        # Publish results from chosen run directory
        _publish_results(run_dir=run_dir, out_results_dir=out_results_dir)

        # Optionally overwrite global with a tiny wrapper-global that records provenance
        # (keeps original data in the copied file; but we can keep it as-is to avoid touching content)
        # Here we keep copied global intact and add a small wrapper note file.
        wrapper_note = _make_ok_global(published_from=str(run_dir), returncode=returncode, stdout_tail=stdout_tail)
        _write_json(out_results_dir / "wrapper_status.json", wrapper_note)

        # If engine returned non-zero, do NOT fail pipeline; reflect in wrapper_status.json only.
        return 0

    except Exception as e:
        err_text = "".join(traceback.format_exception_only(type(e), e)).strip()
        wrapper_global = _make_error_global(error=err_text, stdout_tail=stdout_tail)
        _write_json(out_results_dir / "results_global.json", wrapper_global)
        _write_min_items_csv(out_results_dir / "results_items.csv", status="error", note=err_text)
        # Never break pipeline:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
