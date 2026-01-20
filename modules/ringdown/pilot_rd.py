"""
Minimal adapter to plug Ringdown (RD) engine into UCM-T calibration pipeline.

Responsibilities (no more):
- accept --outdir
- run RD engine benchmark (default: RD_BENCH_3_NEW) from repo root
- publish contract results to:
    <outdir>/results/results_global.json
    <outdir>/results/results_items.csv
    <outdir>/results/wrapper_status.json

Robustness:
- never raises to the caller; on failure publishes status="error" contract outputs
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_items_contract(path: Path, rows: list[dict]) -> None:
    _safe_mkdir(path.parent)
    fieldnames = ["item_id", "status", "score", "metric_value", "summary"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_wrapper_status(results_dir: Path, status: str, error: str, published_from: str) -> None:
    p = results_dir / "wrapper_status.json"
    payload = {
        "schema": "ucm_wrapper_status_v1",
        "status": status,
        "error": error,
        "published_from": published_from,
    }
    _write_json(p, payload)


def _list_subdirs(p: Path) -> set[Path]:
    if not p.exists():
        return set()
    return {x for x in p.iterdir() if x.is_dir()}


def _pick_newest_dir(dirs: set[Path]) -> Optional[Path]:
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)


def _run_rd_engine(
    repo_root: Path,
    ringdown_root: Path,
    bench: str,
    tag: str,
    score: str,
    B: int,
    no_run: bool,
) -> Tuple[int, str]:
    """
    Runs RD engine via subprocess from repo_root (important for relative paths).
    Returns (returncode, combined_output_snippet).
    """
    pilot = ringdown_root / "engine" / "core" / "pilot_cvn_rd.py"
    if not pilot.exists():
        raise FileNotFoundError(f"RD pilot script not found: {pilot}")

    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(pilot),
        "--bench",
        bench,
        "--tag",
        tag,
        "--score",
        score,
        "--B",
        str(B),
        "--root",
        str(ringdown_root),
    ]
    # If engine supports --no-run, pass it; if it doesn't, caller can still set B=0.
    if no_run:
        cmd.append("--dry_run")

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


def _make_contract_global(
    *,
    engine_global: Optional[dict],
    status: str,
    returncode: int,
    bench: str,
    tag: str,
    score: str,
    ringdown_root: Path,
    stdout_tail: str,
    error: str = "",
    n_items: int = 1,
) -> dict:
    payload: dict = {
        "schema": "ucm_results_contract_v1",
        "module": "rd",
        "timestamp_utc": _now_iso(),
        "status": "ok" if status == "ok" else "error",
        "engine_returncode": int(returncode),
        "n_items": int(max(1, n_items)),
        "engine": "pilot_cvn_rd.py",
        "engine_path": os.path.relpath(ringdown_root / "engine" / "core" / "pilot_cvn_rd.py", Path.cwd()),
        "bench": bench,
        "tag": tag,
        "score_type": score,
        "stdout_tail": stdout_tail,
    }
    if error:
        payload["error"] = error
    if engine_global is not None:
        # keep original engine outputs for traceability, but do not rely on its schema
        payload["engine_results"] = engine_global
    return payload


def _metric_from_engine_global(engine_global: Optional[dict]) -> float:
    """
    Pick a numeric metric_value for contract items.csv.
    Prefer model_metrics.nll_sum_w, else model_metrics.nll_sum, else zmax, else 0.0.
    """
    if not engine_global:
        return 0.0
    mm = engine_global.get("model_metrics") or {}
    for k in ("nll_sum_w", "nll_sum", "zmax"):
        v = mm.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # fallback: try delta_f_hat, delta_tau_hat
    for k in ("delta_f_hat", "delta_tau_hat"):
        v = engine_global.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _publish_contract_from_run(
    *,
    run_dir: Path,
    out_results_dir: Path,
    status: str,
    returncode: int,
    bench: str,
    tag: str,
    score: str,
    ringdown_root: Path,
    stdout_tail: str,
) -> None:
    """
    Reads engine run outputs and publishes contract outputs in out_results_dir.
    Keeps engine originals as extra artifacts for debugging.
    """
    _safe_mkdir(out_results_dir)

    src_engine_global = run_dir / "results_global.json"
    src_engine_items = run_dir / "results_event.csv"  # engine-specific, not contract
    dst_engine_global_copy = out_results_dir / "engine_results_global.json"
    dst_engine_items_copy = out_results_dir / "engine_results_event.csv"

    engine_global: Optional[dict] = None
    n_items = 1

    if src_engine_global.exists():
        engine_global = _read_json(src_engine_global)
        n_items = int(engine_global.get("N_events") or engine_global.get("events_used") or 1)
        shutil.copy2(src_engine_global, dst_engine_global_copy)

    if src_engine_items.exists():
        shutil.copy2(src_engine_items, dst_engine_items_copy)

    metric_value = _metric_from_engine_global(engine_global)

    # Contract items: single GLOBAL row is enough for calibration
    rows = [{
        "item_id": "GLOBAL",
        "status": "ok" if status == "ok" else "fail",
        "score": 1.0 if status == "ok" else 0.0,
        "metric_value": float(metric_value),
        "summary": f"RD {bench} {score} metric (see engine_results in results_global.json)",
    }]

    # Write contract results
    contract_global = _make_contract_global(
        engine_global=engine_global,
        status=status,
        returncode=returncode,
        bench=bench,
        tag=tag,
        score=score,
        ringdown_root=ringdown_root,
        stdout_tail=stdout_tail,
        error="" if status == "ok" else "RD engine reported non-zero return code",
        n_items=n_items,
    )
    _write_json(out_results_dir / "results_global.json", contract_global)
    _write_items_contract(out_results_dir / "results_items.csv", rows)
    _write_wrapper_status(
        out_results_dir,
        status="ok" if status == "ok" else "error",
        error="" if status == "ok" else "RD wrapper error (non-zero return code)",
        published_from="results_global.json",
    )


def _publish_error_contract(out_results_dir: Path, bench: str, tag: str, score: str, ringdown_root: Path, returncode: int, stdout_tail: str, error: str) -> None:
    metric_value = 0.0
    rows = [{
        "item_id": "GLOBAL",
        "status": "fail",
        "score": 0.0,
        "metric_value": metric_value,
        "summary": error,
    }]
    contract_global = _make_contract_global(
        engine_global=None,
        status="error",
        returncode=returncode,
        bench=bench,
        tag=tag,
        score=score,
        ringdown_root=ringdown_root,
        stdout_tail=stdout_tail,
        error=error,
        n_items=1,
    )
    _write_json(out_results_dir / "results_global.json", contract_global)
    _write_items_contract(out_results_dir / "results_items.csv", rows)
    _write_wrapper_status(out_results_dir, status="error", error=error, published_from="results_global.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T RD wrapper (results contract output).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--bench", default=DEFAULT_BENCH, help="RD benchmark name (default RD_BENCH_3_NEW).")
    ap.add_argument("--tag", default="CALIB_RD", help="RD run tag passed to engine.")
    ap.add_argument("--score", default="model_nll", help="Score type passed to engine.")
    ap.add_argument("--B", type=int, default=0, help="Bootstrap reps passed to engine (0 is fast).")
    ap.add_argument("--no-run", action="store_true", help="Ask engine to skip heavy run if supported.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_results_dir = outdir / "results"
    _safe_mkdir(out_results_dir)

    repo_root = _repo_root_from_this_file()
    ringdown_root = repo_root / "modules" / "ringdown"

    # Run engine and capture tail
    try:
        rc, out_tail = _run_rd_engine(
            repo_root=repo_root,
            ringdown_root=ringdown_root,
            bench=args.bench,
            tag=args.tag,
            score=args.score,
            B=args.B,
            no_run=args.no_run,
        )

        # Locate newest run directory created by engine
        runs_dir = ringdown_root / "RUNS"
        newest = _pick_newest_dir(_list_subdirs(runs_dir))
        if newest is None:
            _publish_error_contract(
                out_results_dir,
                bench=args.bench,
                tag=args.tag,
                score=args.score,
                ringdown_root=ringdown_root,
                returncode=rc,
                stdout_tail=out_tail,
                error=f"No run directory found under {runs_dir}",
            )
            return 0

        status = "ok" if rc == 0 else "error"
        _publish_contract_from_run(
            run_dir=newest,
            out_results_dir=out_results_dir,
            status=status,
            returncode=rc,
            bench=args.bench,
            tag=args.tag,
            score=args.score,
            ringdown_root=ringdown_root,
            stdout_tail=out_tail,
        )
        return 0

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        tail = (tb + "\n").strip()
        if len(tail) > 8000:
            tail = tail[-8000:]
        _publish_error_contract(
            out_results_dir,
            bench=args.bench,
            tag=args.tag,
            score=args.score,
            ringdown_root=ringdown_root,
            returncode=1,
            stdout_tail=tail,
            error=err,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
