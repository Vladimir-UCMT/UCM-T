# modules/nv/pilot_nv.py
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

from datetime import datetime, timezone
from pathlib import Path

# Import "raw" engine module (we keep it as-is)

import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../UCM-T
ENGINE_PATH = REPO_ROOT / "modules" / "nv" / "engine" / "nv_engine_v023.py"


def load_engine_module(engine_path: Path):
    spec = importlib.util.spec_from_file_location("nv_engine_v023", str(engine_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load engine module from: {engine_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_results_dir(outdir: Path) -> Path:
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def write_results_global(results_dir: Path, payload: dict) -> None:
    p = results_dir / "results_global.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_wrapper_status(results_dir: Path, status: str, error: str, published_from: str) -> None:
    p = results_dir / "wrapper_status.json"
    payload = {
        "schema": "ucm_wrapper_status_v1",
        "status": status,
        "error": error,
        "published_from": published_from,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_results_items(results_dir: Path, rows: list[dict]) -> None:
    p = results_dir / "results_items.csv"
    # Minimal contract: always create CSV with header (even if empty)
    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else ["item_id", "metric", "value"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T NV engine wrapper (results contract output).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--tag", default="NV_WRAPPER_DEMO", help="Run tag/name for bookkeeping.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_dir = ensure_results_dir(outdir)

    # Load module (sanity: file exists & imports)
    _ = load_engine_module(ENGINE_PATH)

    # Run NV-engine in demo mode (no CSV) with forced UTF-8 (Windows-safe)
    cmd = ["python", "-X", "utf8", str(ENGINE_PATH), "--no-plots"]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

    stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
    stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
    status = "ok" if proc.returncode == 0 else "error"

    global_payload = {
        "schema": "ucm_results_contract_v1",
        "module": "nv",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "engine_returncode": proc.returncode,
        "n_items": 1,
        "engine": "nv_engine_v023.py",
        "engine_path": os.path.relpath(ENGINE_PATH, Path.cwd()),
        "tag": args.tag,
        "notes": "NV demo run executed via subprocess.",
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    rows = [{
        "item_id": "DEMO",
        "status": "ok" if proc.returncode == 0 else "fail",
        "score": 1.0 if proc.returncode == 0 else 0.0,
        "metric_value": float(proc.returncode),
        "summary": "NV demo return code",
    }]

    write_results_global(results_dir, global_payload)

    
    write_results_items(results_dir, rows=rows)
        # wrapper-facing status (written only after publishing results)
    write_wrapper_status(
        results_dir,
        status="ok" if status == "ok" else "error",
        error="" if status == "ok" else (stderr_tail or "NV wrapper error"),
        published_from="results_global.json",
    )


    print(f"[done] wrote: {results_dir / 'results_global.json'}")
    print(f"[done] wrote: {results_dir / 'results_items.csv'}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
