# modules/nv/pilot_nv.py
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../UCM-T
ENGINE_PATH = REPO_ROOT / "modules" / "nv" / "engine" / "nv_engine_v023.py"

sys.path.insert(0, str(REPO_ROOT))
from tools.contract_meta import contract_meta  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_results_global(results_dir: Path, payload: dict) -> None:
    write_json(results_dir / "results_global.json", payload)


def write_wrapper_status(
    results_dir: Path,
    status: str,
    returncode: int,
    has_items_csv: bool,
    error: str,
    published_from: str,
    out: str = "",
) -> None:
    payload = {
        "schema": "ucm_wrapper_status_v1",
        "status": status,
        "returncode": int(returncode),
        "has_items_csv": bool(has_items_csv),
        "out": out,
        "error": error,
        "published_from": published_from,
    }
    write_json(results_dir / "wrapper_status.json", payload)


def write_results_items(results_dir: Path, rows: list[dict]) -> None:
    p = results_dir / "results_items.csv"
    fieldnames = ["item_id", "status", "score", "metric_value", "summary"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "item_id": r.get("item_id", ""),
                "status": r.get("status", ""),
                "score": r.get("score", ""),
                "metric_value": r.get("metric_value", ""),
                "summary": r.get("summary", ""),
            })

def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T NV wrapper (results contract output).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--tag", default="NV_WRAPPER_DEMO", help="Run tag/name for bookkeeping.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_dir = ensure_results_dir(outdir)

    try:
        if not ENGINE_PATH.exists():
            raise FileNotFoundError(f"NV engine not found: {ENGINE_PATH}")

        # Sanity: import engine module (catch import errors early)
        _ = load_engine_module(ENGINE_PATH)

        cmd = ["python", "-X", "utf8", str(ENGINE_PATH), "--no-plots"]
        env = os.environ.copy()
        # shared medium param (Phase 0 inventory): c0
        try:
            c0 = float(env.get("UCM_C0", "2.0"))
        except Exception:
            c0 = 2.0
            
        # shared medium param (Phase 0 inventory): rho_inf
        try:
            rho_inf = float(env.get("UCM_RHO_INF", "0.0"))
        except Exception:
            rho_inf = 0.0

        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)

        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])

        ok = (proc.returncode == 0)
        status = "ok" if ok else "error"

        rows = [{
            "item_id": "DEMO",
            "status": "ok" if ok else "fail",
            "score": 1.0 if ok else 0.0,
            "metric_value": float(proc.returncode),
            "summary": "NV demo run",
        }]

        global_payload = {
            "schema": "ucm_results_contract_v1",
            "module": "nv",
            "timestamp_utc": now_iso(),
            "status": status,
            "engine_returncode": int(proc.returncode),
            "n_items": len(rows),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "tag": args.tag,
            "c0": c0,
            "rho_inf": rho_inf,            

            **contract_meta(wrapper_version="calib-v2.3"),
        }
        if not ok:
            global_payload["error"] = f"engine_returncode={proc.returncode}"

        write_results_items(results_dir, rows)
        write_results_global(results_dir, global_payload)

        # Wrapper itself succeeded in publishing artifacts => returncode=0
        write_wrapper_status(
            results_dir,
            status=status,
            returncode=0,
            has_items_csv=True,
            error="",
            published_from="pilot_nv.py",
            out="",
        )
        return 0

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        rows = [{
            "item_id": "__error__",
            "status": "fail",
            "score": 0.0,
            "metric_value": 1.0,
            "summary": err,
        }]

        global_payload = {
            "schema": "ucm_results_contract_v1",
            "module": "nv",
            "timestamp_utc": now_iso(),
            "status": "error",
            "engine_returncode": 1,
            "n_items": 1,
            "error": err,
            **contract_meta(wrapper_version="calib-v2.3"),
        }

        # Best-effort publish
        try:
            write_results_items(results_dir, rows)
            write_results_global(results_dir, global_payload)
            write_wrapper_status(
                results_dir,
                status="error",
                returncode=0,
                has_items_csv=True,
                error=err,
                published_from="pilot_nv.py",
                out="",
            )
        except Exception:
            # If even publishing fails, propagate nonzero
            return 2

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
