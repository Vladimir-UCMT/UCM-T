# modules/casimir/pilot_casimir.py
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import importlib.util

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../UCM-T
import sys
sys.path.insert(0, str(REPO_ROOT))
from tools.contract_meta import contract_meta

ENGINE_PATH = REPO_ROOT / "modules" / "casimir" / "engine" / "casimir_ucm.py"


def load_engine_module(engine_path: Path):
    spec = importlib.util.spec_from_file_location("casimir_ucm", str(engine_path))
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
    # Keep stable contract columns; do not auto-vary.
    fieldnames = ["item_id", "status", "score", "metric_value", "summary"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T Casimir engine wrapper (results contract output).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--tag", default="CASIMIR_WRAPPER_SMOKE", help="Run tag/name for bookkeeping.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_dir = ensure_results_dir(outdir)

    # Defaults (so we can always publish something)
    F = None
    E = None
    status = "error"
    note2 = ""
    error_msg = ""

    try:
        # Load module to ensure it imports cleanly
        engine_mod = load_engine_module(ENGINE_PATH)

        # Deterministic smoke parameters (placeholders)
        L = 1.0
        N = 200
        rho = 1.0
        kappa = 1.0

        try:
            E = engine_mod.casimir_energy(L, N, rho, kappa)
            F = engine_mod.casimir_force(L, N, rho, kappa)
            status = "ok"
            note2 = f"Computed Casimir on L={L}, N={N}, rho={rho}, kappa={kappa}."
        except TypeError as e:
            # Signature mismatch (engine functions exist but arguments differ)
            status = "needs_wiring"
            error_msg = f"TypeError: {e}"
            note2 = "Casimir functions loaded, but argument signature mismatch; adjust wrapper to correct signature."

    except Exception as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        note2 = "Casimir wrapper failed before compute."

    # n_items must be defined for all branches
    n_items = 0
    if F is not None:
        n_items += 1
    if E is not None:
        n_items += 1
    if n_items == 0:
        n_items = 1  # ensure contract indicates at least one item row

    global_payload = {
        "schema": "ucm_results_contract_v1",
        "module": "casimir",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if status == "ok" else "error",
        "engine_returncode": 0 if status == "ok" else 1,
        "n_items": n_items,
        "engine": "casimir_ucm.py",
        "engine_path": os.path.relpath(ENGINE_PATH, Path.cwd()),
        "tag": args.tag,
        "notes": "Casimir wrapper: contract files created. " + note2,
        "metrics": {
            "casimir_force_smoke": F,
            "casimir_energy_smoke": E,
        },
    }
    if error_msg:
        global_payload["error"] = error_msg

    # Build contract rows: must have at least 1 row and numeric metric_value somewhere
    rows: list[dict] = []
    if F is not None:
        rows.append(
            {
                "item_id": "SMOKE",
                "status": "ok",
                "score": 1.0,
                "metric_value": float(F),
                "summary": "Casimir force (smoke)",
            }
        )
    if E is not None:
        rows.append(
            {
                "item_id": "SMOKE_E",
                "status": "ok",
                "score": 1.0,
                "metric_value": float(E),
                "summary": "Casimir energy (smoke)",
            }
        )

    if not rows:
        # Fallback row to satisfy contract
        rows = [
            {
                "item_id": "SMOKE",
                "status": "fail",
                "score": 0.0,
                "metric_value": 0.0,
                "summary": error_msg or "Casimir wrapper produced no metrics (needs wiring).",
            }
        ]

    # Publish (global + items) then wrapper status
    global_payload.update(contract_meta(wrapper_version="calib-v2.3"))

    write_results_global(results_dir, global_payload)
    write_results_items(results_dir, rows=rows)
    write_wrapper_status(
        results_dir,
        status="ok" if status == "ok" else "error",
        error="" if status == "ok" else (error_msg or "Casimir wrapper error"),
        published_from="results_global.json",
    )

    print(f"[done] wrote: {results_dir / 'results_global.json'}")
    print(f"[done] wrote: {results_dir / 'results_items.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
