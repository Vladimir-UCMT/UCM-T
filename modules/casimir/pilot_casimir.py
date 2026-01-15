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


def write_results_items(results_dir: Path, rows: list[dict]) -> None:
    p = results_dir / "results_items.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else ["item_id", "metric", "value"]
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

    # Load module to ensure it imports cleanly
    engine_mod = load_engine_module(ENGINE_PATH)
    # Minimal real metric (deterministic smoke)
    # NOTE: units/meaning depend on casimir_ucm.py definitions; this is just a pipeline sanity check.
    try:
        # Deterministic smoke parameters (dimensionless placeholders)
        L = 1.0
        N = 200
        rho = 1.0
        kappa = 1.0

        E = engine_mod.casimir_energy(L, N, rho, kappa)
        F = engine_mod.casimir_force(L, N, rho, kappa)
        status = "ok"
        note2 = f"Computed Casimir on L={L}, N={N}, rho={rho}, kappa={kappa}."

    except TypeError:
        # If signature differs, keep wrapper alive; we'll adjust after inspecting signatures.
        F = None
        E = None
        status = "needs_wiring"
        note2 = "Casimir functions loaded, but argument signature mismatch; adjust wrapper to correct signature."

    global_payload = {
        "schema": "ucm_results_contract_v1",
        "module": "casimir",
        "engine": "casimir_ucm.py",
        "engine_path": os.path.relpath(ENGINE_PATH, Path.cwd()),
        "tag": args.tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "notes": "Casimir wrapper: contract files created. " + note2,
        "metrics": {
            "items_count": 1 if F is not None else 0,
            "casimir_force_smoke": F,
            "casimir_energy_smoke": E,
        },
    }
    write_results_global(results_dir, global_payload)

    rows = []
    if F is not None:
        rows.append({"item_id": "SMOKE", "metric": "casimir_force_smoke", "value": F})
    if E is not None:
        rows.append({"item_id": "SMOKE", "metric": "casimir_energy_smoke", "value": E})

    write_results_items(results_dir, rows=rows)

    print(f"[done] wrote: {results_dir / 'results_global.json'}")
    print(f"[done] wrote: {results_dir / 'results_items.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
