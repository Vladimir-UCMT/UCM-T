#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""tools/run_calib_all.py

Run all calibration module pilots and write a compact machine-readable summary.

Modules (default plan):
- NV:      modules/nv/pilot_nv.py
- Casimir: modules/casimir/pilot_casimir.py
- RC:      modules/rotation-curves/pilot_rc.py
- RD:      modules/ringdown/pilot_rd.py

Outputs (when running):
- <outdir>/<module>/results/{results_global.json, results_items.csv, wrapper_status.json?}
- <outdir>/calib_summary.json
- <outdir>/calib_summary.csv

Modes:
- normal run (default): execute pilots
- --dry-run:        only verify pilot scripts exist + create folders
- --check-engines:  verify key engine files exist (currently RC) + create folders
- --check-contract: validate an existing run directory against the results contract
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    # tools/run_calib_all.py -> parents[1] = repo root
    return Path(__file__).resolve().parents[1]


ENGINE_CHECKS: dict[str, list[str]] = {
    # keep this intentionally small: only files that must exist for the pilot to run
    "rc": [
        "modules/rotation-curves/engine/scripts/"
        "pilot_sparc_ucmfit_v4_grad_pchip_rhoslab_gradfix_smartRcut_v12_1_production.py",
        "modules/rotation-curves/engine/core/ucm_rotation_curve_2d_sparse_BASE_grad_pchip.py",
    ],
}


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def contract_check(module_out: Path) -> tuple[str, str, str, bool]:
    """Return (status, error, published_from, has_items_csv)."""
    gpath = module_out / "results" / "results_global.json"
    ipath = module_out / "results" / "results_items.csv"
    wpath = module_out / "results" / "wrapper_status.json"

    # Prefer wrapper_status.json if present (pipeline-facing status)
    if wpath.exists():
        try:
            w = read_json(wpath)
            status = str(w.get("status", "unknown"))
            error = str(w.get("error", ""))
            published_from = str(w.get("published_from", ""))
        except Exception as e:
            return ("bad_wrapper_status", f"{type(e).__name__}: {e}", "", ipath.exists())
    else:
        status = "unknown"
        error = ""
        published_from = ""

    if not gpath.exists():
        return ("missing_results", f"Missing {gpath}", published_from, ipath.exists())
    if not ipath.exists():
        return ("missing_results", f"Missing {ipath}", published_from, False)

    # Validate global JSON readability + minimal expected keys
    try:
        g = read_json(gpath)
    except Exception as e:
        return ("bad_global_json", f"{type(e).__name__}: {e}", published_from, True)

    for k in ("module", "timestamp_utc", "status", "engine_returncode", "n_items"):
        if k not in g:
            return ("bad_global_json", f"Missing key '{k}' in {gpath.name}", published_from, True)

    # Validate items CSV readability + required columns + at least one numeric metric_value
    try:
        with ipath.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return ("bad_items_csv", "CSV has no header", published_from, True)
            required_cols = {"item_id", "status", "score", "metric_value", "summary"}
            missing = sorted(required_cols - set(reader.fieldnames))
            if missing:
                return ("bad_items_csv", f"Missing columns: {', '.join(missing)}", published_from, True)

            seen_numeric_metric = False
            nrows = 0
            for row in reader:
                nrows += 1
                mv = (row.get("metric_value") or "").strip()
                if mv and _safe_float(mv):
                    seen_numeric_metric = True
                    break
            if nrows == 0:
                return ("bad_items_csv", "CSV has 0 data rows", published_from, True)
            if not seen_numeric_metric:
                return ("bad_items_csv", "No numeric metric_value found", published_from, True)

    except Exception as e:
        return ("bad_items_csv", f"{type(e).__name__}: {e}", published_from, True)

    # If wrapper/global reported error, keep it; otherwise mark contract_ok
    if status in ("error", "fail"):
        return ("error", error or "Contract check: module reported error", published_from, True)

    return ("contract_ok", "", published_from, True)


def run_one(
    name: str,
    script_rel: str,
    outdir: Path,
    extra_args: list[str],
    *,
    mode: str,
) -> dict:
    rr = repo_root()
    script = rr / script_rel
    module_out = outdir / name

    module_out.mkdir(parents=True, exist_ok=True)
    (module_out / "results").mkdir(parents=True, exist_ok=True)

    if mode == "dry_run":
        if script.exists():
            return {
                "module": name,
                "script": script_rel,
                "returncode": 0,
                "status": "dry_ok",
                "has_items_csv": (module_out / "results" / "results_items.csv").exists(),
                "outdir": str(module_out),
                "error": "",
                "published_from": "",
                "stdout_tail": "",
            }
        return {
            "module": name,
            "script": script_rel,
            "returncode": 0,
            "status": "missing_pilot",
            "has_items_csv": False,
            "outdir": str(module_out),
            "error": f"Missing pilot: {script}",
            "published_from": "",
            "stdout_tail": "",
        }

    if mode == "check_engines":
        if not script.exists():
            return {
                "module": name,
                "script": script_rel,
                "returncode": 0,
                "status": "missing_pilot",
                "has_items_csv": False,
                "outdir": str(module_out),
                "error": f"Missing pilot: {script}",
                "published_from": "",
                "stdout_tail": "",
            }

        missing_files: list[str] = []
        for rel in ENGINE_CHECKS.get(name, []):
            if not (rr / rel).exists():
                missing_files.append(rel)

        if missing_files:
            return {
                "module": name,
                "script": script_rel,
                "returncode": 0,
                "status": "missing_engine",
                "has_items_csv": False,
                "outdir": str(module_out),
                "error": "Missing engine files: " + "; ".join(missing_files),
                "published_from": "",
                "stdout_tail": "",
            }

        return {
            "module": name,
            "script": script_rel,
            "returncode": 0,
            "status": "check_ok",
            "has_items_csv": (module_out / "results" / "results_items.csv").exists(),
            "outdir": str(module_out),
            "error": "",
            "published_from": "",
            "stdout_tail": "",
        }

    if mode == "check_contract":
        status, error, published_from, has_items = contract_check(module_out)
        return {
            "module": name,
            "script": script_rel,
            "returncode": 0,
            "status": status,
            "error": error,
            "has_items_csv": has_items,
            "published_from": published_from,
            "stdout_tail": "",
            "outdir": str(module_out),
        }

    # normal run
    cmd = [sys.executable, "-X", "utf8", str(script), "--outdir", str(module_out)] + extra_args
    proc = subprocess.run(
        cmd,
        cwd=str(rr),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_tail = (proc.stdout or "")[-4000:]

    # prefer wrapper_status.json when present; else fall back to results_global.json
    gpath = module_out / "results" / "results_global.json"
    ipath = module_out / "results" / "results_items.csv"
    wpath = module_out / "results" / "wrapper_status.json"

    status = "missing_results"
    error = ""
    published_from = ""

    if wpath.exists():
        try:
            w = read_json(wpath)
            status = str(w.get("status", "unknown"))
            error = str(w.get("error", ""))
            published_from = str(w.get("published_from", ""))
        except Exception as e:
            status = "bad_wrapper_status"
            error = f"{type(e).__name__}: {e}"
    elif gpath.exists():
        try:
            g = read_json(gpath)
            status = str(g.get("status", "unknown"))
            error = str(g.get("error", ""))
            published_from = str(g.get("published_from", ""))
        except Exception as e:
            status = "bad_global_json"
            error = f"{type(e).__name__}: {e}"
    else:
        error = f"Missing {gpath}"

    return {
        "module": name,
        "script": script_rel,
        "returncode": proc.returncode,
        "status": status,
        "error": error,
        "has_items_csv": ipath.exists(),
        "published_from": published_from,
        "stdout_tail": stdout_tail,
        "outdir": str(module_out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all UCM-T calibration module pilots.")
    ap.add_argument("--outdir", required=True, help="Root output directory for the whole calibration run.")
    ap.add_argument("--skip", default="", help="Comma list: nv,casimir,rc,rd")
    ap.add_argument("--rd-no-run", action="store_true", help="Pass --no-run to RD adapter.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run engines; only check pilot script existence and create folder structure.",
    )
    ap.add_argument(
        "--check-engines",
        action="store_true",
        help="Do not run engines; also check existence of key engine/runner files (currently RC).",
    )
    ap.add_argument(
        "--check-contract",
        action="store_true",
        help="Do not run engines; validate an existing run directory against the results contract.",
    )

    args = ap.parse_args()

    # exactly one mode flag at most
    mode_flags = [args.dry_run, args.check_engines, args.check_contract]
    if sum(1 for x in mode_flags if x) > 1:
        print("ERROR: use at most one of --dry-run / --check-engines / --check-contract", file=sys.stderr)
        return 2

    mode = "run"
    if args.dry_run:
        mode = "dry_run"
    elif args.check_engines:
        mode = "check_engines"
    elif args.check_contract:
        mode = "check_contract"

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    skip = {x.strip().lower() for x in args.skip.split(",") if x.strip()}

    plan = [
        ("nv", "modules/nv/pilot_nv.py", []),
        ("casimir", "modules/casimir/pilot_casimir.py", []),
        ("rc", "modules/rotation-curves/pilot_rc.py", []),
        ("rd", "modules/ringdown/pilot_rd.py", ["--no-run"] if args.rd_no_run else []),
    ]

    results = []
    for name, script, extra in plan:
        if name in skip:
            results.append(
                {
                    "module": name,
                    "script": script,
                    "returncode": 0,
                    "status": "skipped",
                    "error": "",
                    "has_items_csv": False,
                    "published_from": "",
                    "stdout_tail": "",
                    "outdir": str(outdir / name),
                }
            )
            continue
        results.append(run_one(name, script, outdir, extra, mode=mode))

    # Write summary JSON/CSV at root
    (outdir / "calib_summary.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now_iso(),
                "outdir": str(outdir),
                "mode": mode,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with (outdir / "calib_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["module", "status", "returncode", "has_items_csv", "outdir", "error"])
        for r in results:
            w.writerow([r["module"], r["status"], r["returncode"], r["has_items_csv"], r["outdir"], r["error"]])

    # Print compact console summary
    for r in results:
        print(f"[{r['module']}] status={r['status']} rc={r['returncode']} items={r['has_items_csv']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
