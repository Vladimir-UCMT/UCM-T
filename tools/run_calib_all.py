#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all calibration module pilots and write a small summary.

Modules:
- NV:      modules/nv/pilot_nv.py
- Casimir: modules/casimir/pilot_casimir.py
- RC:      modules/rotation-curves/pilot_rc.py
- RD:      modules/ringdown/pilot_rd.py

Outputs:
- <outdir>/<MODULE>/results/{results_global.json,results_items.csv}
- <outdir>/calib_summary.json
- <outdir>/calib_summary.csv
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
    "rc": [
        "modules/rotation-curves/engine/scripts/pilot_sparc_ucmfit_v4_grad_pchip_rhoslab_gradfix_smartRcut_v12_1_production.py",
        "modules/rotation-curves/engine/core/ucm_rotation_curve_2d_sparse_BASE_grad_pchip.py",
    ],
}

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def run_one(name: str, script_rel: str, outdir: Path, extra_args: list[str], dry_run: bool = False, check_engines: bool = False) -> dict:
    rr = repo_root()
    script = rr / script_rel
    module_out = outdir / name

    module_out.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-X", "utf8", str(script), "--outdir", str(module_out)] + extra_args
    # dry-run: do not execute anything, only verify pilot path and prepare folders
    if dry_run or check_engines:
        gpath = module_out / "results" / "results_global.json"
        ipath = module_out / "results" / "results_items.csv"
        wpath = module_out / "results" / "wrapper_status.json"

        # ensure expected directory exists
        (module_out / "results").mkdir(parents=True, exist_ok=True)

        if script.exists():
            return {
                "module": name,
                "script": script_rel,
                "returncode": 0,
                "status": "dry_ok",
                "has_items_csv": ipath.exists(),
                "outdir": str(module_out),
                "error": "",
                "published_from": "",
                "stdout_tail": "",
            }
        else:
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

    gpath = module_out / "results" / "results_global.json"
    ipath = module_out / "results" / "results_items.csv"
    wpath = module_out / "results" / "wrapper_status.json"

    status = "missing_results"
    error = ""
    published_from = ""

    if wpath.exists():
        try:
            w = read_json(wpath)
            status = w.get("status", "unknown")
            error = w.get("error", "")
            published_from = w.get("published_from", "")
        except Exception as e:
            status = "bad_wrapper_status"
            error = f"{type(e).__name__}: {e}"
    elif gpath.exists():
        try:
            g = read_json(gpath)
            status = g.get("status", "unknown")
            error = g.get("error", "")
            published_from = g.get("published_from", "")
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

    action="store_true",
    help="Do not run engines; also check existence of key engine/runner files (currently RC).",


    args = ap.parse_args()

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
            results.append({
                "module": name,
                "script": script,
                "returncode": 0,
                "status": "skipped",
                "error": "",
                "has_items_csv": False,
                "published_from": "",
                "stdout_tail": "",
                "outdir": str(outdir / name),
            })
            continue
        results.append(run_one(name, script, outdir, extra, dry_run=args.dry_run, check_engines=args.check_engines))

    # Write summary JSON/CSV at root
    (outdir / "calib_summary.json").write_text(
        json.dumps(
            {
                "timestamp_utc": now_iso(),
                "outdir": str(outdir),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
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
