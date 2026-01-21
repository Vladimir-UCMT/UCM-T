#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UCM-T Rotation Curves adapter (pilot_rc.py)

Задача:
- запустить RC-runner
- собрать pilot_results.csv
- опубликовать results contract:
    results/results_global.json
    results/results_items.csv
    results/wrapper_status.json

Гарантии:
- results_items.csv ВСЕГДА содержит числовую колонку metric_value
- wrapper_status.json = ok только если results_global.json реально опубликован
- конвейер не падает исключением (последний рубеж в __main__)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.contract_meta import contract_meta


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_wrapper_status(results_dir: Path, status: str, error: str = "", published_from: str = "") -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ucm_wrapper_status_v1",
        "status": status,
        "error": error,
        "published_from": published_from,
    }
    (results_dir / "wrapper_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_error_contract(outdir: Path, error: str, stdout_tail: str = "") -> None:
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "results_global.json").write_text(
        json.dumps(
            {
                "schema": "ucm_results_contract_v1",
                "module": "rc",
                "timestamp_utc": now_iso(),
                "status": "error",
                "engine_returncode": 1,
                "n_items": 1,
                "error": error,
                "stdout_tail": stdout_tail,
                **contract_meta(wrapper_version="calib-v2.3"),
            },

            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with (results_dir / "results_items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "status", "score", "metric_value", "summary"])
        w.writerow(["GLOBAL", "error", "", 0.0, error])

    write_wrapper_status(results_dir, status="error", error=error, published_from="results_global.json")


def find_pilot_results_csv(run_dir: Path) -> Path | None:
    p = run_dir / "pilot_results.csv"
    return p if p.exists() else None


def publish_success_contract(results_dir: Path, rows: list[dict], proc_returncode: int, pilot_csv: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

    global_payload = {
        "schema": "ucm_results_contract_v1",
        "module": "rc",
        "timestamp_utc": now_iso(),
        "status": "ok",
        "n_items": len(rows),
        "engine_returncode": int(proc_returncode),
        "pilot_results_csv": str(pilot_csv) if pilot_csv else "",
        **contract_meta(wrapper_version="calib-v2.3"),
    }

    (results_dir / "results_global.json").write_text(
        json.dumps(global_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # items
    with (results_dir / "results_items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "status", "score", "metric_value", "summary"])
        for r in rows:
            gal = r.get("Galaxy") or r.get("galaxy") or r.get("Name") or "UNKNOWN"
            chi2 = r.get("chi2_nu") or r.get("chi2") or r.get("score") or ""
            try:
                mv = float(chi2)
                st = "ok"
            except Exception:
                mv = 0.0
                st = "fail"
            w.writerow([gal, st, str(chi2), mv, "preset=v3"])

    # publish wrapper status last
    write_wrapper_status(results_dir, status="ok", error="", published_from="results_global.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()

    # outdir already points to .../<run>/<module>/ (e.g. .../CALIB.../rc)
    rc_dir = outdir
    results_dir = rc_dir / "results"
    work_dir = rc_dir / "_work_rc"
    run_dir = work_dir / "run"

    repo = Path(__file__).resolve().parents[2]

    runner = repo / "modules" / "rotation-curves" / "engine" / "scripts" / (
        "pilot_sparc_ucmfit_v4_grad_pchip_rhoslab_gradfix_smartRcut_v12_1_production.py"
    )
    engine = repo / "modules" / "rotation-curves" / "engine" / "core" / "ucm_rotation_curve_2d_sparse_BASE_grad_pchip.py"

    # Ensure dirs
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not runner.exists():
        write_error_contract(outdir, f"Runner not found: {runner}", "")
        return 0

    if not engine.exists():
        write_error_contract(outdir, f"Engine not found: {engine}", "")
        return 0

    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(runner),
        "--engine",
        str(engine),
        "--preset",
        "v3",
        "--out",
        str(run_dir),
        "--sparc-dir",
        str(work_dir / "sparc_rotmod"),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Always save logs
    (rc_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").splitlines()[-40:]
        err_msg = "\n".join(tail) if tail else f"Runner failed with return code {proc.returncode}"
        write_error_contract(outdir, err_msg, stdout_tail=(proc.stdout or "")[-4000:])
        return 1

    # SUCCESS PATH
    pilot_csv = find_pilot_results_csv(run_dir)
    if pilot_csv is None:
        write_error_contract(outdir, f"Missing pilot_results.csv at {run_dir}", stdout_tail=(proc.stdout or "")[-4000:])
        return 1

    with pilot_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    publish_success_contract(results_dir, rows, proc.returncode, pilot_csv)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        # Last resort: publish error-contract into expected place
        outdir = None
        try:
            argv = sys.argv
            if "--outdir" in argv:
                outdir = Path(argv[argv.index("--outdir") + 1]).resolve()
        except Exception:
            outdir = None

        if outdir is not None:
            results_dir = outdir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            err = f"pilot_rc.py crashed: {e}\n\nTRACEBACK:\n{traceback.format_exc()}"
            try:
                (results_dir / "results_global.json").write_text(
                    json.dumps(
                        {
                            "schema": "ucm_results_contract_v1",
                            "module": "rc",
                            "timestamp_utc": now_iso(),
                            "status": "error",
                            "engine_returncode": 1,
                            "n_items": 1,
                            "error": err,
                            **contract_meta(wrapper_version="calib-v2.3"),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (results_dir / "results_items.csv").write_text(
                    "item_id,status,score,metric_value,summary\n"
                    f"GLOBAL,error,,0.0,{str(e).replace(',', ';')}\n",
                    encoding="utf-8",
                )
                write_wrapper_status(results_dir, status="error", error=str(e), published_from="pilot_rc.py")
            except Exception:
                pass

        rc = 0

    sys.exit(rc)
