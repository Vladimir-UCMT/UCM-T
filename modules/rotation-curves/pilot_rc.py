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

Гарантии:
- results_items.csv ВСЕГДА содержит числовую колонку metric_value
- конвейер никогда не падает
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_error_contract(outdir: Path, error: str, stdout_tail: str):
    results = outdir / "results"
    results.mkdir(parents=True, exist_ok=True)

    # global
    (results / "results_global.json").write_text(
        json.dumps(
            {
                "module": "modules/rotation-curves",
                "timestamp_utc": now_iso(),
                "status": "error",
                "error": error,
                "stdout_tail": stdout_tail,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # items (numeric metric enforced)
    with (results / "results_items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "status", "score", "metric_value", "summary"])
        w.writerow(["GLOBAL", "error", "", 0.0, error])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    work = outdir / "_work_rc"
    run_dir = work / "run"
    results = outdir / "results"

    run_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[2]

    runner = (
        repo
        / "modules"
        / "rotation-curves"
        / "engine"
        / "scripts"
        / "pilot_sparc_ucmfit_v4_grad_pchip_rhoslab_gradfix_smartRcut_v12_1_production.py"
    )

    engine = (
        repo
        / "modules"
        / "rotation-curves"
        / "engine"
        / "ucm_rotation_curve_2d_sparse_BASE.py"
    )

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
        str(work / "sparc_rotmod"),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout_tail = proc.stdout[-8000:] if proc.stdout else ""

    pilot_csv = run_dir / "pilot_results.csv"
    if not pilot_csv.exists():
        write_error_contract(
            outdir,
            f"pilot_results.csv not produced by runner",
            stdout_tail,
        )
        return 0

    # ===== SUCCESS PATH =====

    rows = []
    with pilot_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # items
    with (results / "results_items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "status", "score", "metric_value", "summary"])

        for r in rows:
            gal = r.get("Galaxy", "UNKNOWN")
            chi2 = r.get("chi2_nu", "")

            try:
                mv = float(chi2)
                status = "ok"
            except Exception:
                mv = 0.0
                status = "fail"

            w.writerow([gal, status, chi2, mv, "preset=v3"])

    # global
    (results / "results_global.json").write_text(
        json.dumps(
            {
                "module": "modules/rotation-curves",
                "timestamp_utc": now_iso(),
                "status": "ok",
                "n_items": len(rows),
                "engine_returncode": proc.returncode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
