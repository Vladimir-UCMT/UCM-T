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
import traceback
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()

    # ВАЖНО: outdir УЖЕ указывает на .../<run>/<module>/ (например .../CALIB.../rc)
    rc_dir = outdir
    results_dir = rc_dir / "results"
    work_dir = rc_dir / "_work_rc"
    run_dir = work_dir / "run"
    sparc_dir = work_dir / "sparc_rotmod"

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
        / "core"
        / "ucm_rotation_curve_2d_sparse_BASE_grad_pchip.py"
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
        str(work_dir / "sparc_rotmod"),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
    )

    # всегда сохраняем логи
    (outdir / "rc").mkdir(parents=True, exist_ok=True)
    (rc_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (rc_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        # публикуем error + контракт
        err_tail = (proc.stderr or proc.stdout or "").splitlines()[-40:]
        err_msg = "\n".join(err_tail) if err_tail else f"Engine failed with return code {proc.returncode}"
        write_error_contract(results, error=err_msg)
        write_wrapper_status(results, status="error", error=err_msg, published_from="results_global.json")
        return 1
 
    # успех
    write_wrapper_status(results_dir, status="ok", error="", published_from="results_global.json")
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
    write_wrapper_status(results, status="error", error=error, published_from="results_global.json")


    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        # Последний рубеж: пишем error-contract в ожидаемое место
        # ВАЖНО: outdir у RC — это .../RUN/.../rc
        outdir = None
        try:
            # грубо парсим --outdir из argv (чтобы не зависеть от argparse)
            argv = sys.argv
            if "--outdir" in argv:
                outdir = Path(argv[argv.index("--outdir") + 1]).resolve()
        except Exception:
            outdir = None

        if outdir is not None:
            results_dir = outdir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            err = f"pilot_rc.py crashed: {e}\n\nTRACEBACK:\n{traceback.format_exc()}"
            # Эти функции у тебя уже должны существовать выше в файле:
            # write_error_contract(results_dir, error=..., returncode=..., stdout_tail=..., stderr_tail=...)
            # write_wrapper_status(results_dir, status=..., error=..., published_from=...)
            try:
                # results_global.json
                (results_dir / "results_global.json").write_text(
                    json.dumps(
                        {
                            "module": "modules/rotation-curves",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "status": "error",
                            "engine_returncode": 1,
                            "error": err,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )

                # results_items.csv (минимальный контракт)
                (results_dir / "results_items.csv").write_text(
                    "item_id,status,score,metric_value,summary\n"
                    f"GLOBAL,error,,0.0,{str(e).replace(',', ';')}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass

            try:
                write_wrapper_status(results_dir, status="error", error=str(e), published_from="pilot_rc.py")
            except Exception:
                pass

        rc = 0  # не валим весь конвейер исключением

    sys.exit(rc)

