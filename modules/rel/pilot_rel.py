from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = REPO_ROOT / "modules" / "rel" / "engine" / "rel_engine_v001.py"

sys.path.insert(0, str(REPO_ROOT))
from tools.contract_meta import contract_meta  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_results_dir(outdir: Path) -> Path:
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_results_items(results_dir: Path, rows: list[dict]) -> None:
    # Contract-required columns for the validator:
    fieldnames = ["item_id", "status", "score", "metric_value", "summary"]
    p = results_dir / "results_items.csv"
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
    ap = argparse.ArgumentParser(description="UCM-T REL wrapper (results contract output).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--tag", default="REL_DEMO", help="Run tag/name for bookkeeping.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_dir = ensure_results_dir(outdir)

    try:

        if not ENGINE_PATH.exists():
            raise FileNotFoundError(f"REL engine not found: {ENGINE_PATH}")
        env = os.environ.copy()
        
        rel_input_json = env.get("REL_INPUT_JSON", "").strip()
        rel_mode = env.get("REL_MODE", "").strip().lower()
        rel_input_basename = Path(rel_input_json).name if rel_input_json else ""
        rel_mode_effective = "demo"
        if rel_input_json:
            rel_mode_effective = rel_mode if rel_mode else "horizon"
            if rel_mode_effective not in {"horizon", "profile", "null_speeds"}:
                rel_mode_effective = "demo"
        
        
        if rel_input_json and rel_mode == "profile":
            out_json = str(outdir / "rel_calc_out.json")
            cmd = ["python", "-X", "utf8", str(ENGINE_PATH),
                   "--calc-profile", rel_input_json,
                   "--out", out_json]
        elif rel_input_json and rel_mode == "null_speeds":
            out_json = str(outdir / "rel_calc_out.json")
            cmd = ["python", "-X", "utf8", str(ENGINE_PATH),
                   "--calc-null-speeds", rel_input_json,
                   "--out", out_json]
        elif rel_input_json and (rel_mode == "" or rel_mode == "horizon"):
            out_json = str(outdir / "rel_calc_out.json")
            cmd = ["python", "-X", "utf8", str(ENGINE_PATH),
                   "--calc-horizon", rel_input_json,
                   "--out", out_json]
        else:
            cmd = ["python", "-X", "utf8", str(ENGINE_PATH), "--demo", "--outdir", str(outdir)]

        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
        ok = (proc.returncode == 0)
        
        # If real-mode produced an output json, attach key metrics to global payload
        calc_metrics = {}
        try:
            out_json_path = outdir / "rel_calc_out.json"
            if out_json_path.exists():
                calc = json.loads(out_json_path.read_text(encoding="utf-8-sig"))
                calc_metrics = {
                    "c0": calc.get("c0"),
                    "rho_inf": calc.get("rho_inf"),
                    "kappa": calc.get("kappa"),
                    "kappa_s": calc.get("kappa_s"),                    
                    "horizon_x": calc.get("horizon_x"),
                    "Omega_H": calc.get("Omega_H"),
                    "T_H_coeff": calc.get("T_H_coeff"),
                    "v0_at_x": calc.get("v0_at_x"),
                    "dxdt_minus": calc.get("dxdt_minus"),
                    "dxdt_plus": calc.get("dxdt_plus"),
                    "l_kappa": calc.get("l_kappa"),
                    "l_s": calc.get("l_s"),
                    "k_max_for_eps": calc.get("k_max_for_eps"),
                    "eps": calc.get("eps"),
                    "phase_loop_coeff": calc.get("phase_loop_coeff"),
                    "loop_int_vdl": calc.get("loop_int_vdl"),
                    "phase_loop": calc.get("phase_loop"),
                    "phase_sagnac": calc.get("phase_sagnac"),
                    "Omega": calc.get("Omega"),
                    "area": calc.get("area"),
                    "Gamma": calc.get("Gamma"),
                    "phase_ab": calc.get("phase_ab"),
                    "n": calc.get("n"),
                    "Gamma0": calc.get("Gamma0"),
                    "Omega0": calc.get("Omega0"),
                    "Omega0_over_c0": calc.get("Omega0_over_c0"),
                    "dispersion_relation": calc.get("dispersion_relation"),
                }

                # Fallback for smoke/demo: engine writes rel_engine_demo.json (not rel_calc_out.json)
                if calc_metrics.get("c0") is None:
                    demo_path = outdir / "rel_engine_demo.json"
                    if demo_path.exists():
                        demo = json.loads(demo_path.read_text(encoding="utf-8-sig"))
                        c0_demo = (((demo.get("calc") or {}).get("profile") or {}).get("c0"))
                        if c0_demo is not None:
                            calc_metrics["c0"] = c0_demo
                            
                # Fallback for rho_inf (demo/smoke): take from env if missing
                if calc_metrics.get("rho_inf") is None:
                    try:
                        calc_metrics["rho_inf"] = float(os.environ.get("UCM_RHO_INF", "0.0"))
                    except Exception:
                        calc_metrics["rho_inf"] = 0.0
                # Fallback for kappa (demo/smoke): take from env if missing
                if calc_metrics.get("kappa") is None:
                    try:
                        calc_metrics["kappa"] = float(os.environ.get("UCM_KAPPA", "0.0"))
                    except Exception:
                        calc_metrics["kappa"] = 0.0

                # Fallback for kappa_s (demo/smoke): take from env if missing
                if calc_metrics.get("kappa_s") is None:
                    try:
                        calc_metrics["kappa_s"] = float(os.environ.get("UCM_KAPPA_S", "0.0"))
                    except Exception:
                        calc_metrics["kappa_s"] = 0.0

              
        except Exception:
            calc_metrics = {"calc_metrics_error": "failed_to_parse_rel_calc_out_json"}

        

        rows = [{
            "item_id": "DEMO",
            "status": "ok" if ok else "fail",
            "score": 1.0 if ok else 0.0,
            "metric_value": float(proc.returncode),
            "summary": "REL calc-horizon wrapper run" if rel_input_json else "REL demo wrapper run",

        }]

        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])

        global_payload = {
            "schema": "ucm_results_contract_v1",
            "module": "rel",
            "timestamp_utc": now_iso(),
            "status": "ok" if ok else "error",
            "engine_returncode": int(proc.returncode),
            "n_items": len(rows),
            "tag": args.tag,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "engine_cmd": cmd,
            "rel_mode": rel_mode_effective,
            "rel_input_json_basename": rel_input_basename,
            **calc_metrics,
            **contract_meta(wrapper_version="calib-v2.3.1"),
        }
        if not ok:
            global_payload["error"] = f"engine_returncode={proc.returncode}"

        write_results_items(results_dir, rows)
        write_json(results_dir / "results_global.json", global_payload)

        # Wrapper publish status (0 means publish succeeded)
        write_json(results_dir / "wrapper_status.json", {
            "schema": "ucm_wrapper_status_v1",
            "status": "ok" if ok else "error",
            "returncode": 0,
            "has_items_csv": True,
            "out": "",
            "error": "" if ok else global_payload.get("error", ""),
            "published_from": "pilot_rel.py",
        })

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
            "module": "rel",
            "timestamp_utc": now_iso(),
            "status": "error",
            "engine_returncode": 1,
            "n_items": 1,
            "error": err,
            **contract_meta(wrapper_version="calib-v2.3.1"),
        }

        try:
            write_results_items(results_dir, rows)
            write_json(results_dir / "results_global.json", global_payload)
            write_json(results_dir / "wrapper_status.json", {
                "schema": "ucm_wrapper_status_v1",
                "status": "error",
                "returncode": 0,
                "has_items_csv": True,
                "out": "",
                "error": err,
                "published_from": "pilot_rel.py",
            })
            return 0
        except Exception:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
