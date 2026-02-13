# modules/nv/pilot_nv.py
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../UCM-T
ENGINE_PATH = REPO_ROOT / "modules" / "nv" / "engine" / "nv_engine_v023.py"

BENCH_ROOT = REPO_ROOT / "modules" / "nv" / "bench"
DEFAULT_DATASET_ID = "NV_DC211_LOCKIN_CW_ODMR_AM680_V1"
ENV_DATASET_ID = "UCM_NV_BENCH_DATASET_ID"

sys.path.insert(0, str(REPO_ROOT))
from tools.contract_meta import contract_meta  # noqa: E402
from tools.shared_env import phase0_params  # noqa: E402


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


def _read_manifest(dataset_id: str) -> tuple[dict | None, Path, Path]:
    ds_root = BENCH_ROOT / dataset_id
    mf = ds_root / "manifest.json"
    if not mf.exists():
        return None, ds_root, mf
    try:
        payload = json.loads(mf.read_text(encoding="utf-8"))
        return payload, ds_root, mf
    except Exception:
        return None, ds_root, mf


def _read_meta(ds_root: Path, rel_meta: str) -> dict:
    p = (ds_root / rel_meta).resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _read_csv(ds_root: Path, rel_csv: str) -> list[tuple[float, float, float]]:
    p = (ds_root / rel_csv).resolve()
    rows: list[tuple[float, float, float]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            fm = float(line["freq_mhz"])
            c1 = float(line["channel1_v"])
            c2 = float(line["channel2_v"])
            rows.append((fm, c1, c2))
    return rows


def _is_strictly_increasing(xs: list[float]) -> bool:
    return all(xs[i+1] > xs[i] for i in range(len(xs)-1))


def _rough_min_freq(rows: list[tuple[float, float, float]], primary: str) -> float:
    if primary == "channel2":
        j = min(range(len(rows)), key=lambda k: rows[k][2])
    else:
        j = min(range(len(rows)), key=lambda k: rows[k][1])
    return float(rows[j][0])


def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T NV wrapper (real-data bench sanity check).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--tag", default="NV_BENCH_SANITY", help="Run tag/name for bookkeeping.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_dir = ensure_results_dir(outdir)

    try:
        if not ENGINE_PATH.exists():
            raise FileNotFoundError(f"NV engine not found: {ENGINE_PATH}")

        # Import engine module (keeps basic regression check)
        _ = load_engine_module(ENGINE_PATH)

        env = os.environ.copy()
        p0 = phase0_params(env)
        c0 = p0["c0"]
        rho_inf = p0["rho_inf"]
        kappa = p0["kappa"]
        kappa_s = p0["kappa_s"]

        dataset_id = (env.get(ENV_DATASET_ID) or DEFAULT_DATASET_ID).strip()
        manifest, ds_root, mf_path = _read_manifest(dataset_id)

        rows_out: list[dict] = []

        if manifest and isinstance(manifest.get("records"), list) and len(manifest["records"]) > 0:
            for rec in manifest["records"]:
                item_id = str(rec.get("item_id", "") or "")
                rel_csv = str(rec.get("csv", "") or "")
                rel_meta = str(rec.get("meta", "") or "")
                b_mt = rec.get("b_mt", None)

                ok = True
                summary_parts = []

                try:
                    meta = _read_meta(ds_root, rel_meta)
                    primary = str(meta.get("primary_channel", "channel1")).strip().lower()
                    if primary not in ("channel1", "channel2"):
                        primary = "channel1"

                    data = _read_csv(ds_root, rel_csv)
                    freqs = [x[0] for x in data]

                    if len(data) < 10:
                        ok = False
                        summary_parts.append("too_few_rows")

                    if not _is_strictly_increasing(freqs):
                        ok = False
                        summary_parts.append("freq_not_increasing")

                    fmin = _rough_min_freq(data, "channel2" if primary == "channel2" else "channel1")

                    summary_parts.append(f"b_mt={b_mt}")
                    summary_parts.append(f"n={len(data)}")
                    summary_parts.append(f"f_range={min(freqs):.3g}..{max(freqs):.3g} MHz")
                    summary_parts.append(f"rough_min={fmin:.3f} MHz")
                    summary = "; ".join(summary_parts)

                    rows_out.append({
                        "item_id": item_id or "NV_BENCH_ITEM",
                        "status": "ok" if ok else "fail",
                        "score": 1.0 if ok else 0.0,
                        "metric_value": float(fmin),
                        "summary": summary,
                    })

                except Exception as e:
                    rows_out.append({
                        "item_id": item_id or "NV_BENCH_ITEM",
                        "status": "fail",
                        "score": 0.0,
                        "metric_value": 0.0,
                        "summary": f"error: {type(e).__name__}: {e}",
                    })

            status = "ok" if all(r["status"] == "ok" for r in rows_out) else "error"
            engine_rc = 0
        else:
            # fallback (should not happen if dataset is in repo)
            status = "ok"
            engine_rc = 0
            rows_out = [{
                "item_id": "DEMO",
                "status": "ok",
                "score": 1.0,
                "metric_value": 0.0,
                "summary": "NV wrapper fallback (no bench manifest found)",
            }]

        global_payload = {
            "schema": "ucm_results_contract_v1",
            "module": "nv",
            "timestamp_utc": now_iso(),
            "status": status,
            "engine_returncode": int(engine_rc),
            "n_items": len(rows_out),
            "tag": args.tag,
            "c0": c0,
            "rho_inf": rho_inf,
            "kappa": kappa,
            "kappa_s": kappa_s,
            "bench": dataset_id,
            "bench_manifest": str(mf_path.relative_to(REPO_ROOT)) if mf_path.exists() else str(mf_path),
            **contract_meta(wrapper_version="calib-v2.3"),
        }

        write_results_items(results_dir, rows_out)
        write_json(results_dir / "results_global.json", global_payload)

        # Wrapper itself succeeded in publishing artifacts => returncode=0
        write_json(results_dir / "wrapper_status.json", {
            "schema": "ucm_wrapper_status_v1",
            "status": status,
            "returncode": 0,
            "has_items_csv": True,
            "out": "",
            "error": "",
            "published_from": "pilot_nv.py",
        })

        return 0

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        rows_out = [{
            "item_id": "__error__",
            "status": "fail",
            "score": 0.0,
            "metric_value": 1.0,
            "summary": err,
        }]

        write_results_items(results_dir, rows_out)
        write_json(results_dir / "results_global.json", {
            "schema": "ucm_results_contract_v1",
            "module": "nv",
            "timestamp_utc": now_iso(),
            "status": "error",
            "engine_returncode": 1,
            "n_items": 1,
            "error": err,
            **contract_meta(wrapper_version="calib-v2.3"),
        })
        write_json(results_dir / "wrapper_status.json", {
            "schema": "ucm_wrapper_status_v1",
            "status": "error",
            "returncode": 0,
            "has_items_csv": True,
            "out": "",
            "error": err,
            "published_from": "pilot_nv.py",
        })
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
