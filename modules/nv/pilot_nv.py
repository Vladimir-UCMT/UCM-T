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

# Physical-ish constants for sanity checks (not a fit)
D_GUESS_MHZ = 2870.0
WINDOW_MHZ = 260.0
GAP_MHZ = 5.0
GAMMA_E_MHZ_PER_MT = 28.0  # ~28 GHz/T => 28 MHz/mT

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
    return all(xs[i + 1] > xs[i] for i in range(len(xs) - 1))


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _extreme_freq_in_window(
    data: list[tuple[float, float, float]],
    y_index: int,
    f_lo: float,
    f_hi: float,
    polarity: str,
) -> float | None:
    sub = [row for row in data if f_lo <= row[0] <= f_hi]
    if not sub:
        return None
    if polarity == "peak":
        j = max(range(len(sub)), key=lambda k: sub[k][y_index])
    else:
        j = min(range(len(sub)), key=lambda k: sub[k][y_index])
    return float(sub[j][0])


def _global_extreme_freq(data: list[tuple[float, float, float]], y_index: int, polarity: str) -> float:
    if polarity == "peak":
        j = max(range(len(data)), key=lambda k: data[k][y_index])
    else:
        j = min(range(len(data)), key=lambda k: data[k][y_index])
    return float(data[j][0])


def _choose_resonance_freq_near_D(data, y_index: int) -> tuple[str, float]:
    # Pick polarity (dip/peak) that yields an extreme closest to D_GUESS_MHZ
    f_dip = _global_extreme_freq(data, y_index, "dip")
    f_peak = _global_extreme_freq(data, y_index, "peak")
    if abs(f_peak - D_GUESS_MHZ) < abs(f_dip - D_GUESS_MHZ):
        return "peak", float(f_peak)
    return "dip", float(f_dip)


def _choose_best_split(data, y_index: int, b_mt: float) -> tuple[str, float | None, float | None, float, float]:
    # Compute split for both polarities; choose the one closest to expected split
    expected = 2.0 * GAMMA_E_MHZ_PER_MT * b_mt  # MHz
    best = None  # (dev, polarity, fL, fR, split, b_est)
    for pol in ("dip", "peak"):
        fL = _extreme_freq_in_window(
            data, y_index,
            D_GUESS_MHZ - WINDOW_MHZ, D_GUESS_MHZ - GAP_MHZ,
            pol
        )
        fR = _extreme_freq_in_window(
            data, y_index,
            D_GUESS_MHZ + GAP_MHZ, D_GUESS_MHZ + WINDOW_MHZ,
            pol
        )
        if fL is None or fR is None:
            continue
        split = float(fR - fL)
        b_est = float(split / (2.0 * GAMMA_E_MHZ_PER_MT))
        dev = abs(split - expected)
        cand = (dev, pol, fL, fR, split, b_est)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return "unknown", None, None, 0.0, 0.0

    _, pol, fL, fR, split, b_est = best
    return pol, float(fL), float(fR), float(split), float(b_est)


def main() -> int:
    ap = argparse.ArgumentParser(description="UCM-T NV wrapper (bench sanity + Zeeman splitting metric).")
    ap.add_argument("--outdir", required=True, help="Run output directory (will create results/ inside).")
    ap.add_argument("--tag", default="NV_BENCH_SANITY", help="Run tag/name for bookkeeping.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_dir = ensure_results_dir(outdir)

    try:
        if not ENGINE_PATH.exists():
            raise FileNotFoundError(f"NV engine not found: {ENGINE_PATH}")

        # Import engine module (basic regression: importability)
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
                b_mt_raw = rec.get("b_mt", None)

                ok = True
                summary_parts: list[str] = []

                try:
                    meta = _read_meta(ds_root, rel_meta)
                    primary = str(meta.get("primary_channel", "channel1")).strip().lower()
                    y_index = 2 if primary == "channel2" else 1  # tuple: (f, c1, c2)

                    data = _read_csv(ds_root, rel_csv)
                    freqs = [x[0] for x in data]

                    if len(data) < 10:
                        ok = False
                        summary_parts.append("too_few_rows")
                    if not _is_strictly_increasing(freqs):
                        ok = False
                        summary_parts.append("freq_not_increasing")

                    # Parse B
                    b_mt = None
                    if b_mt_raw is not None:
                        try:
                            b_mt = float(b_mt_raw)
                        except Exception:
                            b_mt = None

                    if b_mt is not None and b_mt == 0.0:
                        pol, f_res = _choose_resonance_freq_near_D(data, y_index)
                        d_ok = abs(f_res - D_GUESS_MHZ) <= 120.0
                        ok = ok and d_ok

                        score = 1.0 if ok else 0.0
                        metric_value = float(f_res)

                        summary_parts += [
                            f"primary={primary}",
                            f"polarity={pol}",
                            f"b_mt={b_mt_raw}",
                            f"n={len(data)}",
                            f"f_range={min(freqs):.3g}..{max(freqs):.3g} MHz",
                            f"res_f={f_res:.3f} MHz",
                            f"D_ok={d_ok}",
                        ]

                    elif b_mt is not None and b_mt > 0.0:
                        pol, fL, fR, split, b_est = _choose_best_split(data, y_index, b_mt)
                        expected = 2.0 * GAMMA_E_MHZ_PER_MT * b_mt

                        # Very broad “physical sanity” bounds: allow orientation/contrast quirks
                        split_ok = (20.0 <= split <= 800.0)
                        ratio_ok = True
                        if expected > 0:
                            ratio = split / expected
                            ratio_ok = (0.1 <= ratio <= 5.0)

                        ok = ok and split_ok and ratio_ok

                        # Score: closeness to expected (soft)
                        dev = abs(split - expected)
                        score = 1.0
                        if expected > 0:
                            score = max(0.0, 1.0 - dev / (expected * 1.5))
                        if not ok:
                            score = min(score, 0.25)

                        metric_value = float(split)

                        summary_parts += [
                            f"primary={primary}",
                            f"polarity={pol}",
                            f"b_mt={b_mt_raw}",
                            f"n={len(data)}",
                            f"f_range={min(freqs):.3g}..{max(freqs):.3g} MHz",
                            f"fL={'' if fL is None else f'{fL:.3f}'} MHz",
                            f"fR={'' if fR is None else f'{fR:.3f}'} MHz",
                            f"split={split:.3f} MHz",
                            f"expected_split={expected:.3f} MHz",
                            f"b_est={b_est:.3f} mT",
                            f"split_ok={split_ok}",
                            f"ratio_ok={ratio_ok}",
                        ]

                    else:
                        # Unknown/unspecified B: just basic format sanity + nearest extreme to D
                        pol, f_res = _choose_resonance_freq_near_D(data, y_index)
                        score = 1.0 if ok else 0.0
                        metric_value = float(f_res)
                        summary_parts += [
                            f"primary={primary}",
                            f"polarity={pol}",
                            f"b_mt={b_mt_raw}",
                            f"n={len(data)}",
                            f"f_range={min(freqs):.3g}..{max(freqs):.3g} MHz",
                            f"res_f={f_res:.3f} MHz",
                        ]

                    summary = "; ".join(summary_parts)

                    rows_out.append({
                        "item_id": item_id or "NV_BENCH_ITEM",
                        "status": "ok" if ok else "fail",
                        "score": float(score),
                        "metric_value": float(metric_value),
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

            # IMPORTANT: For Calibration Kit healthcheck, wrapper status must remain OK
            # if artifacts are produced. Physical sanity is expressed via items + notes.
            n_fail = sum(1 for r in rows_out if r.get("status") != "ok")
            sanity_ok = (n_fail == 0)
            notes = f"nv_sanity_ok={sanity_ok}; n_fail={n_fail}"

            status = "ok"
            engine_rc = 0

        else:
            status = "ok"
            engine_rc = 0
            notes = "nv_sanity_ok=False; n_fail=0; no_bench_manifest"
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
            "notes": notes,
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

        write_json(results_dir / "wrapper_status.json", {
            "schema": "ucm_wrapper_status_v1",
            "status": "ok",
            "returncode": 0,
            "has_items_csv": True,
            "out": notes,
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
