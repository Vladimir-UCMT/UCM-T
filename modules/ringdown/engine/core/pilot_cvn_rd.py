# Ringdown-engine (CVN/RD) — Pilot
# UCM Calibration Hub / modules/ringdown
#
# Status: pilot / evolving (stable locally, not frozen)
# Reproducibility guide: modules/ringdown/reproducibility.md
#
# Note: large datasets and benchmark outputs must not be committed to Git.
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_ROOT = Path(os.environ.get("CVN_RD_ROOT", Path(__file__).resolve().parents[1]))
@dataclass
class EventRow:
    event_id: str
    posterior_file: str
    posterior_format: str = "delta_csv"
    weight: float = 1.0
    meta: Dict[str, str] = field(default_factory=dict)


def read_events_csv(path: Path) -> Dict[str, EventRow]:
    out: Dict[str, EventRow] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id") or "").strip()
            if not eid:
                continue
            post = (row.get("posterior_file") or "").strip()
            fmt = (row.get("posterior_format") or "delta_csv").strip()
            w = float(row.get("weight", "1.0") or 1.0)
            meta = {k: (v or "").strip() for k, v in row.items() if k not in {"event_id","posterior_file","posterior_format","weight","note"} and v is not None}
            meta = {k: v for k, v in meta.items() if v != ""}
            out[eid] = EventRow(eid, post, fmt, w, meta)
    return out


def parse_model_params_string(s: str) -> Dict[str, float | str | bool]:
    """Parse model params either as strict JSON or as a forgiving k=v / k:v list.

    Accepts examples:
      {"M0":60,"pf":0,"pt":0}
      {M0:60,pf:0,pt:0}  (PowerShell may strip quotes)
      M0=60,pf=0,pt=0
      @... optional braces and separators ',' ';'
    """
    s = (s or "").strip()
    if not s:
        return {}

    # First try strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Tolerant parsing
    t = s.strip()
    # strip PowerShell hashtable wrapper @{...}
    if t.startswith('@{') and t.endswith('}'):
        t = t[2:-1].strip()
    # strip braces {...}
    if t.startswith('{') and t.endswith('}'):
        t = t[1:-1].strip()

    out: Dict[str, float | str | bool] = {}
    # split by comma/semicolon
    parts = re.split(r'[;,]\s*', t)
    for part in parts:
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
        elif ':' in part:
            k, v = part.split(':', 1)
        else:
            continue
        k = k.strip().strip("\"'")
        v = v.strip().strip("\"'")
        if not k:
            continue
        # parse bool/int/float where possible
        vl: float | str | bool
        if v.lower() in ('true','false'):
            vl = (v.lower() == 'true')
        else:
            try:
                if re.fullmatch(r'[-+]?\d+', v):
                    vl = int(v)
                else:
                    vl = float(v)
            except Exception:
                vl = v
        out[k] = vl
    return out


def read_bench_list(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]



def auto_find_posterior_rel(root: Path, eid: str) -> str | None:
    """Try to find a posterior CSV for event_id under root/data/cvn/* when events.csv has no entry."""
    # Common folders
    search_dirs = [
        root / "data" / "cvn" / "posteriors",
        root / "data" / "cvn" / "posteriors_demo",
        root / "data" / "cvn" / "posteriors_import",
        root / "data" / "cvn" / "posteriors_real",
    ]
    cands: List[Path] = []
    for d in search_dirs:
        if d.exists():
            cands.extend(sorted(d.glob(f"{eid}*.csv")))
    # Fallback: recursive search under data/cvn
    if not cands:
        d = root / "data" / "cvn"
        if d.exists():
            cands = sorted(d.rglob(f"{eid}*.csv"))
    if not cands:
        return None
    # Prefer shortest path (most direct)
    p = min(cands, key=lambda x: len(str(x)))
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)

def load_config(path: Path) -> dict:
    # utf-8-sig fixes BOM issues
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_samples(path: Path, fmt: str, thin: int = 1, max_n: int | None = None):
    df_list, dt_list = [], []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if thin > 1 and (i % thin != 0):
                continue

            if fmt == "delta_csv":
                df = float(row["delta_f"])
                dt = float(row["delta_tau"])
            elif fmt == "ft_csv":
                # legacy / demo format
                df = float(row["fhat"])
                dt = float(row["that"])
            else:
                raise ValueError(f"Unknown posterior_format='{fmt}' for file {path}")

            df_list.append(df)
            dt_list.append(dt)

            if max_n is not None and len(df_list) >= max_n:
                break

    if not df_list:
        raise ValueError(f"No samples read from {path}")
    return df_list, dt_list


def quantile(sorted_x: List[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    n = len(sorted_x)
    if n == 1:
        return sorted_x[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_x[lo]
    frac = pos - lo
    return sorted_x[lo] * (1 - frac) + sorted_x[hi] * frac


def summarize_samples(x: List[float], qs: List[float]) -> Dict[str, float]:
    xs = sorted(x)
    return {f"q{int(q * 100):02d}": quantile(xs, q) for q in qs}



def sigma_from_q05_q95(q05: float, q95: float) -> float:
    """Approx sigma assuming (q05,q95) are 5% and 95% quantiles of a normal."""
    # For N(0,1): q95 ≈ +1.6448536269514722, q05 ≈ -1.6448536269514722
    z = 1.6448536269514722
    width = float(q95) - float(q05)
    s = width / (2.0 * z)
    return s if s > 0 else float("nan")


def _get_meta_float(er: EventRow, keys: list[str]) -> float | None:
    """Try to fetch a float from EventRow.meta by trying multiple key names."""
    for k in keys:
        v = er.meta.get(k, "")
        if v in ("", None):
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def _posterior_median_col(csv_path: Path, cand_cols: List[str]) -> Optional[float]:
    """Read a posterior CSV and return median of the first available column in cand_cols (case-insensitive)."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rdr = csv.DictReader(f)
            if not rdr.fieldnames:
                return None
            lower_map = {h.lower(): h for h in rdr.fieldnames}
            col = None
            for c in cand_cols:
                key = c.lower()
                if key in lower_map:
                    col = lower_map[key]
                    break
            if col is None:
                return None
            vals: List[float] = []
            for row in rdr:
                v = row.get(col, "")
                if v in ("", None):
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if not vals:
                return None
            vals.sort()
            return float(quantile(vals, 0.5))
    except Exception:
        return None


def qnm_coeffs_berti_l2_m2(mode: str = "220") -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Kerr QNM fit coefficients from Berti, Cardoso & Will (Phys. Rev. D 73, 064030, 2006), Table VIII.

    Returns: (f1,f2,f3), (q1,q2,q3) for:
      mode="220"  (l=2,m=2,n=0)
      mode="221"  (l=2,m=2,n=1)
    """
    mode = (mode or "220").strip()
    if mode == "221":
        # m=2, n=1
        f = (1.3673, -1.0260, 0.1628)
        q = (0.1000, 0.5436, -0.4731)
        return f, q
    # default: 220 (m=2,n=0)
    f = (1.5251, -1.1568, 0.1292)
    q = (0.7000, 1.4187, -0.4990)
    return f, q


_T_SUN = 4.92549095e-6  # G*M_sun/c^3 [s]


def qnm_f_hz(Mf_solar: float, af: float, mode: str = "220") -> Optional[float]:
    """Return GR QNM frequency (Hz) for Kerr l=2,m=2,n in {0,1} using Berti et al. fit."""
    try:
        Mf = float(Mf_solar)
        a = float(af)
        if not (Mf > 0) or not (a == a):
            return None
        x = max(1.0 - a, 1e-9)
        (f1, f2, f3), _ = qnm_coeffs_berti_l2_m2(mode)
        F = f1 + f2 * (x ** f3)  # dimensionless: M * omega_R
        return float(F / (2.0 * math.pi * _T_SUN * Mf))
    except Exception:
        return None


def qnm_tau_s(Mf_solar: float, af: float, mode: str = "220") -> Optional[float]:
    """Return GR QNM damping time (s) via Q = pi f tau."""
    try:
        f = qnm_f_hz(Mf_solar, af, mode)
        if not f or not (f > 0):
            return None
        Mf = float(Mf_solar)
        a = float(af)
        x = max(1.0 - a, 1e-9)
        _, (q1, q2, q3) = qnm_coeffs_berti_l2_m2(mode)
        Q = q1 + q2 * (x ** q3)
        return float(Q / (math.pi * f))
    except Exception:
        return None


def ucm_predict(er: EventRow, model_params: dict) -> tuple[float, float]:
    """
    UCM model (S2): predicts (delta_f, delta_tau) using a scale extracted from posterior if available.

    Preferred (frequency-scale) form:
        df = df0 + kf * ln(f_ref/f0)
        dt = dt0 + kt * ln(f_ref/f0)

    Here f_ref is taken from posterior in this priority:
      1) explicit f220_hz/f221_hz columns (if present)
      2) computed GR QNM frequency from mf_solar & af columns (Berti fit; mode auto-detected 220/221)

    Parameters (JSON or friendly string):
      df0, dt0  : base offsets (fallback to delta_f/delta_tau if provided)
      kf, kt    : slopes (default 0)
      f0        : reference frequency in Hz (default 250)
      df_clip, dt_clip : abs clip (default 0.2; <=0 disables)

    Backward-compatible:
      If kf/kt are both 0 but pf/pt provided, the old Mf power-law is used:
         df = df0 * (Mf/M0)^pf, dt = dt0 * (Mf/M0)^pt
    """
    df0 = float(model_params.get("df0", model_params.get("delta_f", 0.0)))
    dt0 = float(model_params.get("dt0", model_params.get("delta_tau", 0.0)))

    kf = float(model_params.get("kf", 0.0))
    kt = float(model_params.get("kt", 0.0))
    f0 = float(model_params.get("f0", 250.0))

    df_clip = model_params.get("df_clip", 0.2)
    dt_clip = model_params.get("dt_clip", 0.2)

    # Resolve posterior path.
    # In KIT runs, events.csv typically stores paths relative to --root.
    # main() stores an absolute path into er.meta["_posterior_abs"]; prefer that here to avoid relying on extra attributes.
    post_path: Optional[Path] = None
    try:
        abs_p = (er.meta.get("_posterior_abs") or "").strip()
        if abs_p:
            post_path = Path(abs_p)
    except Exception:
        post_path = None
    if post_path is None and er.posterior_file:
        post_path = Path(er.posterior_file)
    if post_path is not None and not post_path.is_absolute():
        # Last resort: try to use an injected attribute (some wrappers may set er.root)
        try:
            root_like = getattr(er, "root", None)
            if root_like:
                post_path = (Path(root_like) / post_path).resolve()
        except Exception:
            pass
    # Prefer computing GR QNM frequency from Mf & af when available (robust with real HDF5-derived posteriors).
    # This also avoids the "f220_hz=250" placeholder trap.
    f_ref = None
    mf_q50 = None
    af_q50 = None
    qnm_mode = None
    tau_ref = None

    if post_path:
        mf_q50 = _posterior_median_col(post_path, [
            "mf_solar", "Mf", "m_final", "final_mass", "mass_final", "Mf_msun", "Mfinal", "Mf_solar"
        ])
        af_q50 = _posterior_median_col(post_path, [
            "af", "final_spin", "chi_f", "a_f", "spin_final", "j"
        ])
        if (mf_q50 is not None) and (af_q50 is not None):
            name = (str(er.posterior_file) + " " + str(er.event_id)).lower()
                        # Allow forcing the mode via model_params (useful when posterior comes from 221 but filenames do not carry "221").
            forced = str(model_params.get("qnm_mode", "")).strip()
            if forced in {"220", "221"}:
                qnm_mode = forced
            else:
                qnm_mode = "221" if "221" in name else "220"
            f_ref = qnm_f_hz(mf_q50, af_q50, qnm_mode)
            tau_ref = qnm_tau_s(mf_q50, af_q50, qnm_mode)

    # Fallback: explicit frequency columns (demo/test CSVs)
    if f_ref is None and post_path:
        f_ref = _posterior_median_col(post_path, [
            "f220_hz", "f221_hz", "f_mode_hz", "f_qnm_hz", "f_ring_hz",
            "f_final_hz", "f_final", "f", "freq_hz",
        ])

    # Predict deltas
    if f_ref is not None and (kf != 0.0 or kt != 0.0) and f_ref > 0:
        x = math.log(float(f_ref) / float(f0))
        df_raw = df0 + kf * x
        dt_raw = dt0 + kt * x
    else:
        # legacy Mf power-law branch if requested
        use_legacy = ("pf" in model_params) or ("pt" in model_params)
        if use_legacy:
            M0 = float(model_params.get("M0", 60.0))
            pf = float(model_params.get("pf", 0.0))
            pt = float(model_params.get("pt", 0.0))
            Mf = mf_q50
            if Mf is None:
                # try from event meta (events.csv)
                try:
                    Mf = _get_meta_float(er, ["Mf", "mf_solar", "m_final", "final_mass", "mass_final", "Mfinal"])
                except Exception:
                    Mf = None
            if Mf is not None and Mf > 0 and M0 > 0:
                df_raw = df0 * ((Mf / M0) ** pf)
                dt_raw = dt0 * ((Mf / M0) ** pt)
            else:
                df_raw = df0
                dt_raw = dt0
        else:
            df_raw = df0
            dt_raw = dt0

    # Clamp (optional)
    df = float(df_raw); dt = float(dt_raw)
    clipped_df = False; clipped_dt = False
    try:
        c = float(df_clip) if df_clip is not None else 0.0
        if c > 0:
            c = abs(c)
            if df < -c:
                df = -c; clipped_df = True
            elif df > c:
                df = c; clipped_df = True
    except Exception:
        pass
    try:
        c = float(dt_clip) if dt_clip is not None else 0.0
        if c > 0:
            c = abs(c)
            if dt < -c:
                dt = -c; clipped_dt = True
            elif dt > c:
                dt = c; clipped_dt = True
    except Exception:
        pass

    # record debug meta (stored into results_event.csv)
    try:
        er.meta["ucm_df_raw"] = f"{df_raw:.12g}"
        er.meta["ucm_dt_raw"] = f"{dt_raw:.12g}"
        er.meta["ucm_clipped_df"] = "1" if clipped_df else "0"
        er.meta["ucm_clipped_dt"] = "1" if clipped_dt else "0"

        # Keep legacy name "ucm_f220_q50" for downstream scripts; it now means f_ref (Hz) used by the model.
        er.meta["ucm_f220_q50"] = f"{float(f_ref):.12g}" if (f_ref is not None) else ""
        if mf_q50 is not None:
            er.meta["ucm_mf_q50"] = f"{float(mf_q50):.12g}"
        if af_q50 is not None:
            er.meta["ucm_af_q50"] = f"{float(af_q50):.12g}"
        if qnm_mode:
            er.meta["ucm_qnm_mode"] = qnm_mode
        if tau_ref is not None:
            er.meta["ucm_tau_q50"] = f"{float(tau_ref):.12g}"
    except Exception:
        pass

    return float(df), float(dt)


def predict_deltas(model_name: str, model_params: dict, er: EventRow) -> tuple[float, float]:
    """
    Return (delta_f_pred, delta_tau_pred) for an event.

    Supported models:
      - gr0: (0,0)
      - const: use delta_f, delta_tau
      - ucm: mass-scaled placeholder using EventRow.meta (Mf)
    """
    name = (model_name or "gr0").strip().lower()
    if name in {"gr0", "zero", "null"}:
        return 0.0, 0.0
    if name in {"const", "constant"}:
        return float(model_params.get("delta_f", model_params.get("df0", 0.0))), float(model_params.get("delta_tau", model_params.get("dt0", 0.0)))
    if name == "ucm":
        return ucm_predict(er, model_params)
    return 0.0, 0.0
    if name in {"const", "constant"}:
        return float(model_params.get("delta_f", 0.0)), float(model_params.get("delta_tau", 0.0))
    # Fallback: unknown model → behave like GR but be explicit in report via model_name
    return 0.0, 0.0


def weighted_mean(values: List[float], weights: List[float]) -> float:
    s = 0.0
    w = 0.0
    for v, ww in zip(values, weights):
        s += v * ww
        w += ww
    return s / w if w != 0 else float("nan")


def ensure_run_dir(out_root: Path, bench_name: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{bench_name}_{tag}"
    out = out_root / bench_name / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return out


def write_manifest(path: Path, cfg: dict, inputs: dict):
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "python": sys.version.replace("\n", " "),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cwd": os.getcwd(),
        "config": cfg,
        "inputs": inputs,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_weights_report(run_dir: Path):
    """Write weights_report.csv based on results_event.csv in run_dir."""
    res = run_dir / "results_event.csv"
    if not res.exists():
        return

    out = run_dir / "weights_report.csv"

    rows = []
    with res.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = row["event_id"].strip()
            w = float(row.get("weight", 1.0))
            q05f = float(row["delta_f_q05"]); q95f = float(row["delta_f_q95"])
            q05t = float(row["delta_tau_q05"]); q95t = float(row["delta_tau_q95"])
            sf = q95f - q05f
            st = q95t - q05t
            rows.append((eid, w, sf, st))

    rows.sort(key=lambda x: x[1], reverse=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "weight", "sigma_f", "sigma_tau"])
        for eid, ww, sf, st in rows:
            w.writerow([eid, ww, sf, st])


def append_weights_diagnostic_to_report(run_dir: Path, report_lines: list[str], top_n: int = 5):
    wr = run_dir / "weights_report.csv"
    if not wr.exists():
        return

    rows = []
    with wr.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((
                row["event_id"].strip(),
                float(row["weight"]),
                float(row["sigma_f"]),
                float(row["sigma_tau"]),
            ))

    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    top = rows_sorted[:top_n]
    bot = list(reversed(rows_sorted[-top_n:]))

    report_lines.append("")
    report_lines.append(f"Weights diagnostic (top/bottom {top_n} by weight):")
    report_lines.append("TOP:")
    for eid, w, sf, st in top:
        report_lines.append(f"  {eid:24s} w={w:.4g}  sf={sf:.4g}  st={st:.4g}")
    report_lines.append("BOTTOM:")
    for eid, w, sf, st in bot:
        report_lines.append(f"  {eid:24s} w={w:.4g}  sf={sf:.4g}  st={st:.4g}")


def write_worst10(run_dir: Path):
    """worst10.txt with score_q50 + weight + event_id"""
    res_path = run_dir / "results_event.csv"
    if not res_path.exists():
        return

    tmp = []
    with res_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            score = float(row.get("score", row.get("score_q50", "0")) or 0.0)
            w = float(row.get("weight", "1"))
            eid = row.get("event_id", "")
            tmp.append((score, w, eid))

    tmp.sort(reverse=True, key=lambda x: x[0])
    top = tmp[:10]

    worst_path = run_dir / "worst10.txt"
    with worst_path.open("w", encoding="utf-8") as f:
        f.write("score,weight,event_id\n")
        for score, w, eid in top:
            f.write(f"{score:.9g},{w:.9g},{eid}\n")


def main():
    ap = argparse.ArgumentParser(description="CVN Ringdown Engine (KIT-first) — FAST mode prototype")
    ap.add_argument("--bench", required=True, help="Bench name, e.g. RD_BENCH10")
    ap.add_argument("--mode", default="fast", choices=["fast"], help="Currently only fast mode is implemented")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Project root (default: env CVN_RD_ROOT or script folder)")
    ap.add_argument("--events", default=None, help="Path to events.csv (default: <root>/data/cvn/events.csv)")
    ap.add_argument("--bench_list", default=None, help="Path to bench list txt (default: bench/<BENCH>.txt)")
    ap.add_argument("--config", default=None, help="Path to config.json (default: <root>/configs/cvn_rd_default.json)")
    ap.add_argument("--tag", default="DEV", help="Run tag suffix")
    ap.add_argument("--events_pick", default=None, help="Comma-separated event_ids to override bench list")
    ap.add_argument("--out_root", default=None, help="Output root for RUNS (default: <root>/RUNS)")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs and exit (no computation)")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for bootstrap (default: 12345)")
    ap.add_argument("--B", type=int, default=2000, help="Bootstrap iterations (default: 2000)")
    ap.add_argument("--score", default="q50_abs", choices=["q50_abs", "z0", "model_nll"],
                    help="Per-event score: q50_abs=|q50_f|+|q50_tau|, z0=distance of 0 from posterior median, model_nll=Gaussian NLL of model prediction")
    ap.add_argument("--model", default=None, help="Model name for prediction (default: from config or gr0). Supported: gr0|const|ucm")
    ap.add_argument("--model_params", default="", help='JSON dict, e.g. {"delta_f":0.1,"delta_tau":-0.2} or for ucm {"df0":...,"dt0":...,"M0":60,"pf":0,"pt":0}')
    ap.add_argument("--model_params_file", default="", help="Path to JSON file with model params (optional)")
    ap.add_argument("--baseline_const", default="", help="Path to best_const_*.json (optional; used to fill params)")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    # Input paths
    events_path = (Path(args.events) if args.events else (root / "data" / "cvn" / "events.csv")).resolve()
    config_path = (Path(args.config) if args.config else (root / "configs" / "cvn_rd_default.json")).resolve()
    bench_list_path = (Path(args.bench_list) if args.bench_list else (root / "bench" / f"{args.bench}.txt")).resolve()

    # Output root (where RUNS will be written)
    out_root = (Path(args.out_root) if args.out_root else (root / "RUNS")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Config is optional (to support "working script" setups without config file)
    cfg = {}
    if config_path.exists():
        cfg = load_config(config_path)
    cfg_model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    # Optional baseline const (best_const_*.json)
    baseline = None
    if args.baseline_const:
        try:
            with open(args.baseline_const, "r", encoding="utf-8") as f:
                baseline = json.load(f)
        except Exception as e:
            baseline = {"error": str(e), "path": args.baseline_const}


    model_name = (args.model or cfg_model.get("name") or "gr0").strip()
    model_params = cfg_model.get("params", {}) if isinstance(cfg_model.get("params", {}), dict) else {}

    # --model_params overrides config
    if args.model_params_file:
        try:
            with open(args.model_params_file, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                model_params.update(obj)
        except Exception as e:
            raise ValueError(f"Failed to read --model_params_file: {e}")

    if args.model_params:
        try:
            obj = parse_model_params_string(args.model_params)
            if isinstance(obj, dict):
                model_params.update(obj)
        except Exception as e:
            raise ValueError(f"Failed to parse --model_params: {e}")

    # Fill params from baseline const if requested/available
    b_df = b_dt = None
    if isinstance(baseline, dict) and baseline.get("model") == "const":
        try:
            b_df = float(baseline.get("delta_f"))
            b_dt = float(baseline.get("delta_tau"))
        except Exception:
            b_df = b_dt = None

    if model_name.lower() in {"const", "constant"}:
        if ("delta_f" not in model_params) and (b_df is not None):
            model_params["delta_f"] = b_df
        if ("delta_tau" not in model_params) and (b_dt is not None):
            model_params["delta_tau"] = b_dt

    if model_name.lower() == "ucm":
        # convenience: if df0/dt0 not provided, seed them from baseline const
        if ("df0" not in model_params) and ("delta_f" not in model_params) and (b_df is not None):
            model_params["df0"] = b_df
        if ("dt0" not in model_params) and ("delta_tau" not in model_params) and (b_dt is not None):
            model_params["dt0"] = b_dt

    if not events_path.exists():
        raise FileNotFoundError(f"events.csv not found: {events_path}")

    if not bench_list_path.exists():
        raise FileNotFoundError(f"bench_list not found: {bench_list_path}")
    all_events = read_events_csv(events_path)



    bench_ids = read_bench_list(bench_list_path)
    if args.events_pick:
        bench_ids = [x.strip() for x in args.events_pick.split(",") if x.strip()]

    if args.dry_run:
        print(f"DRY RUN OK: N_events={len(bench_ids)}")
        for eid in bench_ids[:20]:
            if eid in all_events:
                er = all_events[eid]
                print(f"  {eid} -> {er.posterior_file} ({er.posterior_format}) w={er.weight}")
            else:
                print(f"  {eid} -> NOT_IN_EVENTS_CSV")
        if len(bench_ids) > 20:
            print(f"  ... ({len(bench_ids)-20} more)")
        return
    run_dir = ensure_run_dir(out_root, args.bench, args.tag)

    thin = int(cfg.get("samples", {}).get("thin", 1))
    max_n = cfg.get("samples", {}).get("max_per_event", None)
    max_n = int(max_n) if max_n is not None else None

    q_event = cfg.get("outputs", {}).get("save_event_quantiles", [0.05, 0.5, 0.95])
    q_global = cfg.get("outputs", {}).get("save_global_quantiles", [0.05, 0.5, 0.95])

    # validate inputs
    missing = []
    posterior_files = []
    for eid in bench_ids:
        if eid not in all_events:
            auto_rel = auto_find_posterior_rel(root, eid)
            if auto_rel:
                print(f"[warn] {eid}: NOT_IN_EVENTS_CSV -> auto posterior_file={auto_rel}", flush=True)
                all_events[eid] = EventRow(eid, auto_rel, "delta_csv", 1.0, {"note": "AUTO_ADDED"})
            else:
                missing.append((eid, "NOT_IN_EVENTS_CSV"))
                continue

        post_rel = (all_events[eid].posterior_file or "").strip()
        if not post_rel:
            auto_rel = auto_find_posterior_rel(root, eid)
            if auto_rel:
                print(f"[warn] {eid}: empty posterior_file -> auto posterior_file={auto_rel}", flush=True)
                all_events[eid].posterior_file = auto_rel
                post_rel = auto_rel
            else:
                missing.append((eid, "EMPTY_POSTERIOR_FILE"))
                continue

        post_path = (root / post_rel)
        posterior_files.append(post_rel)
        if not post_path.exists():
            # last chance: try auto
            auto_rel = auto_find_posterior_rel(root, eid)
            if auto_rel and (root / auto_rel).exists():
                print(f"[warn] {eid}: missing {post_rel} -> auto posterior_file={auto_rel}", flush=True)
                all_events[eid].posterior_file = auto_rel
                post_rel = auto_rel
                post_path = (root / post_rel)
            else:
                missing.append((eid, post_rel))

        # store absolute posterior path for downstream models (e.g., ucm reading posterior scale)
        try:
            all_events[eid].meta["_posterior_abs"] = str(post_path.resolve())
        except Exception:
            all_events[eid].meta["_posterior_abs"] = str(post_path)

    if missing:
        msg = ["Missing inputs:"]
        for eid, info in missing:
            msg.append(f"  {eid}: {info}")
        msg.append("")
        msg.append("Hints:")
        msg.append(f"  events.csv = {events_path}")
        msg.append(f"  root      = {root}")
        raise FileNotFoundError("\n".join(msg))

    # Per-event summaries
    rows_out = []
    global_df_medians = []
    global_dt_medians = []
    global_weights = []

    for eid in bench_ids:
        er = all_events[eid]
        post_path = (root / er.posterior_file).resolve()
        df_s, dt_s = load_samples(post_path, er.posterior_format, thin=thin, max_n=max_n)

        s_df = summarize_samples(df_s, q_event)
        s_dt = summarize_samples(dt_s, q_event)

        med_df = s_df.get("q50", statistics.median(df_s))
        med_dt = s_dt.get("q50", statistics.median(dt_s))


        # Posterior widths (approx sigmas) from q05..q95 (robust + stdlib only)
        sigma_f = sigma_from_q05_q95(s_df.get("q05", float("nan")), s_df.get("q95", float("nan")))
        sigma_tau = sigma_from_q05_q95(s_dt.get("q05", float("nan")), s_dt.get("q95", float("nan")))

        # Baseline: how far GR(0,0) is from posterior median (in sigma units)
        z0_f = (0.0 - med_df) / sigma_f if sigma_f == sigma_f else float("nan")
        z0_tau = (0.0 - med_dt) / sigma_tau if sigma_tau == sigma_tau else float("nan")
        z0 = math.sqrt(z0_f * z0_f + z0_tau * z0_tau) if (z0_f == z0_f and z0_tau == z0_tau) else float("nan")

        # Model prediction hook (S1): params + optional event meta → (delta_f_pred, delta_tau_pred)
        pred_df, pred_dt = predict_deltas(model_name, model_params, er)
        z_f = (pred_df - med_df) / sigma_f if sigma_f == sigma_f else float("nan")
        z_tau = (pred_dt - med_dt) / sigma_tau if sigma_tau == sigma_tau else float("nan")

        # Scores
        score_q50 = abs(med_df) + abs(med_dt)
        if args.score == "q50_abs":
            score = score_q50
        elif args.score == "z0":
            score = z0
        elif args.score == "model_nll":
            # Gaussian negative log-likelihood up to an additive constant
            # (includes log(sigma) to prefer tighter posteriors when comparing models)
            if (sigma_f == sigma_f) and (sigma_tau == sigma_tau):
                score = 0.5 * (z_f * z_f + z_tau * z_tau) + math.log(max(sigma_f, 1e-300)) + math.log(max(sigma_tau, 1e-300))
            else:
                score = float("nan")
        elif args.score == "model_zmax":
            if (z_f == z_f) and (z_tau == z_tau):
                score = max(abs(z_f), abs(z_tau))
            else:
                score = float("nan")
        else:
            score = score_q50

        rows_out.append({
            "event_id": eid,
            "posterior_format": er.posterior_format,
            "weight": er.weight,
            "N": len(df_s),
            "score_q50": score_q50,
            "score": score,
            "score_type": args.score,
            "sigma_f": sigma_f,
            "sigma_tau": sigma_tau,
            "z0": z0,
            "pred_delta_f": pred_df,
            "pred_delta_tau": pred_dt,
            "ucm_f220_q50": (er.meta.get("ucm_f220_q50") or ""),
            "ucm_df_raw": (er.meta.get("ucm_df_raw") or ""),
            "ucm_dt_raw": (er.meta.get("ucm_dt_raw") or ""),
            "ucm_clipped_df": (er.meta.get("ucm_clipped_df") or ""),
            "ucm_clipped_dt": (er.meta.get("ucm_clipped_dt") or ""),
            "z_pred_f": z_f,
            "z_pred_tau": z_tau,
            "model": model_name,
            **{f"delta_f_{k}": v for k, v in s_df.items()},
            **{f"delta_tau_{k}": v for k, v in s_dt.items()},
        })

        global_df_medians.append(med_df)
        global_dt_medians.append(med_dt)
        global_weights.append(er.weight)

    # Global aggregate (weighted mean of per-event medians)
    df_hat = weighted_mean(global_df_medians, global_weights)
    dt_hat = weighted_mean(global_dt_medians, global_weights)

    # Bootstrap CI over events
    rng_seed = int(args.seed)
    rnd = (rng_seed * 1103515245 + 12345) & 0x7FFFFFFF

    def rand_int(n: int) -> int:
        nonlocal rnd
        rnd = (rnd * 1103515245 + 12345) & 0x7FFFFFFF
        return rnd % n

    B = int(args.B)
    boot_df = []
    boot_dt = []
    nE = len(global_df_medians)
    for _ in range(B):
        idxs = [rand_int(nE) for _ in range(nE)]
        vals_df = [global_df_medians[i] for i in idxs]
        vals_dt = [global_dt_medians[i] for i in idxs]
        wts = [global_weights[i] for i in idxs]
        boot_df.append(weighted_mean(vals_df, wts))
        boot_dt.append(weighted_mean(vals_dt, wts))

    s_boot_df = summarize_samples(boot_df, q_global)
    s_boot_dt = summarize_samples(boot_dt, q_global)

    global_out = {
        "bench": args.bench,
        "mode": args.mode,
        "score_type": args.score,
        "model": model_name,
        "model_params": model_params,
        "N_events": len(bench_ids),
        "delta_f_hat": df_hat,
        "delta_tau_hat": dt_hat,
        "delta_f_ci": s_boot_df,
        "delta_tau_ci": s_boot_dt,
    }
    # Attach optional baseline const + metrics (useful for KIT calibration)
    global_out["baseline_const"] = baseline

    def _rows_metrics(rows: list[dict], df0: float, dt0: float) -> dict:
        nll_sum = 0.0
        nll_sum_w = 0.0
        zmax = 0.0
        used = 0
        for rr in rows:
            try:
                med_df = float(rr.get("delta_f_q50"))
                med_dt = float(rr.get("delta_tau_q50"))
                sigma_f = float(rr.get("sigma_f"))
                sigma_t = float(rr.get("sigma_tau"))
                w = float(rr.get("weight", 1.0))
            except Exception:
                continue
            if not (math.isfinite(med_df) and math.isfinite(med_dt) and math.isfinite(sigma_f) and math.isfinite(sigma_t)):
                continue
            sigma_f = max(sigma_f, 1e-12)
            sigma_t = max(sigma_t, 1e-12)
            zf = (df0 - med_df) / sigma_f
            zt = (dt0 - med_dt) / sigma_t
            if not (math.isfinite(zf) and math.isfinite(zt)):
                continue
            nll = 0.5 * (zf*zf + zt*zt) + math.log(sigma_f) + math.log(sigma_t)
            nll_sum += nll
            nll_sum_w += w * nll
            zmax = max(zmax, abs(zf), abs(zt))
            used += 1
        return {"nll_sum": nll_sum, "nll_sum_w": nll_sum_w, "zmax": zmax, "events_used": used}

    baseline_metrics = None
    if isinstance(baseline, dict) and baseline.get("model") == "const":
        try:
            bdf = float(baseline["delta_f"])
            bdt = float(baseline["delta_tau"])
            baseline_metrics = _rows_metrics(rows_out, bdf, bdt)
        except Exception as e:
            baseline_metrics = {"error": str(e)}
    global_out["baseline_metrics"] = baseline_metrics

    # model_metrics for any model (computed from z_pred and sigma)
    try:
        nll_sum = 0.0
        nll_sum_w = 0.0
        zmax = 0.0
        used = 0
        for rr in rows_out:
            try:
                zf = float(rr.get("z_pred_f"))
                zt = float(rr.get("z_pred_tau"))
                sigma_f = float(rr.get("sigma_f"))
                sigma_t = float(rr.get("sigma_tau"))
                w = float(rr.get("weight", 1.0))
            except Exception:
                continue
            if not (math.isfinite(zf) and math.isfinite(zt) and math.isfinite(sigma_f) and math.isfinite(sigma_t)):
                continue
            sigma_f = max(sigma_f, 1e-12)
            sigma_t = max(sigma_t, 1e-12)
            nll = 0.5 * (zf*zf + zt*zt) + math.log(sigma_f) + math.log(sigma_t)
            nll_sum += nll
            nll_sum_w += w * nll
            zmax = max(zmax, abs(zf), abs(zt))
            used += 1
        clip_df = 0
        clip_dt = 0
        try:
            clip_df = sum(1 for rr in rows_out if str(rr.get("ucm_clipped_df", "")) == "1")
            clip_dt = sum(1 for rr in rows_out if str(rr.get("ucm_clipped_dt", "")) == "1")
        except Exception:
            pass
        global_out["model_metrics"] = {"nll_sum": nll_sum, "nll_sum_w": nll_sum_w, "zmax": zmax, "events_used": used, "ucm_clip_df": clip_df, "ucm_clip_dt": clip_dt}
    except Exception as e:
        global_out["model_metrics"] = {"error": str(e)}
    # save bootstrap samples (diagnostics)
    boot_dir = run_dir / "bootstrap"
    boot_dir.mkdir(exist_ok=True)
    (boot_dir / "boot_delta_f.csv").write_text("delta_f\n" + "\n".join(f"{v:.10g}" for v in boot_df) + "\n", encoding="utf-8")
    (boot_dir / "boot_delta_tau.csv").write_text("delta_tau\n" + "\n".join(f"{v:.10g}" for v in boot_dt) + "\n", encoding="utf-8")

    # plots: bootstrap histograms
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(boot_df, bins=50)
        plt.xlabel("delta_f (bootstrap)")
        plt.ylabel("count")
        plt.title(f"{args.bench} bootstrap delta_f (B={B})")
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "boot_delta_f_hist.png", dpi=160)
        plt.close()

        plt.figure()
        plt.hist(boot_dt, bins=50)
        plt.xlabel("delta_tau (bootstrap)")
        plt.ylabel("count")
        plt.title(f"{args.bench} bootstrap delta_tau (B={B})")
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "boot_delta_tau_hist.png", dpi=160)
        plt.close()
    except Exception as e:
        (run_dir / "plots" / "_plot_boot_hist_error.txt").write_text(str(e) + "\n", encoding="utf-8")

    # results_event.csv
    ev_csv = run_dir / "results_event.csv"
    with ev_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys()) if rows_out else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    # weights_report.csv
    write_weights_report(run_dir)

    # results_global.json
    (run_dir / "results_global.json").write_text(json.dumps(global_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # results contract artifacts
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)
    shutil.copy2(run_dir / "results_global.json", results_dir / "results_global.json")
    shutil.copy2(ev_csv, results_dir / "results_items.csv")

    # manifest
    write_manifest(run_dir / "_manifest.json", cfg, inputs={
        "events_csv": str(events_path),
        "bench_list": str(bench_list_path),
        "config": str(config_path),
        "config_exists": bool(config_path.exists()),
        "posteriors": posterior_files,
        "out_root": str(out_root),
        "seed": rng_seed,
        "B": B,
    })

    # report.txt
    rep = []
    rep.append("CVN-RD Engine (KIT-first) — RUN REPORT")
    rep.append(f"bench={args.bench}  mode={args.mode}  tag={args.tag}")
    rep.append(f"run_dir={run_dir}")
    rep.append("")
    rep.append(f"N_events={global_out['N_events']}")
    rep.append(f"delta_f_hat={df_hat:.6g}  delta_tau_hat={dt_hat:.6g}")
    rep.append(f"delta_f_CI={global_out['delta_f_ci']}")
    rep.append(f"delta_tau_CI={global_out['delta_tau_ci']}")
    append_weights_diagnostic_to_report(run_dir, rep, top_n=5)
    (run_dir / "report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")

    # summary.txt (KIT-style)
    scores: list[float] = []
    worst_score = -1.0
    worst_event = None

    for row in rows_out:
        eid = row.get("event_id", "")
        raw = row.get("score", row.get("score_q50", 0.0))
        try:
            score = float(raw) if raw is not None else 0.0
        except Exception:
            score = 0.0

        if not math.isfinite(score):
            continue

        scores.append(score)
        if worst_event is None or score > worst_score:
            worst_score, worst_event = score, eid

    scores.sort()

    def qv(qp: float) -> float:
        if not scores:
            return float("nan")
        n = len(scores)
        pos = (n - 1) * qp
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return scores[lo]
        frac = pos - lo
        return scores[lo] * (1 - frac) + scores[hi] * frac

    line = f"CVN_RD: median={qv(0.5):.6g}  q90={qv(0.9):.6g}  worst={worst_score:.6g}  worst_event={worst_event}"
    (run_dir / "summary.txt").write_text(line + "\n", encoding="utf-8")

    # worst10.txt (with weight)
    write_worst10(run_dir)

    # plot: global CI (q05/q50/q95)
    try:
        import matplotlib.pyplot as plt

        df_ci = global_out["delta_f_ci"]
        dt_ci = global_out["delta_tau_ci"]

        def pick(ci: dict, key: str, fallback: float) -> float:
            v = ci.get(key, None)
            return float(v) if v is not None else float(fallback)

        df_med = pick(df_ci, "q50", global_out["delta_f_hat"])
        df_lo  = pick(df_ci, "q05", df_med)
        df_hi  = pick(df_ci, "q95", df_med)

        dt_med = pick(dt_ci, "q50", global_out["delta_tau_hat"])
        dt_lo  = pick(dt_ci, "q05", dt_med)
        dt_hi  = pick(dt_ci, "q95", dt_med)

        labels = ["delta_f", "delta_tau"]
        meds = [df_med, dt_med]
        yerr_low = [df_med - df_lo, dt_med - dt_lo]
        yerr_high = [df_hi - df_med, dt_hi - dt_med]

        plt.figure()
        x = range(len(labels))
        plt.errorbar(x, meds, yerr=[yerr_low, yerr_high], fmt="o", capsize=6)
        plt.xticks(list(x), labels)
        plt.ylabel("value")
        plt.title(f"{args.bench} global 90% CI")
        plt.axhline(0.0, linewidth=1)
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "global_ci.png", dpi=160)
        plt.close()
    except Exception as e:
        (run_dir / "plots" / "_plot_global_ci_error.txt").write_text(str(e) + "\n", encoding="utf-8")

    print("DONE:", run_dir)

    # plot: score by event
    try:
        import matplotlib.pyplot as plt

        scores_evt = [(r["event_id"], float(r.get("score", r.get("score_q50", 0.0)))) for r in rows_out]
        scores_evt.sort(key=lambda x: x[1], reverse=True)
        labels = [e for e, _ in scores_evt]
        vals = [v for _, v in scores_evt]

        plt.figure()
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
        plt.ylabel(f"score ({args.score})")
        plt.title(f"{args.bench} score by event")
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "score_by_event.png", dpi=160)
        plt.close()
    except Exception as e:
        (run_dir / "plots" / "_plot_error.txt").write_text(str(e) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
