#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare FULL vs OUTER runs for SPARC pilot fits.

Usage:
  python compare_full_vs_outer.py --full RUNS/BENCH30_FULL/pilot_results.csv --outer RUNS/BENCH30_OUTER_SMART/pilot_results.csv --out RUNS/compare_full_vs_outer.txt --csv RUNS/compare_full_vs_outer.csv

No pandas required.
"""
# UCM-T Rotation Curves Utilities
# RC V12 (BENCH30 OUTER) — reference helper script
#
# Canonical reproducible bundle is archived on Zenodo:
# https://doi.org/10.5281/zenodo.18213329

from __future__ import annotations
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def read_results(path: Path) -> Dict[str, dict]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "Galaxy" not in r.fieldnames:
            raise SystemExit(f"CSV has no 'Galaxy' column: {path}")
        for row in r:
            g = (row.get("Galaxy") or "").strip()
            if not g:
                continue
            out[g] = row
    return out

def quantile(vals: List[float], q: float) -> float:
    v = sorted([x for x in vals if math.isfinite(x)])
    if not v:
        return float("nan")
    if q <= 0:
        return v[0]
    if q >= 1:
        return v[-1]
    # linear interpolation
    pos = (len(v) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    w = pos - lo
    return v[lo] * (1 - w) + v[hi] * w

def summary(vals: List[float]) -> str:
    v = [x for x in vals if math.isfinite(x)]
    if not v:
        return "N=0"
    return (
        f"N={len(v)}  "
        f"min={min(v):.3g}  med={quantile(v,0.5):.3g}  "
        f"q90={quantile(v,0.9):.3g}  worst={max(v):.3g}"
    )

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--full", required=True, help="Path to FULL pilot_results.csv")
    p.add_argument("--outer", required=True, help="Path to OUTER pilot_results.csv")
    p.add_argument("--out", default="", help="Write text report here (optional)")
    p.add_argument("--csv", default="", help="Write per-galaxy compare CSV here (optional)")
    p.add_argument("--top", type=int, default=10, help="How many items to show in top lists")
    args = p.parse_args()

    full = read_results(Path(args.full))
    outer = read_results(Path(args.outer))

    galaxies = sorted(set(full.keys()) | set(outer.keys()))
    missing_full = [g for g in galaxies if g not in full]
    missing_outer = [g for g in galaxies if g not in outer]

    rows = []
    chi_full = []
    chi_outer = []
    for g in galaxies:
        rf = full.get(g, {})
        ro = outer.get(g, {})
        cf = _f(rf.get("chi2_nu","")) if rf else float("nan")
        co = _f(ro.get("chi2_nu","")) if ro else float("nan")
        chi_full.append(cf)
        chi_outer.append(co)

        rcut = ro.get("Rcut_kpc","") or ro.get("Rcut_used","") or ""
        npts_o = ro.get("Npts","") or ro.get("npts","") or ""
        rows.append({
            "Galaxy": g,
            "chi2_nu_full": cf,
            "chi2_nu_outer": co,
            "delta": (cf - co) if (math.isfinite(cf) and math.isfinite(co)) else float("nan"),
            "ratio": (cf / co) if (math.isfinite(cf) and math.isfinite(co) and co>0) else float("nan"),
            "Rcut_kpc": rcut,
            "Npts_outer": npts_o,
        })

    # top improvements
    improvs = [r for r in rows if math.isfinite(r["delta"])]
    improvs.sort(key=lambda r: r["delta"], reverse=True)

    worst_outer = [r for r in rows if math.isfinite(r["chi2_nu_outer"])]
    worst_outer.sort(key=lambda r: r["chi2_nu_outer"], reverse=True)

    lines: List[str] = []
    lines.append("FULL vs OUTER comparison")
    lines.append(f"FULL : {args.full}")
    lines.append(f"OUTER: {args.outer}")
    lines.append("")
    lines.append("Summary chi2_nu")
    lines.append(f"  FULL : {summary([r['chi2_nu_full'] for r in rows])}")
    lines.append(f"  OUTER: {summary([r['chi2_nu_outer'] for r in rows])}")
    lines.append("")
    if missing_full:
        lines.append("Missing in FULL ("+str(len(missing_full))+"):")
        lines.append("  "+", ".join(missing_full))
        lines.append("")
    if missing_outer:
        lines.append("Missing in OUTER ("+str(len(missing_outer))+"):")
        lines.append("  "+", ".join(missing_outer))
        lines.append("")

    lines.append(f"Top improvements by delta (FULL-OUTER), top {args.top}")
    for r in improvs[:args.top]:
        lines.append(f"  {r['Galaxy']:10s}  full={r['chi2_nu_full']:.3g}  outer={r['chi2_nu_outer']:.3g}  "
                     f"delta={r['delta']:.3g}  ratio={r['ratio']:.3g}  Rcut={r['Rcut_kpc']}  N_outer={r['Npts_outer']}")
    lines.append("")
    lines.append(f"Worst OUTER chi2_nu, top {args.top}")
    for r in worst_outer[:args.top]:
        lines.append(f"  {r['Galaxy']:10s}  outer={r['chi2_nu_outer']:.3g}  full={r['chi2_nu_full']:.3g}  "
                     f"Rcut={r['Rcut_kpc']}  N_outer={r['Npts_outer']}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"[written] {args.out}")

    if args.csv:
        with Path(args.csv).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Galaxy","chi2_nu_full","chi2_nu_outer","delta","ratio","Rcut_kpc","Npts_outer"])
            w.writeheader()
            for r in rows:
                w.writerow({
                    "Galaxy": r["Galaxy"],
                    "chi2_nu_full": ("" if not math.isfinite(r["chi2_nu_full"]) else f"{r['chi2_nu_full']:.12g}"),
                    "chi2_nu_outer": ("" if not math.isfinite(r["chi2_nu_outer"]) else f"{r['chi2_nu_outer']:.12g}"),
                    "delta": ("" if not math.isfinite(r["delta"]) else f"{r['delta']:.12g}"),
                    "ratio": ("" if not math.isfinite(r["ratio"]) else f"{r['ratio']:.12g}"),
                    "Rcut_kpc": r["Rcut_kpc"],
                    "Npts_outer": r["Npts_outer"],
                })
        print(f"[written] {args.csv}")

if __name__ == "__main__":
    main()
