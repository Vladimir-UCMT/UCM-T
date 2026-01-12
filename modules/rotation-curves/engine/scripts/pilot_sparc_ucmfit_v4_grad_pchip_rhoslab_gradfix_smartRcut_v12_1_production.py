#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pilot-test for UCM-Engine (rotation curves) on SPARC Rotmod files.

What it does (one command):
1) Downloads SPARC Rotmod_LTG.zip (if not present) and unzips it.
2) Computes ONE reference UCM-Engine v(R) curve with fixed regime params (L, k0, etc.).
3) For each selected galaxy, fits only 1–2 nuisance parameters:
   - A  : velocity scaling (km/s per model unit)
   - Rs : radial scaling (kpc per model unit)  [optional but enabled by default]
   using weighted least squares (grid over Rs, analytic A).
4) Exports:
   - pilot_results.csv
   - pilot_table.tex (LaTeX table with booktabs)
   - pilot_fits.pdf (multi-panel figure)

Usage (examples):
  python pilot_sparc_ucmfit.py --galaxies NGC3198,NGC2403,UGC128,DDO154,F563-V2,IC2574
  python pilot_sparc_ucmfit.py --k0 0.30 --out pilot_k0_030

Requirements:
  - numpy, matplotlib
  - Your UCM engine script: ucm_rotation_curve_2d_sparse_v3_7_2_fixed.py must be
    in the SAME directory as this file, or importable on PYTHONPATH.
"""
# UCM-T Rotation Curves Benchmark Runner
# RC V12 (BENCH30 OUTER) — frozen reference runner script
#
# Canonical reproducible bundle (code + data + reference outputs) is archived on Zenodo:
# https://doi.org/10.5281/zenodo.18213329

from __future__ import annotations
import argparse
import re
import csv
import io
import math
import os
import sys
import zipfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
def smooth_box(y, w=11):
    """Simple moving-average smoothing (odd window). Returns array same length as y."""
    y = np.asarray(y, dtype=float)
    w = int(w)
    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    ypad = np.pad(y, (w//2, w//2), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")

import matplotlib.pyplot as plt


# -----------------------------
# SPARC download / IO
# -----------------------------

SPARC_URLS = [
    # Primary (as in docs)
    "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
    # Mirror (Case)
    "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip",
]

@dataclass
class GalaxyRC:
    name: str
    R_kpc: np.ndarray
    V_kms: np.ndarray
    eV_kms: np.ndarray

def download_if_needed(zip_path: Path) -> None:
    if zip_path.exists() and zip_path.stat().st_size > 10_000:
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in SPARC_URLS:
        try:
            print(f"[download] {url}")
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            zip_path.write_bytes(data)
            print(f"[download] saved -> {zip_path} ({zip_path.stat().st_size/1024:.1f} kB)")
            return
        except Exception as e:
            last_err = e
            print(f"[download] failed: {e}")
    raise RuntimeError(f"Could not download Rotmod_LTG.zip. Last error: {last_err}")

def unzip_rotmod(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    # Many zips unpack as files directly; normalize to a folder path
    # Heuristic: if files are in a subfolder, use it; else use out_dir.
    subfolders = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(subfolders) == 1:
        return subfolders[0]
    return out_dir

def read_rotmod_file(path: Path) -> GalaxyRC:
    """
    SPARC Rotmod file format typically has commented header lines starting with '#'
    and columns like:
      Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
    We only need first 3 columns.
    """
    name = path.stem.replace("_rotmod", "")
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                r, v, ev = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            if not (np.isfinite(r) and np.isfinite(v) and np.isfinite(ev)):
                continue
            if r <= 0 or ev <= 0:
                continue
            rows.append((r, v, ev))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    arr = np.array(rows, float)
    return GalaxyRC(name=name, R_kpc=arr[:, 0], V_kms=arr[:, 1], eV_kms=arr[:, 2])


def _normalize_sparc_name(nm: str) -> str:
    """
    SPARC Rotmod filenames often use zero-padded catalogs, e.g. UGC0128, DDO069.
    This helper makes common user inputs robust (e.g. 'UGC128' -> 'UGC0128').
    """
    s = nm.strip()
    m = re.match(r"^(UGC)\s*0*([0-9]{1,4})$", s, flags=re.IGNORECASE)
    if m:
        return f"UGC{int(m.group(2)):05d}"
    m = re.match(r"^(DDO)\s*0*([0-9]{1,3})$", s, flags=re.IGNORECASE)
    if m:
        return f"DDO{int(m.group(2)):03d}"
    m = re.match(r"^(NGC)\s*0*([0-9]{1,4})$", s, flags=re.IGNORECASE)
    if m:
        return f"NGC{int(m.group(2)):04d}"
    return s

def _available_galaxy_stems(rotmod_dir: Path) -> List[str]:
    stems = []
    for p in rotmod_dir.glob("*_rotmod.dat"):
        stems.append(p.stem.replace("_rotmod", ""))
    return sorted(set(stems))

def load_galaxies(rotmod_dir: Path, names: List[str]) -> List[GalaxyRC]:
    out = []
    avail = _available_galaxy_stems(rotmod_dir)
    for nm_in in names:
        nm = _normalize_sparc_name(nm_in)
        # SPARC uses "F563-V2_rotmod.dat" style naming; normalize
        candidates = [
            rotmod_dir / f"{nm}_rotmod.dat",
            rotmod_dir / f"{nm.upper()}_rotmod.dat",
            rotmod_dir / f"{nm.lower()}_rotmod.dat",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            # try fuzzy: replace spaces and dots
            nm2 = nm.replace(" ", "").replace(".", "")
            for p in rotmod_dir.glob("*_rotmod.dat"):
                if p.stem.replace("_rotmod", "").replace(" ", "").replace(".", "") == nm2:
                    found = p
                    break
        if found is None:
            # suggest close matches
            try:
                import difflib
                sugg = difflib.get_close_matches(nm, avail, n=8, cutoff=0.5)
            except Exception:
                sugg = []
            hint = ("\nClosest available names: " + ", ".join(sugg)) if sugg else ""
            raise FileNotFoundError(
                f"Galaxy file not found for '{nm_in}' (normalized to '{nm}') in {rotmod_dir}." + hint +
                "\nTip: SPARC often uses zero-padded names like UGC0128, DDO069."
            )
        out.append(read_rotmod_file(found))
    return out


# -----------------------------
# UCM-Engine model curve
# -----------------------------

@dataclass
class EngineCurve:
    R_model: np.ndarray  # model units
    v_model: np.ndarray  # model units (dimensionless)
    
def compute_engine_curve(*, engine_py: Path, L: float, k0: float,
                         Rmax: float, Zmax: float, Nr: int, Nz: int,
                         Rd: float, hz: float, nu: float,
                         mode: str = "rho_weighted",
                         zcut_factor: float = 2.0,
                          grad_method: str = "fd2d",
                         obs_smooth: str = "none",
                         phi_eff_smooth: str = "",
                         smooth_w0: int = 11,
                         smooth_w1: int = 31,
                         smooth_tail_frac: float = 0.7,
                         sg_poly: int = 2,
                         force_attractive: bool | None = None,
                         force_attractive_eps: float = 0.0,
                         cache_dir: Path | None = None) -> EngineCurve:
    """
    Imports the user's UCM engine module and computes v(R) ONCE.
    Caches to npz if cache_dir is provided.
    """
    if not str(phi_eff_smooth).strip():
        phi_eff_smooth = obs_smooth

    cache_key = (
        f"engine_L{L:g}_k0{k0:g}_R{Rmax:g}_Z{Zmax:g}_Nr{Nr}_Nz{Nz}_"
        f"Rd{Rd:g}_hz{hz:g}_nu{nu:.3e}_{mode}_zc{zcut_factor:g}_{grad_method}_"
        f"os{obs_smooth}_pe{phi_eff_smooth}_w0{smooth_w0}_w1{smooth_w1}_tf{smooth_tail_frac:.3g}_p{sg_poly}"
        f"_fa{1 if force_attractive else 0}_fe{force_attractive_eps:g}.npz"
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            d = np.load(cache_path)
            return EngineCurve(R_model=d["R"], v_model=d["v"])

    # dynamic import from path
    import importlib.util
    spec = importlib.util.spec_from_file_location("ucm_engine_rc", str(engine_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import engine from {engine_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    R, Z, Phi, U = mod.solve_twofield(Rmax=Rmax, Zmax=Zmax, Nr=Nr, Nz=Nz,
                                     L=float(L), k0=float(k0), Rd=float(Rd), hz=float(hz), nu=float(nu))
    if mode in ("rho_weighted", "rho_slab"):
        import inspect
        kw = {}
        try:
            sig = inspect.signature(mod.v_from_rho_weighted)
            if "grad_method" in sig.parameters:
                kw["grad_method"] = str(grad_method)
            if "obs_smooth" in sig.parameters:
                kw["obs_smooth"] = str(obs_smooth)
            if "phi_eff_smooth" in sig.parameters:
                kw["phi_eff_smooth"] = str(phi_eff_smooth)
            if "smooth_w0" in sig.parameters:
                kw["smooth_w0"] = int(smooth_w0)
            if "smooth_w1" in sig.parameters:
                kw["smooth_w1"] = int(smooth_w1)
            if "smooth_tail_frac" in sig.parameters:
                kw["smooth_tail_frac"] = float(smooth_tail_frac)
            if "grad_method" in sig.parameters:
                kw["grad_method"] = str(grad_method)
            if "sg_poly" in sig.parameters:
                kw["sg_poly"] = int(sg_poly)
            if "force_attractive" in sig.parameters and force_attractive is not None:
                kw["force_attractive"] = bool(force_attractive)
            if "force_attractive_eps" in sig.parameters:
                kw["force_attractive_eps"] = float(force_attractive_eps)
        except Exception:
            kw = {}
        # rho_slab: optionally restrict z-range for rho-weighted reduction
        else:
            import inspect
            kw2 = dict(kw)
            try:
                sigw = inspect.signature(mod.v_from_rho_weighted)
                if "zcut_factor" in sigw.parameters:
                    kw2["zcut_factor"] = float(zcut_factor)
                else:
                    print("[engine] NOTE: v_from_rho_weighted() has no zcut_factor -> rho_slab falls back to rho_weighted (no z-cut).")
            except Exception:
                print("[engine] NOTE: could not inspect v_from_rho_weighted() -> rho_slab falls back to rho_weighted (no z-cut).")
            v, *_ = mod.v_from_rho_weighted(R, Z, Phi, Rd=float(Rd), hz=float(hz), **kw2)
    elif mode == "midline":
        import inspect
        kw = {}
        try:
            sig = inspect.signature(mod.v_from_midline)
            if "grad_method" in sig.parameters:
                kw["grad_method"] = str(grad_method)
            # keep the legacy default nsl_mid=3 but allow override if present
            if "nsl_mid" in sig.parameters:
                kw["nsl_mid"] = 3
        except Exception:
            kw = {}
        v, *_ = mod.v_from_midline(R, Phi, **kw)
    else:
        raise ValueError("mode must be 'rho_weighted', 'rho_slab', or 'midline'")

    # clean nans
    v = np.asarray(v, float)
    R = np.asarray(R, float)
    m = np.isfinite(R) & np.isfinite(v) & (R > 0)
    R, v = R[m], v[m]

    if cache_dir is not None:
        np.savez(cache_path, R=R, v=v)
    return EngineCurve(R_model=R, v_model=v)


# -----------------------------
# Fitting (grid over Rs, analytic A)
# -----------------------------

@dataclass
class FitResult:
    name: str
    npts: int
    Rmin: float
    Rmax: float
    Rd: float
    hz: float
    # Rcut actually used for the fit (kpc)
    Rcut: float
    # "physical" Rcut from the selected mode (kpc), before any fallback policy
    Rcut_phys: float = float("nan")
    # number of data points kept after the actually used cut
    N_after: int = 0
    # number of data points kept after the physical cut
    N_after_phys: int = 0
    # 1 if N_after_phys < Rcut_min_points (i.e., tail is sparse under the physical cut)
    outer_lowN: int = 0
    # effective fallback policy used (legacy/lower/keep/skip)
    rcut_policy: str = "legacy"

    # selection window: "tail" (R>=Rcut) or "ring" (Rcut<=R<Rcut2)
    outer_window: str = "tail"
    Rcut2: float = float("nan")
    Rcut2_phys: float = float("nan")

    # optional two-zone radial scaling (kpc per model unit)
    Rs_in: float = float("nan")
    Rs_out: float = float("nan")
    Rbreak_kpc: float = float("nan")

    A: float = float("nan")
    Rs: float = float("nan")
    chi2: float = float("nan")
    chi2_nu: float = float("nan")

def interp1_safe(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    # x must be sorted
    return np.interp(xq, x, y, left=np.nan, right=np.nan)

def choose_rcut_with_policy(R_kpc: np.ndarray, Rd_kpc: float, args: argparse.Namespace) -> Tuple[float, int, float, int, int, str, bool]:
    """
    Choose Rcut for each galaxy.

    Returns:
      (Rcut_used_kpc, N_after_used, Rcut_phys_kpc, N_after_phys, outer_lowN, policy_used, skip_galaxy)

    - "physical" cut Rcut_phys comes from --Rcut-mode (kpc / 2Rd / max(5,2Rd)) and is clamped to <= Rmax - margin.
    - If N_after_phys < --Rcut-min-points, we apply a fallback policy:
        * lower: reduce Rcut to keep at least Nmin points (may pull points from inner region)
        * keep : keep the physical cut even if low-N (marks outer_lowN=1)
        * skip : skip this galaxy (marks outer_lowN=1)
      For "big disks" (Rd >= --Rcut-big-Rd), we apply --Rcut-big-policy instead of --Rcut-fallback.
    """
    R = np.asarray(R_kpc, dtype=float)
    m = np.isfinite(R) & (R > 0)
    R = np.sort(R[m])
    n_tot = int(R.size)
    if n_tot == 0:
        return 0.0, 0, float("nan"), 0, 1, "skip", True

    Rmax = float(R[-1])
    margin = float(getattr(args, "Rcut_margin", 0.5))
    nmin = int(getattr(args, "Rcut_min_points", 6))
    nmin = max(1, nmin)

    mode = str(getattr(args, "Rcut_mode", "legacy"))

    def n_after(cut: float) -> int:
        i0 = int(np.searchsorted(R, float(cut), side="left"))
        return int(n_tot - i0)

    # physical cut (before fallback policy)
    if mode == "legacy":
        rcut_phys = float(getattr(args, "Rcut", 0.0))
    elif mode == "kpc":
        rcut_phys = float(getattr(args, "Rcut_kpc", 0.0))
    else:
        kRd = float(getattr(args, "Rcut_kRd", 2.0))
        Rd_val = float(Rd_kpc) if (Rd_kpc is not None and np.isfinite(Rd_kpc)) else 0.0
        rcut_phys = kRd * Rd_val
        if mode == "max(5,2Rd)":
            rcut_phys = max(5.0, rcut_phys)

    # clamp to Rmax - margin
    rcut_phys = min(float(rcut_phys), float(Rmax - margin))
    rcut_phys = max(0.0, float(rcut_phys))

    kept_phys = n_after(rcut_phys)
    outer_lowN = 1 if (mode != "legacy" and kept_phys < nmin) else 0

    # no fallback needed / legacy mode
    if mode == "legacy" or kept_phys >= nmin:
        return float(rcut_phys), int(kept_phys), float(rcut_phys), int(kept_phys), int(outer_lowN), ("legacy" if mode == "legacy" else "phys"), False

    # choose effective fallback policy (patch for big disks)
    fallback = str(getattr(args, "Rcut_fallback", "lower"))
    big_Rd = float(getattr(args, "Rcut_big_Rd", 6.0))
    big_policy = str(getattr(args, "Rcut_big_policy", "keep"))
    if big_Rd > 0 and np.isfinite(Rd_kpc) and float(Rd_kpc) >= big_Rd:
        fallback = big_policy

    policy_used = fallback

    skip = False
    rcut_used = float(rcut_phys)
    kept_used = int(kept_phys)

    if fallback == "lower":
        # reduce Rcut to keep at least nmin points.
        # BUT: if the whole dataset has <= nmin points, we cannot satisfy the constraint anyway.
        # In that case, do NOT collapse OUTER into FULL (Rcut=0); keep the physical cut.
        if n_tot <= nmin:
            rcut_used = float(rcut_phys)
            kept_used = int(kept_phys)
            policy_used = "keep_totalN"
        else:
            rcut_used = float(R[-nmin])
            kept_used = n_after(rcut_used)
            policy_used = "lower"
    elif fallback == "keep":
        # keep the physical cut even if it leaves < nmin points
        rcut_used = float(rcut_phys)
        kept_used = int(kept_phys)
        policy_used = "keep"
    elif fallback == "skip":
        skip = True
        policy_used = "skip"
    else:
        # unknown -> be conservative
        rcut_used = float(rcut_phys)
        kept_used = int(kept_phys)
        policy_used = "keep"

    return float(rcut_used), int(kept_used), float(rcut_phys), int(kept_phys), int(outer_lowN), str(policy_used), bool(skip)


def fit_one_galaxy(rc: GalaxyRC, eng: EngineCurve,
                   Rs_grid: np.ndarray,
                   *, Rd_used: float = float('nan'), hz_used: float = float('nan'),
                   fit_Rs: bool = True,
                   fit_two_Rs: bool = False,
                   two_Rs_fixed: bool = False,
                   two_Rs_fixed_in: bool = False,
                   two_Rs_Rs_in: float = 0.5,
                   two_Rs_Rs_out: float = 2.0,
                   two_Rs_Rd_min: float = 0.0,
                   Rs_break_mode: str = "2Rd",
                   Rs_break_kRd: float = 4.0,
                   Rs_break_kpc: float = 0.0,
                   sigma_floor: float = 0.0,
                   Rcut: float = 0.0,
                   Rcut_phys: float = float("nan"),
                   N_after: int = 0,
                   N_after_phys: int = 0,
                   outer_lowN: int = 0,
                   rcut_policy: str = "legacy",
                   outer_window: str = "tail",
                   Rcut2: float = float("nan"),
                   Rcut2_phys: float = float("nan")) -> FitResult:
    # optional radial window selection (outer-curve focus)
    mfit = np.isfinite(rc.R_kpc) & np.isfinite(rc.V_kms) & np.isfinite(rc.eV_kms)

    win = str(outer_window or "tail").strip().lower()
    if win == "ring":
        # ring window: [Rcut, Rcut2)
        if (Rcut and float(Rcut) > 0) and (np.isfinite(Rcut2) and float(Rcut2) > 0):
            lo = float(Rcut)
            hi = float(Rcut2)
            if hi <= lo:
                raise RuntimeError(f"Invalid ring window for {rc.name}: Rcut2<=Rcut (Rcut={lo}, Rcut2={hi})")
            mfit = mfit & (rc.R_kpc >= lo) & (rc.R_kpc < hi)
        elif Rcut and float(Rcut) > 0:
            # fallback to tail if Rcut2 is not set
            mfit = mfit & (rc.R_kpc >= float(Rcut))
            win = "tail"
    else:
        # tail window: [Rcut, +inf)
        if Rcut and float(Rcut) > 0:
            mfit = mfit & (rc.R_kpc >= float(Rcut))
        win = "tail"

    R = rc.R_kpc[mfit]
    V = rc.V_kms[mfit]
    eV = rc.eV_kms[mfit]
    if R.size < 3:
        raise RuntimeError(f"Not enough points in window for {rc.name}: window={win} N={R.size}. Try adjusting Rcut / Rcut2.")
    # error floor to avoid unrealistically huge chi^2 when formal errors are tiny
    sf = float(sigma_floor) if sigma_floor is not None else 0.0
    if sf > 0:
        eV_eff = np.sqrt(eV*eV + sf*sf)
    else:
        eV_eff = eV
    w = 1.0 / (eV_eff * eV_eff)

        # Ensure engine curve monotonic in R
    idx = np.argsort(eng.R_model)
    Rm = eng.R_model[idx]
    vm = eng.v_model[idx]

    best = None

    if fit_two_Rs and not fit_Rs:
        raise RuntimeError("--fit-two-Rs requires fitting Rs (do not use --no-fit-Rs).")

    # Break radius in kpc (used only if fit_two_Rs)
    Rbreak = float("nan")
    if fit_two_Rs:
        mode = (Rs_break_mode or "2Rd")
        if mode == "kpc":
            Rbreak = float(Rs_break_kpc)
        elif mode == "max(5,2Rd)":
            Rbreak = max(5.0, float(Rs_break_kRd) * float(Rd_used))
        else:  # "2Rd"
            Rbreak = float(Rs_break_kRd) * float(Rd_used)
        if not np.isfinite(Rbreak) or Rbreak <= 0:
            raise RuntimeError(f"Invalid Rbreak for {rc.name}: {Rbreak}")

    if not fit_Rs:
        Rs_grid = np.array([1.0], float)

    if not fit_two_Rs:
        for Rs in Rs_grid:
            if Rs <= 0:
                continue
            rq = R / Rs
            vshape = interp1_safe(Rm, vm, rq)
            if not np.all(np.isfinite(vshape)):
                continue

            denom = np.sum(w * vshape * vshape)
            if denom <= 0:
                continue
            A = np.sum(w * V * vshape) / denom

            resid = V - A * vshape
            chi2 = float(np.sum(w * resid * resid))
            npar = 2 if fit_Rs else 1
            dof = max(1, int(R.size) - npar)
            chi2_nu = chi2 / dof

            if best is None or chi2 < best.chi2:
                best = FitResult(
                    name=rc.name,
                    npts=int(R.size),
                    Rmin=float(np.min(R)),
                    Rmax=float(np.max(R)),
                    Rd=float(Rd_used),
                    hz=float(hz_used),
                    Rcut=float(Rcut),
                    Rcut_phys=float(Rcut_phys),
                    N_after=int(N_after),
                    N_after_phys=int(N_after_phys),
                    outer_lowN=int(outer_lowN),
                    rcut_policy=str(rcut_policy),
                    outer_window=str(outer_window or "tail"),
                    Rcut2=float(Rcut2),
                    Rcut2_phys=float(Rcut2_phys),
                    Rs_in=float("nan"),
                    Rs_out=float(Rs),
                    Rbreak_kpc=float("nan"),
                    A=float(A),
                    Rs=float(Rs),
                    chi2=float(chi2),
                    chi2_nu=float(chi2_nu),
                )
    else:
        if two_Rs_fixed:
            Rs_out = float(two_Rs_Rs_out)
            Rs_in  = float(two_Rs_Rs_in)
            if Rs_out <= 0 or Rs_in <= 0:
                raise RuntimeError(f"Invalid fixed Rs values for {rc.name}: Rs_in={Rs_in}, Rs_out={Rs_out}")
            # Apply two-Rs only for sufficiently large disks, if requested
            Rbreak_eff = float(Rbreak)
            if two_Rs_Rd_min and np.isfinite(Rd_used) and float(Rd_used) < float(two_Rs_Rd_min):
                # disable inner scale for small disks -> revert to single-Rs behavior
                Rs_in = Rs_out
                Rbreak_eff = 0.0
            Rs_eff = np.where(R < Rbreak_eff, Rs_in, Rs_out)
            rq = R / Rs_eff
            vshape = interp1_safe(Rm, vm, rq)
            if np.all(np.isfinite(vshape)):
                denom = np.sum(w * vshape * vshape)
                if denom > 0:
                    A = np.sum(w * V * vshape) / denom
                    resid = V - A * vshape
                    chi2 = float(np.sum(w * resid * resid))
                    npar = 1  # only A is fit; Rs_in/Rs_out are fixed global constants
                    dof = max(1, int(R.size) - npar)
                    chi2_nu = chi2 / dof
                    best = FitResult(
                        name=rc.name,
                        npts=int(R.size),
                        Rmin=float(np.min(R)),
                        Rmax=float(np.max(R)),
                        Rd=float(Rd_used),
                        hz=float(hz_used),
                        Rcut=float(Rcut),
                        Rcut_phys=float(Rcut_phys),
                        N_after=int(N_after),
                        N_after_phys=int(N_after_phys),
                        outer_lowN=int(outer_lowN),
                        rcut_policy=str(rcut_policy),
                        outer_window=str(outer_window or "tail"),
                        Rcut2=float(Rcut2),
                        Rcut2_phys=float(Rcut2_phys),
                        Rs_in=float(Rs_in),
                        Rs_out=float(Rs_out),
                        Rbreak_kpc=float(Rbreak_eff),
                        A=float(A),
                        Rs=float(Rs_out),
                        chi2=float(chi2),
                        chi2_nu=float(chi2_nu),
                    )
        elif two_Rs_fixed_in:
            Rs_in = float(two_Rs_Rs_in)
            if Rs_in <= 0:
                raise RuntimeError(f"Invalid fixed Rs_in for {rc.name}: Rs_in={Rs_in}")
            # per-galaxy Rs_out scan on the standard Rs grid
            # optional: disable for small disks
            if two_Rs_Rd_min and np.isfinite(Rd_used) and float(Rd_used) < float(two_Rs_Rd_min):
                # behave like single-Rs scan
                for Rs_out in Rs_grid:
                    if Rs_out <= 0: continue
                    rq = R / Rs_out
                    vshape = interp1_safe(Rm, vm, rq)
                    if not np.all(np.isfinite(vshape)): continue
                    denom = np.sum(w * vshape * vshape)
                    if denom <= 0: continue
                    A = np.sum(w * V * vshape) / denom
                    resid = V - A * vshape
                    chi2 = float(np.sum(w * resid * resid))
                    npar = 2  # A + Rs_out
                    dof = max(1, int(R.size) - npar)
                    chi2_nu = chi2 / dof
                    if best is None or chi2 < best.chi2:
                        best = FitResult(name=rc.name, npts=int(R.size), Rmin=float(np.min(R)), Rmax=float(np.max(R)),
                                         Rd=float(Rd_used), hz=float(hz_used), Rcut=float(Rcut), Rcut_phys=float(Rcut_phys),
                                         N_after=int(N_after), N_after_phys=int(N_after_phys), outer_lowN=int(outer_lowN),
                                         rcut_policy=str(rcut_policy), outer_window=str(outer_window or 'tail'),
                                         Rcut2=float(Rcut2), Rcut2_phys=float(Rcut2_phys),
                                         Rs_in=float(Rs_out), Rs_out=float(Rs_out), Rbreak_kpc=0.0,
                                         A=float(A), Rs=float(Rs_out), chi2=float(chi2), chi2_nu=float(chi2_nu))
            else:
                Rbreak_eff = float(Rbreak)
                for Rs_out in Rs_grid:
                    if Rs_out <= 0: continue
                    Rs_eff = np.where(R < Rbreak_eff, Rs_in, Rs_out)
                    rq = R / Rs_eff
                    vshape = interp1_safe(Rm, vm, rq)
                    if not np.all(np.isfinite(vshape)): continue
                    denom = np.sum(w * vshape * vshape)
                    if denom <= 0: continue
                    A = np.sum(w * V * vshape) / denom
                    resid = V - A * vshape
                    chi2 = float(np.sum(w * resid * resid))
                    npar = 2  # A + Rs_out
                    dof = max(1, int(R.size) - npar)
                    chi2_nu = chi2 / dof
                    if best is None or chi2 < best.chi2:
                        best = FitResult(name=rc.name, npts=int(R.size), Rmin=float(np.min(R)), Rmax=float(np.max(R)),
                                         Rd=float(Rd_used), hz=float(hz_used), Rcut=float(Rcut), Rcut_phys=float(Rcut_phys),
                                         N_after=int(N_after), N_after_phys=int(N_after_phys), outer_lowN=int(outer_lowN),
                                         rcut_policy=str(rcut_policy), outer_window=str(outer_window or 'tail'),
                                         Rcut2=float(Rcut2), Rcut2_phys=float(Rcut2_phys),
                                         Rs_in=float(Rs_in), Rs_out=float(Rs_out), Rbreak_kpc=float(Rbreak_eff),
                                         A=float(A), Rs=float(Rs_out), chi2=float(chi2), chi2_nu=float(chi2_nu))

        else:
            for Rs_out in Rs_grid:
                if Rs_out <= 0:
                    continue
                for Rs_in in Rs_grid:
                    if Rs_in <= 0:
                        continue

                    Rs_eff = np.where(R < Rbreak_eff, Rs_in, Rs_out)
                    rq = R / Rs_eff
                    vshape = interp1_safe(Rm, vm, rq)
                    if not np.all(np.isfinite(vshape)):
                        continue

                    denom = np.sum(w * vshape * vshape)
                    if denom <= 0:
                        continue
                    A = np.sum(w * V * vshape) / denom

                    resid = V - A * vshape
                    chi2 = float(np.sum(w * resid * resid))
                    npar = 3
                    dof = max(1, int(R.size) - npar)
                    chi2_nu = chi2 / dof

                    if best is None or chi2 < best.chi2:
                        best = FitResult(
                            name=rc.name,
                            npts=int(R.size),
                            Rmin=float(np.min(R)),
                            Rmax=float(np.max(R)),
                            Rd=float(Rd_used),
                            hz=float(hz_used),
                            Rcut=float(Rcut),
                            Rcut_phys=float(Rcut_phys),
                            N_after=int(N_after),
                            N_after_phys=int(N_after_phys),
                            outer_lowN=int(outer_lowN),
                            rcut_policy=str(rcut_policy),
                            outer_window=str(outer_window or "tail"),
                            Rcut2=float(Rcut2),
                            Rcut2_phys=float(Rcut2_phys),
                            Rs_in=float(Rs_in),
                            Rs_out=float(Rs_out),
                            Rbreak_kpc=float(Rbreak_eff),
                            A=float(A),
                            Rs=float(Rs_out),
                            chi2=float(chi2),
                            chi2_nu=float(chi2_nu),
                        )

    return best


# -----------------------------
# Export: CSV + LaTeX + Figure
# -----------------------------

def write_csv(results: List[FitResult], path: Path) -> None:
    """Write per-galaxy fit results to CSV.

    Backward-compatible columns (used by compare_full_vs_outer.py):
      Galaxy, Npts, Rcut_kpc, chi2_nu

    Extra diagnostics for smart Rcut:
      Rcut_phys_kpc, N_after, N_after_phys, outer_lowN, rcut_policy
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Galaxy",
            "Npts",
            "Rmin_kpc",
            "Rmax_kpc",
            "Rd_kpc",
            "hz_kpc",
            "Rcut_kpc",
            "Rcut_phys_kpc",
            "N_after",
            "N_after_phys",
            "outer_lowN",
            "rcut_policy",
            "outer_window",
            "Rcut2_kpc",
            "Rcut2_phys_kpc",
            "A_kms",
            "Rs_kpc",
            "Rs_in_kpc",
            "Rs_out_kpc",
            "Rbreak_kpc",
            "chi2",
            "chi2_nu",
        ])
        for r in results:
            rcut_phys = getattr(r, "Rcut_phys", float("nan"))
            rs_in = getattr(r, "Rs_in", float("nan"))
            rs_out = getattr(r, "Rs_out", float("nan"))
            rbreak = getattr(r, "Rbreak_kpc", float("nan"))
            wr.writerow([
                r.name,
                int(r.npts),
                f"{r.Rmin:.6g}",
                f"{r.Rmax:.6g}",
                f"{r.Rd:.6g}",
                f"{r.hz:.6g}",
                f"{r.Rcut:.6g}",
                f"{(rcut_phys if np.isfinite(rcut_phys) else float('nan')):.6g}",
                int(getattr(r, "N_after", 0)),
                int(getattr(r, "N_after_phys", 0)),
                int(getattr(r, "outer_lowN", 0)),
                str(getattr(r, "rcut_policy", "")),
                str(getattr(r, "outer_window", "tail")),
                f"{(getattr(r, 'Rcut2', float('nan')) if np.isfinite(getattr(r, 'Rcut2', float('nan'))) else float('nan')):.6g}",
                f"{(getattr(r, 'Rcut2_phys', float('nan')) if np.isfinite(getattr(r, 'Rcut2_phys', float('nan'))) else float('nan')):.6g}",
                f"{r.A:.6g}",
                f"{r.Rs:.6g}",
                f"{(rs_in if np.isfinite(rs_in) else float('nan')):.6g}",
                f"{(rs_out if np.isfinite(rs_out) else float('nan')):.6g}",
                f"{(rbreak if np.isfinite(rbreak) else float('nan')):.6g}",
                f"{r.chi2:.6g}",
                f"{r.chi2_nu:.6g}",
            ])


def write_table_tex(results: List[FitResult], path: Path, fixed_params_line: str) -> None:
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Galaxy & $N$ & $R_{\min}$--$R_{\max}$ (kpc) & $R_d$ (kpc) & $A$ (km/s) & $R_s$ (kpc) & $\chi^2_\nu$ \\")
    lines.append(r"\midrule")
    for r in results:
        rr = f"{r.Rmin:.2g}--{r.Rmax:.2g}"
        lines.append(f"{r.name} & {r.npts:d} & {rr} & {r.Rd:.3g} & {r.A:.3g} & {r.Rs:.3g} & {r.chi2_nu:.3g} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"% " + fixed_params_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def plot_multipanel(galaxies: List[GalaxyRC],
                    results: Dict[str, FitResult],
                    engines: Dict[str, EngineCurve],
                    out_pdf: Path,
                    ncols: int = 3) -> None:
    """
    Multi-panel plot: data + best-fit model for each galaxy.
    Supports per-galaxy EngineCurve (e.g., when using SPARC Rdisk).
    """
    n = len(galaxies)
    ncols = max(1, int(ncols))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    if isinstance(axes, np.ndarray):
        ax_list = axes.ravel().tolist()
    else:
        ax_list = [axes]

    for j, rc in enumerate(galaxies):
        ax = ax_list[j]
        fr = results.get(rc.name)
        eng = engines.get(rc.name)
        if fr is None or eng is None:
            ax.set_title(rc.name)
            ax.text(0.5, 0.5, "missing fit/engine", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.errorbar(rc.R_kpc, rc.V_kms, yerr=rc.eV_kms, fmt="o", ms=3, lw=0.8, capsize=2, label="data")

        # model curve: V(R) = A * v_model(R/Rs)
        idx = np.argsort(eng.R_model)
        Rm = eng.R_model[idx]
        vm = eng.v_model[idx]
        Rplot = np.linspace(max(1e-6, np.nanmin(rc.R_kpc)), np.nanmax(rc.R_kpc), 250)
        rq = Rplot / fr.Rs
        vshape = np.interp(rq, Rm, vm, left=np.nan, right=np.nan)
        Vmodel = fr.A * vshape
        ax.plot(Rplot, Vmodel, "-", lw=1.5, label="fit")

        ax.set_title(f"{rc.name}  (χ²ν={fr.chi2_nu:.2g})")
        ax.set_xlabel("R (kpc)")
        ax.set_ylabel("V (km/s)")
        ax.grid(True, alpha=0.25)

        # small info box
        ax.text(0.02, 0.02,
                f"Rd={fr.Rd:.3g} kpc\nRs={fr.Rs:.3g} kpc\nA={fr.A:.3g} km/s",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, linewidth=0.5))

    for j in range(n, len(ax_list)):
        ax_list[j].axis("off")

    # shared legend in first visible axis
    if len(galaxies) > 0:
        ax_list[0].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


# -----------------------------
# SPARC helper: Rdisk from catalogue (VizieR)
# -----------------------------


def _canon_gal_name(name: str) -> str:
    """Canonicalize galaxy names for robust matching between SPARC catalogue and rotmod files.
    Example: 'NGC 3198' -> 'NGC3198', 'F563-V2' -> 'F563V2'
    """
    if name is None:
        return ""
    s = name.strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)  # drop spaces, hyphens, underscores, etc.
    return s


def load_sparc_Rdisk_map(cache_path: Path, insecure: bool = False) -> Dict[str, float]:
    """
    Returns dict: {GalaxyName -> Rdisk_kpc}, using SPARC (Lelli+ 2016) Table 1.
    Strategy:
      - If cache_path exists: read it
      - Else download TSV from VizieR ASU service and save to cache_path

    The VizieR table is: J/AJ/152/157/table1, column 'Rdisk' (kpc).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        import urllib.parse, urllib.request
        base = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
        params = {
            "-source": "J/AJ/152/157/table1",
            "-out.max": "9999",
            "-out": "Name,Rdisk",
            "-out.meta": "h",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "UCM-Engine Pilot/1.0"})
        import ssl
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except Exception as e:
            # Common on some Windows/MiKTeX setups: missing CA bundle -> CERTIFICATE_VERIFY_FAILED
            if insecure:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                    data = r.read()
            else:
                raise
        cache_path.write_bytes(data)

    # parse TSV (skip comment lines)
    out: Dict[str, float] = {}
    lines = cache_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = None
    for ln in lines:
        if not ln or ln.lstrip().startswith("#"):
            continue
        cols = ln.split("\t")
        if header is None:
            header = cols
            continue
        if len(cols) < 2:
            continue
        name = cols[0].strip()
        rd_s = cols[1].strip()
        try:
            rd = float(rd_s)
        except Exception:
            continue
        if name:
            out[name] = rd
            out[_canon_gal_name(name)] = rd  # canonical key
    return out


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pilot-test UCM-Engine vs SPARC rotation curves (Table + Figure). "
                    "Supports Benchmark presets v2 (legacy) and v3 (catalog Rdisk + tail smoothing)."
    )

    p.add_argument("--preset", type=str, default="v2", choices=["v2", "v3"],
                   help="Preset config: v2 (legacy) or v3 (Rdisk from SPARC + tail smoothing).")

    p.add_argument("--galaxies", type=str,
                   default="NGC3198,NGC2403,UGC128,DDO154,F563-V2,IC2574",
                   help="Comma-separated galaxy names as in SPARC Rotmod filenames (without _rotmod.dat).")
    p.add_argument("--galaxies-file", type=str, default=None,
                   help="Path to a text file with galaxy names (one per line, optional commas). Overrides --galaxies.")
    p.add_argument("--out", type=str, default="pilot",
                   help="Output prefix (folder will be created).")
    p.add_argument("--sparc-dir", type=str, default="sparc_rotmod",
                   help="Where to store SPARC zip and extracted files.")

    # fixed regime params
    p.add_argument("--L", type=float, default=20.0)
    p.add_argument("--k0", type=float, default=0.15)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--mode", type=str, default="rho_weighted", choices=["rho_weighted", "rho_slab", "midline"],
                   help="How to reduce 2D field to 1D observable: rho_weighted (default slab ~6*hz), rho_slab (tunable z-cut), midline (use near-midplane Phi).")
    p.add_argument("--zcut-factor", type=float, default=2.0,
                   help="For --mode rho_slab: include only |z| <= zcut_factor*hz in rho-weighted reduction (typical 1.5..3).")

    # domain / mesh
    p.add_argument("--Rmax", type=float, default=100.0)
    p.add_argument("--Zmax", type=float, default=60.0)
    p.add_argument("--Nr", type=int, default=260)
    p.add_argument("--Nz", type=int, default=240)

    # baryon geometry (defaults; can be overridden by SPARC Rdisk if enabled)
    p.add_argument("--Rd", type=float, default=4.0, help="Fallback disk scale length Rdisk (kpc).")
    p.add_argument("--hz", type=float, default=0.35, help="Disk vertical scale height (kpc).")
    p.add_argument("--use-sparc-Rd", action="store_true",
                   help="Use per-galaxy Rdisk from SPARC catalogue (not a fit parameter).")
    p.add_argument("--sparc-rd-cache", type=str, default="",
                   help="Optional path to cache TSV with columns Name,Rdisk (default: <out>/_cache/sparc_table1.tsv).")
    p.add_argument("--sparc-insecure", action="store_true",
                   help="Retry SPARC Rdisk catalogue download with SSL verification disabled (workaround for CERTIFICATE_VERIFY_FAILED).")

    # observable smoothing (applied to gRbar(R) inside v_from_rho_weighted if supported by engine)
    p.add_argument("--obs-smooth", type=str, default="",
                   choices=["", "none", "box", "savgol", "adaptive_box"],
                   help="Low-pass smoothing for observable gRbar(R): none/box/savgol/adaptive_box. "
                        "Empty uses preset defaults.")
    p.add_argument("--phi-eff-smooth", type=str, default="",
                   choices=["", "none", "box", "savgol", "adaptive_box"],
                   help="Smoothing for Phi_eff(R) BEFORE derivative when using --grad-method pchip. "
                        "Often reduces high-frequency ripple that PCHIP derivative amplifies. "
                        "Empty uses same value as --obs-smooth.")

    p.add_argument("--smooth-w0", type=int, default=0, help="Base smoothing window (odd). 0 -> preset default.")
    p.add_argument("--smooth-w1", type=int, default=0, help="Tail smoothing window for adaptive_box. 0 -> preset default.")
    p.add_argument("--smooth-tail-frac", type=float, default=0.0,
                   help="Tail start as fraction of R-grid [0..1] for adaptive smoothing. 0 -> preset default.")
    p.add_argument("--sg-poly", type=int, default=2, help="Poly order for Savitzky–Golay (if used).")

    # engine stabilization: enforce inward (attractive) gRbar(R) to avoid abs()-cusps when ripple flips sign
    p.add_argument("--force-attractive", dest="force_attractive", action="store_true", default=None,
                   help="Clamp gRbar(R) to be inward (negative) to suppress sign-flip cusps in v(R). "
                        "If not specified, preset may set a default.")
    p.add_argument("--no-force-attractive", dest="force_attractive", action="store_false",
                   help="Disable inward-field clamping (override preset).")
    p.add_argument("--force-attractive-eps", type=float, default=0.0,
                   help="Absolute clamp floor for gRbar (model units). 0 -> auto (1e-8 * median|gRbar|).")

    # fitting options
    p.add_argument("--fit-Rs", action="store_true", help="Fit Rs (radial scale) per galaxy (default: ON).")
    p.add_argument("--no-fit-Rs", dest="fit_Rs", action="store_false", help="Fix Rs=1 and fit only A.")
    p.set_defaults(fit_Rs=True)

    # experimental: two-zone Rs (transition vs tail)
    p.add_argument("--fit-two-Rs", action="store_true",
                   help="Fit two radial scales Rs_in (R < Rbreak) and Rs_out (R >= Rbreak). Experimental diagnostic for transition-region failures.")
    p.add_argument("--Rs-break-mode", type=str, default="2Rd",
                   choices=["kpc", "2Rd", "max(5,2Rd)"],
                   help="Break radius mode separating Rs_in and Rs_out. Default: 2Rd.")
    p.add_argument("--Rs-break-kRd", type=float, default=4.0,
                   help="If Rs-break-mode is 2Rd/max(5,2Rd): Rbreak = kRd * Rdisk (default 4.0).")
    p.add_argument("--Rs-break-kpc", type=float, default=0.0,
                   help="If Rs-break-mode is kpc: fixed break radius in kpc.")
    p.add_argument("--two-Rs-steps", type=int, default=60,
                   help="Grid steps per dimension for two-Rs search (default 60 -> 3600 combos).")
    p.add_argument("--two-Rs-fixed", action="store_true",
                   help="Use fixed two-zone Rs (global constants) and fit only A per galaxy. "
                        "Requires --fit-two-Rs. Overrides the two-Rs grid search.")
    p.add_argument("--two-Rs-fixed-in", action="store_true",
                   help="Fix Rs_in (global) but still fit Rs_out per galaxy on the Rs grid. This targets large-disk transition issues without degrading small disks. Requires --fit-two-Rs.")
    p.add_argument("--two-Rs-Rs-in", dest="two_Rs_Rs_in", type=float, default=0.5,
                   help="Fixed inner Rs (kpc per model unit) used when --two-Rs-fixed is set (default 0.5).")
    p.add_argument("--two-Rs-Rs-out", dest="two_Rs_Rs_out", type=float, default=2.0,
                   help="Fixed outer Rs (kpc per model unit) used when --two-Rs-fixed is set (default 2.0).")
    p.add_argument("--two-Rs-Rd-min", dest="two_Rs_Rd_min", type=float, default=0.0,
                   help="Apply two-Rs only if Rd_kpc >= this threshold (kpc). If 0, apply to all galaxies. Recommended: 3.0 for 'large disk' patch.")
    p.add_argument("--large-disk-patch", action="store_true",
                   help="Enable recommended large-disk patch: apply two-Rs only for large disks (Rd>=3 kpc), "
                        "fix Rs_in=0.4 kpc per model unit, set break=4.5*Rd, and fit Rs_out as usual. "
                        "Equivalent to: --fit-two-Rs --two-Rs-fixed-in --two-Rs-Rs-in 0.4 --two-Rs-Rd-min 3.0 "
                        "--Rs-break-mode 2Rd --Rs-break-kRd 4.5.")


    p.add_argument("--Rs-min", type=float, default=0.2, help="Min Rs (kpc per model unit) for grid search.")
    p.add_argument("--Rs-max", type=float, default=2.0, help="Max Rs for grid search.")
    p.add_argument("--Rs-steps", type=int, default=300, help="Grid steps for Rs search.")
    p.add_argument("--sigma-floor", type=float, default=0.0,
                   help="Velocity error floor in km/s added in quadrature: eV_eff=sqrt(eV^2+sigma_floor^2). "
                        "0 -> preset default.")
    p.add_argument("--Rcut", type=float, default=0.0,
                   help="Fit only points with R >= Rcut (kpc). Useful to focus on outer-curve regime.")

    # smarter outer-cut controls (optional). Backward compatible with --Rcut.
    p.add_argument("--Rcut-mode", type=str, default="legacy",
                   choices=["legacy", "kpc", "2Rd", "max(5,2Rd)"],
                   help="How to choose Rcut for OUTER selection. "
                        "'legacy' uses --Rcut as-is; other modes compute Rcut per galaxy using Rdisk.")
    p.add_argument("--Rcut-kpc", type=float, default=0.0,
                   help="Rcut in kpc when --Rcut-mode=kpc.")
    p.add_argument("--Rcut-kRd", type=float, default=2.0,
                   help="Multiplier for Rdisk when --Rcut-mode=2Rd or max(5,2Rd).")
    p.add_argument("--Rcut-min-points", type=int, default=6,
                   help="Guarantee at least this many points remain after cut (fallback reduces Rcut).")
    p.add_argument("--Rcut-margin", type=float, default=0.5,
                   help="Clamp Rcut to <= Rmax - margin (kpc).")

    p.add_argument("--Rcut-fallback", type=str, default="lower",
                   choices=["lower", "keep", "skip"],
                   help="If the physical cut (from --Rcut-mode) leaves fewer than --Rcut-min-points points, "
                        "choose how to handle it: "
                        "'lower' reduces Rcut to keep at least Nmin points (may include inner points), "
                        "'keep' preserves the physical cut and marks outer_lowN=1, "
                        "'skip' skips that galaxy.")
    p.add_argument("--Rcut-big-Rd", type=float, default=6.0,
                   help="Patch for big disks: if Rd >= this (kpc) AND the physical cut leaves < Nmin points, "
                        "apply --Rcut-big-policy instead of --Rcut-fallback. Set <=0 to disable.")
    p.add_argument("--Rcut-big-policy", type=str, default="keep",
                   choices=["lower", "keep", "skip"],
                   help="Fallback policy used for big disks (Rd >= --Rcut-big-Rd) when tail is sparse under the physical cut.")


    # data window selection (tail vs ring)
    p.add_argument("--outer-window", type=str, default="tail", choices=["tail", "ring"],
                   help="Which part of the rotation curve to fit: "
                        "'tail' uses R >= Rcut; "
                        "'ring' uses R in [Rcut, Rcut2).")

    # upper boundary for ring window
    p.add_argument("--Rcut2-mode", type=str, default="2Rd",
                   choices=["legacy", "kpc", "2Rd", "max(5,2Rd)"],
                   help="Upper boundary mode for ring window. "
                        "'2Rd' sets Rcut2 = --Rcut2-kRd * Rdisk. "
                        "Default gives 4Rd with --Rcut2-kRd=4.0.")
    p.add_argument("--Rcut2-kpc", type=float, default=0.0,
                   help="Rcut2 in kpc when --Rcut2-mode=kpc.")
    p.add_argument("--Rcut2-kRd", type=float, default=4.0,
                   help="Multiplier for Rdisk when --Rcut2-mode=2Rd or max(5,2Rd). Default 4.0.")
    p.add_argument("--Rcut2-margin", type=float, default=0.5,
                   help="Clamp Rcut2 to <= Rmax - margin (kpc).")


    # plot
    p.add_argument("--ncols", type=int, default=3, help="Columns in multipanel figure.")
    p.add_argument("--engine", type=str, default="ucm_rotation_curve_2d_sparse_BASE.py",
                   help="Path to UCM engine python file.")

    p.add_argument("--grad-method", type=str, default="fd2d", choices=["fd2d", "pchip"],
                   help="Engine observable derivative method (passed to v_from_rho_weighted if supported).")

    # debug / diagnostics
    p.add_argument(
        "--debug-engine-smooth",
        action="store_true",
        help=(
            "Compare engine curves with obs_smooth=none vs current smoothing and save a small plot. "
            "Useful to confirm that smoothing is actually applied and to quantify its effect."
        ),
    )
    p.add_argument(
        "--debug-engine-galaxy",
        type=str,
        default="",
        help="If set, run engine-smoothing debug only for this galaxy name (e.g., NGC3198).",
    )

    # Debug: print Rcut diagnostics (useful to inspect big-disk behavior)
    p.add_argument("--debug-rcut", action="store_true",
                   help="Print an Rcut diagnostics table at the end of the run.")
    p.add_argument("--debug-rcut-galaxies", type=str, default="",
                   help="Comma-separated list of galaxy names to include in the Rcut diagnostics table.")
    p.add_argument("--debug-rcut-top", type=int, default=0,
                   help="If >0, show only the top-N galaxies by chi2_nu in the Rcut diagnostics table.")
    p.add_argument("--debug-rcut-out", type=str, default="",
                   help="Optional path to write the Rcut diagnostics table (txt).")
    return p.parse_args()



def _canon_list_from_csv(s: str) -> List[str]:
    """Parse comma-separated galaxy list and canonicalize names."""
    if not s:
        return []
    return [_canon_gal_name(x) for x in s.split(",") if x.strip()]


def _rcut_debug_lines(results: List[FitResult], args: argparse.Namespace) -> List[str]:
    """Build lines for a compact Rcut diagnostics table."""
    sel = list(results)

    # Filter by explicit list, if provided
    want = set(_canon_list_from_csv(getattr(args, "debug_rcut_galaxies", "")))
    if want:
        sel = [r for r in sel if _canon_gal_name(r.name) in want]

    # If requested, restrict to top-N worst (by chi2_nu)
    topn = int(getattr(args, "debug_rcut_top", 0) or 0)
    if topn > 0:
        sel = sorted(sel, key=lambda r: (float("inf") if not np.isfinite(r.chi2_nu) else r.chi2_nu), reverse=True)[:topn]

    # Sort deterministically (worst first, then name)
    sel = sorted(sel, key=lambda r: (-(r.chi2_nu if np.isfinite(r.chi2_nu) else -1e99), r.name))

    lines: List[str] = []
    lines.append("[debug-rcut] Rcut diagnostics (Rcut_phys vs Rcut_used; big-disk fallback policies)")
    lines.append("Galaxy      win  Rd_kpc  Rmax_kpc   R1_phys   R1_used   R2_kpc   N_phys  N_used  lowN  policy   chi2_nu")
    lines.append("-" * 110)
    for r in sel:
        lines.append(
            f"{r.name:10s}  "
            f"{str(getattr(r,'outer_window','tail'))[:4]:4s} "
            f"{r.Rd:6.2f}  "
            f"{r.Rmax:7.2f}  "
            f"{(r.Rcut_phys if np.isfinite(r.Rcut_phys) else float('nan')):8.2f}  "
            f"{r.Rcut:8.2f}  "
            f"{(getattr(r,'Rcut2',float('nan')) if np.isfinite(getattr(r,'Rcut2',float('nan'))) else float('nan')):7.2f}  "
            f"{int(getattr(r,'N_after_phys',0)):6d}  "
            f"{int(getattr(r,'N_after',0)):6d}  "
            f"{int(getattr(r,'outer_lowN',0)):4d}  "
            f"{str(getattr(r,'rcut_policy','')):6s}  "
            f"{r.chi2_nu:7.2f}"
        )
    return lines


def _print_rcut_debug(results: List[FitResult], args: argparse.Namespace) -> None:
    """Print (and optionally save) the Rcut diagnostics table."""
    lines = _rcut_debug_lines(results, args)
    print("\n" + "\n".join(lines))

    outp = str(getattr(args, "debug_rcut_out", "") or "").strip()
    if outp:
        try:
            Path(outp).parent.mkdir(parents=True, exist_ok=True)
            Path(outp).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[debug-rcut] wrote: {outp}")
        except Exception as e:
            print(f"[debug-rcut] failed to write '{outp}': {e}")
def main() -> None:
    args = parse_args()

    # convenience: recommended large-disk patch (does not affect small disks)
    if getattr(args, "large_disk_patch", False):
        args.fit_two_Rs = True
        args.two_Rs_fixed_in = True
        args.two_Rs_fixed = False
        args.two_Rs_Rs_in = 0.4
        args.two_Rs_Rd_min = 3.0
        args.Rs_break_mode = "2Rd"
        args.Rs_break_kRd = 4.5

    if getattr(args, "two_Rs_fixed", False) and not getattr(args, "fit_two_Rs", False):
        raise RuntimeError("--two-Rs-fixed requires --fit-two-Rs.")
    if getattr(args, "two_Rs_fixed_in", False) and not getattr(args, "fit_two_Rs", False):
        raise RuntimeError("--two-Rs-fixed-in requires --fit-two-Rs.")
    print(f"[run] mode={args.mode} grad_method={args.grad_method} zcut_factor={getattr(args, 'zcut_factor', None)}")

    # --- preset defaults (only applied when user kept 0/empty values)
    if args.preset == "v3":
        if not args.use_sparc_Rd:
            args.use_sparc_Rd = True
        if args.sigma_floor == 0.0:
            args.sigma_floor = 8.0
        if args.obs_smooth == "":
            args.obs_smooth = "adaptive_box"
        if args.smooth_w0 == 0:
            args.smooth_w0 = 11
        if args.smooth_w1 == 0:
            args.smooth_w1 = 31
        if args.smooth_tail_frac == 0.0:
            args.smooth_tail_frac = 0.7
        if args.force_attractive is None:
            args.force_attractive = True

    else:
        # v2 legacy: preserve historical behaviour (no catalogue Rd, no extra smoothing, no sigma floor)
        if args.obs_smooth == "":
            args.obs_smooth = "none"
        if args.smooth_w0 == 0:
            args.smooth_w0 = 11
        if args.smooth_w1 == 0:
            args.smooth_w1 = 31
        if args.smooth_tail_frac == 0.0:
            args.smooth_tail_frac = 0.7

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sparc_root = Path(args.sparc_dir).resolve()
    zip_path = sparc_root / "Rotmod_LTG.zip"
    extract_dir = sparc_root / "Rotmod_LTG_extracted"

    download_if_needed(zip_path)
    rotmod_dir = unzip_rotmod(zip_path, extract_dir)

    # galaxies list: either from --galaxies-file (preferred for reproducibility) or from --galaxies CSV
    galaxy_names: List[str] = []
    if getattr(args, "galaxies_file", None):
        gpath = Path(args.galaxies_file).expanduser()
        if not gpath.is_absolute():
            gpath = (Path.cwd() / gpath).resolve()
        if not gpath.exists():
            raise FileNotFoundError(f"galaxies file not found: {gpath}")
        txt = gpath.read_text(encoding="utf-8", errors="ignore")
        raw: List[str] = []
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split(","):
                tok = tok.strip()
                if tok:
                    raw.append(tok)
        # de-duplicate while preserving order
        seen = set()
        for g in raw:
            if g not in seen:
                galaxy_names.append(g)
                seen.add(g)
    else:
        galaxy_names = [s.strip() for s in args.galaxies.split(",") if s.strip()]
    galaxies = load_galaxies(rotmod_dir, galaxy_names)

    engine_py = Path(args.engine).resolve()
    if not engine_py.exists():
        raise FileNotFoundError(
            f"Engine file not found: {engine_py}\nPut it next to this script, or pass --engine <path>"
        )

    # Rs grid
    Rs_steps_eff = int(args.Rs_steps)
    if getattr(args, 'fit_two_Rs', False):
        Rs_steps_eff = int(max(5, min(Rs_steps_eff, int(args.two_Rs_steps))))
    Rs_grid = np.linspace(args.Rs_min, args.Rs_max, Rs_steps_eff)

    # per-galaxy Rdisk map (optional)
    rd_map: Dict[str, float] = {}
    if args.use_sparc_Rd:
        if args.sparc_rd_cache:
            rd_cache_path = Path(args.sparc_rd_cache).expanduser()
        else:
            rd_cache_path = out_dir / "_cache" / "sparc_table1_Name_Rdisk.tsv"
        try:
            rd_map = load_sparc_Rdisk_map(rd_cache_path, insecure=args.sparc_insecure)
            print(f"[sparc] Rdisk catalogue loaded: {len(rd_map)} entries (cache: {rd_cache_path.name})")
        except Exception as e:
            print(f"[sparc] WARNING: could not load Rdisk catalogue ({e}). Falling back to --Rd for all.")

    # compute engine once if geometry fixed; else cache per galaxy
    fixed_engine: EngineCurve | None = None
    if not args.use_sparc_Rd:
        print("[engine] computing reference v(R) once (fixed Rd) ...")
        fixed_engine = compute_engine_curve(
            engine_py=engine_py,
            L=args.L, k0=args.k0,
            Rmax=args.Rmax, Zmax=args.Zmax, Nr=args.Nr, Nz=args.Nz,
            Rd=args.Rd, hz=args.hz, nu=args.nu,
            mode=args.mode,
                zcut_factor=float(getattr(args,'zcut_factor',2.0)),
            grad_method=args.grad_method,
            obs_smooth=args.obs_smooth,
            phi_eff_smooth=args.phi_eff_smooth,
            smooth_w0=args.smooth_w0,
            smooth_w1=args.smooth_w1,
            smooth_tail_frac=args.smooth_tail_frac,
            sg_poly=args.sg_poly,
            cache_dir=out_dir / "_cache",
            force_attractive=args.force_attractive,
            force_attractive_eps=args.force_attractive_eps)
        print(f"[engine] got {fixed_engine.R_model.size} points; vmax={np.nanmax(fixed_engine.v_model):.3g}")

    # fit
    print("[fit] fitting galaxies ...")
    results_list: List[FitResult] = []
    results_map: Dict[str, FitResult] = {}
    engine_map: Dict[str, EngineCurve] = {}

    for rc in galaxies:
        Rd_used = float(rd_map.get(rc.name, rd_map.get(_canon_gal_name(rc.name), args.Rd))) if args.use_sparc_Rd else float(args.Rd)
        if args.use_sparc_Rd and (rc.name not in rd_map) and (_canon_gal_name(rc.name) not in rd_map):
            print(f"[sparc] NOTE: Rdisk not found for {rc.name!s}; using fallback --Rd={args.Rd:g}")
        hz_used = float(args.hz)

        if fixed_engine is None:
            eng = compute_engine_curve(
                engine_py=engine_py,
                L=args.L, k0=args.k0,
                Rmax=args.Rmax, Zmax=args.Zmax, Nr=args.Nr, Nz=args.Nz,
                Rd=Rd_used, hz=hz_used, nu=args.nu,
                mode=args.mode,
                zcut_factor=float(getattr(args,'zcut_factor',2.0)),
            grad_method=args.grad_method,
                obs_smooth=args.obs_smooth,
            phi_eff_smooth=args.phi_eff_smooth,
                smooth_w0=args.smooth_w0,
                smooth_w1=args.smooth_w1,
                smooth_tail_frac=args.smooth_tail_frac,
                sg_poly=args.sg_poly,
                cache_dir=out_dir / "_cache",
                force_attractive=args.force_attractive,
                force_attractive_eps=args.force_attractive_eps)
        else:
            eng = fixed_engine

        # --- optional diagnostic: confirm/quantify engine-level smoothing
        if bool(args.debug_engine_smooth) and args.mode == "rho_weighted":
            dbg_name = str(args.debug_engine_galaxy or "").strip()
            if (not dbg_name) or (_normalize_sparc_name(dbg_name) == _normalize_sparc_name(rc.name)):
                try:
                    eng0 = compute_engine_curve(
                        engine_py=engine_py,
                        L=args.L, k0=args.k0,
                        Rmax=args.Rmax, Zmax=args.Zmax, Nr=args.Nr, Nz=args.Nz,
                        Rd=Rd_used, hz=hz_used, nu=args.nu,
                        mode=args.mode,
                zcut_factor=float(getattr(args,'zcut_factor',2.0)),
            grad_method=args.grad_method,
                        obs_smooth="none",
                        smooth_w0=args.smooth_w0,
                        smooth_w1=args.smooth_w1,
                        smooth_tail_frac=args.smooth_tail_frac,
                        sg_poly=args.sg_poly,
                        cache_dir=out_dir / "_cache",
                        force_attractive=False,
                        force_attractive_eps=args.force_attractive_eps)

                    v0 = np.asarray(eng0.v_model, float)
                    v1 = np.asarray(eng.v_model, float)
                    n = min(v0.size, v1.size)
                    if n > 5:
                        v0 = v0[:n]
                        v1 = v1[:n]
                        denom = np.maximum(np.abs(v0), 1e-12)
                        rel = np.abs(v1 - v0) / denom
                        rel_max = float(np.nanmax(rel))
                        rel_med = float(np.nanmedian(rel))
                        print(
                            f"[debug] {rc.name}: engine smoothing effect (none -> {args.obs_smooth}) "
                            f"rel_max={rel_max:.3g}, rel_median={rel_med:.3g}"
                        )

                        # small plot
                        try:
                            import matplotlib.pyplot as plt
                            dbg_dir = out_dir / "_debug"
                            dbg_dir.mkdir(parents=True, exist_ok=True)
                            Rm = np.asarray(eng.R_model[:n], float)
                            plt.figure(figsize=(6.2, 4.2))
                            plt.plot(Rm, v0, label="engine: obs_smooth=none")
                            plt.plot(Rm, v1, label=f"engine: obs_smooth={args.obs_smooth}, force_attr={bool(args.force_attractive)}")
                            plt.xlabel("R (model units)")
                            plt.ylabel("v (model units)")
                            plt.title(f"Engine curve smoothing check: {rc.name}")
                            plt.legend(fontsize=8)
                            plt.tight_layout()
                            plt.savefig(dbg_dir / f"engine_compare_{rc.name}.png", dpi=160)
                            plt.close()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[debug] {rc.name}: engine smoothing check failed: {e}")

        # --- choose per-galaxy radial window (tail or ring)
        outer_window = str(getattr(args, "outer_window", "tail") or "tail").strip().lower()
        rcut2_used = float("nan")
        rcut2_phys = float("nan")

        if outer_window == "ring":
            # upper boundary Rcut2 (physical; no fallback)
            mode2 = str(getattr(args, "Rcut2_mode", "2Rd"))
            R = np.asarray(rc.R_kpc, dtype=float)
            R = R[np.isfinite(R)]
            if R.size < 3:
                print(f"[Rcut] {rc.name}: ring -> not enough finite R points -> skipped")
                continue
            Rmax = float(R[-1])
            margin2 = float(getattr(args, "Rcut2_margin", getattr(args, "Rcut_margin", 0.5)))

            if mode2 == "legacy":
                rcut2_phys = float(getattr(args, "Rcut", 0.0))
            elif mode2 == "kpc":
                rcut2_phys = float(getattr(args, "Rcut2_kpc", 0.0))
            else:
                kRd2 = float(getattr(args, "Rcut2_kRd", 4.0))
                Rd_val = float(Rd_used) if (Rd_used is not None and np.isfinite(Rd_used)) else 0.0
                rcut2_phys = kRd2 * Rd_val
                if mode2 == "max(5,2Rd)":
                    rcut2_phys = max(5.0, rcut2_phys)

            rcut2_phys = min(float(rcut2_phys), float(Rmax - margin2))
            rcut2_phys = max(0.0, float(rcut2_phys))
            rcut2_used = float(rcut2_phys)

            # choose lower bound using only points with R < Rcut2, so N_after counts ring points
            i2 = int(np.searchsorted(R, rcut2_used, side="left"))
            if i2 < 3:
                print(f"[Rcut] {rc.name}: ring -> too few points under Rcut2={rcut2_used:.3g} kpc -> skipped")
                continue

            rcut_used, n_after, rcut_phys, n_after_phys, outer_lowN, pol, skip = choose_rcut_with_policy(R[:i2], Rd_used, args)
            if skip:
                print(f"[Rcut] {rc.name}: ring window skipped (policy={pol})")
                continue
            if rcut2_used <= rcut_used + 1e-12:
                print(f"[Rcut] {rc.name}: ring -> invalid bounds (Rcut1={rcut_used:.3g}, Rcut2={rcut2_used:.3g}) -> skipped")
                continue

            print(f"[Rcut] {rc.name}: window=ring mode1={args.Rcut_mode} mode2={mode2} Rd={Rd_used:.3g} "
                  f"R1_phys={rcut_phys:.3g} R1_used={rcut_used:.3g} R2={rcut2_used:.3g} "
                  f"N_phys={n_after_phys} N={n_after} policy={pol} lowN={outer_lowN}")
        else:
            outer_window = "tail"
            rcut_used, n_after, rcut_phys, n_after_phys, outer_lowN, pol, skip = choose_rcut_with_policy(rc.R_kpc, Rd_used, args)
            if skip:
                print(f"[Rcut] {rc.name}: window=tail mode={args.Rcut_mode} Rd={Rd_used:.3g} "
                      f"Rcut_phys={rcut_phys:.3g} N_phys={n_after_phys} < Nmin={args.Rcut_min_points} -> skipped (policy={pol})")
                continue

            if str(getattr(args, "Rcut_mode", "legacy")) != "legacy":
                print(f"[Rcut] {rc.name}: window=tail mode={args.Rcut_mode} Rd={Rd_used:.3g} "
                      f"Rcut_phys={rcut_phys:.3g} Rcut_used={rcut_used:.3g} "
                      f"N_phys={n_after_phys} N={n_after} policy={pol} lowN={outer_lowN}")
            elif rcut_used > 0:
                print(f"[Rcut] {rc.name}: window=tail mode={args.Rcut_mode} Rd={Rd_used:.3g} "
                      f"Rcut_used={rcut_used:.3g} N={n_after}")

        fr = fit_one_galaxy(

            rc, eng,
            Rs_grid=Rs_grid,
            Rd_used=Rd_used,
            hz_used=hz_used,
            fit_Rs=bool(args.fit_Rs),
            fit_two_Rs=bool(getattr(args, "fit_two_Rs", False)),
            two_Rs_fixed=bool(getattr(args, "two_Rs_fixed", False)),
            two_Rs_fixed_in=bool(getattr(args, "two_Rs_fixed_in", False)),
            two_Rs_Rs_in=float(getattr(args, "two_Rs_Rs_in", 0.5)),
            two_Rs_Rs_out=float(getattr(args, "two_Rs_Rs_out", 2.0)),
            two_Rs_Rd_min=float(getattr(args, "two_Rs_Rd_min", 0.0)),
            Rs_break_mode=str(getattr(args, "Rs_break_mode", "2Rd")),
            Rs_break_kRd=float(getattr(args, "Rs_break_kRd", 4.0)),
            Rs_break_kpc=float(getattr(args, "Rs_break_kpc", 0.0)),
            sigma_floor=float(args.sigma_floor),
            Rcut=float(rcut_used),
            Rcut_phys=float(rcut_phys),
            N_after=int(n_after),
            N_after_phys=int(n_after_phys),
            outer_lowN=int(outer_lowN),
            rcut_policy=str(pol),
            outer_window=str(outer_window),
            Rcut2=float(rcut2_used),
            Rcut2_phys=float(rcut2_phys)
        )

        results_list.append(fr)
        results_map[rc.name] = fr
        engine_map[rc.name] = eng

        print(f"  {fr.name:10s}  Rd={fr.Rd:.3g}  A={fr.A:.4g}  Rs={fr.Rs:.4g}  chi2_nu={fr.chi2_nu:.3g}")

    rd_str = "SPARC" if args.use_sparc_Rd else f"{args.Rd:g}"
    fixed_params_line = (
        f"UCM fixed regime: L={args.L:g}, k0={args.k0:g}, nu={args.nu:.2e}, mode={args.mode}; "
        f"obs_smooth={args.obs_smooth}, w0={args.smooth_w0}, w1={args.smooth_w1}, tail={args.smooth_tail_frac:g}; "
        f"mesh: Rmax={args.Rmax:g}, Zmax={args.Zmax:g}, Nr={args.Nr}, Nz={args.Nz}; "
        f"sigma_floor={args.sigma_floor:g} km/s; "
        f"Rcut_mode={getattr(args, 'Rcut_mode', 'legacy')}, Rcut={args.Rcut:g} kpc, "
        f"Rdisk={rd_str} kpc, hz={args.hz:g} kpc."
    )

    write_csv(results_list, out_dir / "pilot_results.csv")
    write_table_tex(results_list, out_dir / "pilot_table.tex", fixed_params_line=fixed_params_line)
    plot_multipanel(galaxies, results_map, engine_map, out_pdf=out_dir / "pilot_fits.pdf", ncols=int(args.ncols))

    # also save png for quick look
    try:
        plot_multipanel(galaxies, results_map, engine_map, out_pdf=out_dir / "pilot_fits.png", ncols=int(args.ncols))
    except Exception:
        pass

    print("\n[done] Outputs:")
    print(f"  {out_dir/'pilot_results.csv'}")
    print(f"  {out_dir/'pilot_table.tex'}")
    print(f"  {out_dir/'pilot_fits.pdf'}")

    if getattr(args, "debug_rcut", False):
        # Tip: use --debug-rcut-top 6 or --debug-rcut-galaxies "NGC5055,NGC6674,..." for a short table
        _print_rcut_debug(results_list, args)
    print("\nLaTeX include examples:")
    print(r"  \begin{table} ... \input{"+str((out_dir/'pilot_table.tex').as_posix())+r"} ... \end{table}")
    print(r"  \begin{figure*} ... \includegraphics[width=\textwidth]{"+str((out_dir/"pilot_fits.pdf").as_posix())+r"} ... \end{figure*}")


if __name__ == "__main__":
    main()
