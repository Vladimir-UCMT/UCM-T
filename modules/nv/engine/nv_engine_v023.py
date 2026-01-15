# =============================================================================
# NV-Engine v0.23b — v0.23a + stable uncertainties + noisy demo CSV
#
# Changes vs v0.23a:
#  1) Uncertainty stabilization:
#     red = max(chi2/dof, 1.0) so SE does NOT collapse to ~0 when chi2 is tiny.
#     (Interpretation: sigmas are treated as known; reduced-chi2 scaling only inflates.)
#
#  2) Demo CSV now includes small random noise (seeded) so chi2 ~ dof.
#     Demo still runs if you execute the script with NO arguments.
#
# Usage:
#   - Run with no args (IDLE Run / double-click): demo mode.
#   - With your data:
#       python nv_engine_v023b.py mydata.csv --scan --d0 2.87e9
# =============================================================================

from __future__ import annotations

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- helpers ---------------------------------

def norm_key(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch == "_")


def pick_col(cols_norm: dict[str, str], *candidates: str) -> str | None:
    for c in candidates:
        c_n = norm_key(c)
        if c_n in cols_norm:
            return cols_norm[c_n]
    return None


def load_csv_columns(path: str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data is None or data.size == 0:
        raise ValueError("CSV appears empty or unreadable.")
    cols = {}
    for name in data.dtype.names:
        cols[name] = np.asarray(data[name], dtype=float)
    return cols


def g_of_z(z_m: np.ndarray, z0_m: float, p: float) -> np.ndarray:
    return 1.0 / (1.0 + (z_m / z0_m) ** p)


def make_differential(y: np.ndarray, sigma: np.ndarray, ref_idx: int) -> tuple[np.ndarray, np.ndarray]:
    y_ref = float(y[ref_idx])
    s_ref = float(sigma[ref_idx])
    y_diff = y - y_ref
    sigma_diff = np.sqrt(sigma**2 + s_ref**2)
    return y_diff, sigma_diff


def chi2_for_b(x: np.ndarray, y: np.ndarray, sigma: np.ndarray, b: float) -> float:
    r = (y - b * x) / sigma
    return float(np.sum(r**2))


def weighted_slope_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float, int, float]:
    """
    Fit y = b*x with weights 1/sigma^2.
    Returns:
      b_hat, se_b, chi2_fit, dof, denom
    where denom = sum(w*x^2) (curvature of chi2 wrt b).

    v0.23b change:
      se_b uses red = max(chi2/dof, 1.0) so uncertainties never collapse below the
      level implied by provided sigmas.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    if np.any(sigma <= 0):
        raise ValueError("All sigma values must be > 0.")
    w = 1.0 / (sigma**2)
    denom = float(np.sum(w * x * x))
    if denom == 0:
        raise ValueError("Degenerate x: sum(w*x^2)=0 (bad z0/p or z grid).")

    b_hat = float(np.sum(w * x * y) / denom)

    y_pred = b_hat * x
    r = (y - y_pred) / sigma
    chi2_fit = float(np.sum(r**2))
    dof = max(0, len(y) - 1)

    red = (chi2_fit / dof) if dof > 0 else 1.0
    red = max(red, 1.0)  # <<< ключевая стабилизация v0.23b
    var_b = red / denom
    se_b = float(np.sqrt(max(var_b, 0.0)))
    return b_hat, se_b, chi2_fit, dof, denom


def parse_grid_list(values: str) -> np.ndarray:
    values = values.strip()
    if ":" in values and "," not in values:
        a, b, step = values.split(":")
        a, b, step = float(a), float(b), float(step)
        if step <= 0:
            raise ValueError("Grid step must be > 0.")
        n = int(np.floor((b - a) / step)) + 1
        return a + step * np.arange(max(n, 0))
    parts = [p.strip() for p in values.split(",") if p.strip()]
    return np.array([float(p) for p in parts], dtype=float)


def topk_rows(rows: list[tuple], k: int = 10) -> list[tuple]:
    return sorted(rows, key=lambda t: t[0])[:k]


def delta_chi2_for_cl(cl: float, onesided: bool) -> float:
    """
    Common presets for 1 parameter:
      95% one-sided UL => Δχ²=2.71
      95% two-sided    => Δχ²=3.84
    """
    if onesided:
        presets = {0.90: 1.64, 0.95: 2.71, 0.99: 6.63}
    else:
        presets = {0.90: 2.71, 0.95: 3.84, 0.99: 6.63}
    if cl in presets:
        return presets[cl]
    keys = np.array(sorted(presets.keys()), float)
    nearest = float(keys[np.argmin(np.abs(keys - cl))])
    return presets[nearest]


def upper_limit_b_profile(
    b_hat: float,
    denom: float,
    delta: float,
    physical_nonneg: bool = True
) -> float:
    """
    chi2(b) = chi2_min + denom*(b-b_hat)^2
    Upper branch: b_UL = b_hat + sqrt(delta/denom)
    """
    if denom <= 0:
        raise ValueError("denom must be > 0.")
    step = float(np.sqrt(delta / denom))
    b_ul = b_hat + step
    if physical_nonneg:
        b_ul = max(0.0, b_ul)
    return b_ul


# ----------------------------- demo CSV (noisy) -----------------------------

def ensure_demo_csv(path: str, seed: int = 7) -> None:
    """
    Creates a demo CSV with a small amount of noise so chi2 ~ dof.
    We generate f_center from the toy model:
      f_center(z) = D0*(1 + eps*g(z))
    and then split into f_plus/f_minus by adding a symmetric Zeeman term ±gamma*Bz.
    """
    if os.path.exists(path):
        return

    rng = np.random.default_rng(seed)

    # Demo truth
    D0 = 2.87e9
    eps = 3e-6
    z0 = 80e-9
    p = 2.0

    # Zeeman split (arbitrary, cancels in f_center)
    gamma = 28.024e9  # Hz/T
    Bz = 50e-6        # Tesla
    dZ = gamma * Bz   # Hz

    # Measurement grid
    z_nm = np.array([20, 30, 50, 80, 120, 200, 400], dtype=float)
    z_m = z_nm * 1e-9

    g = g_of_z(z_m, z0, p)
    f_center_true = D0 * (1.0 + eps * g)

    # Noise levels (Hz)
    sigma_center = 250.0  # this is what we will put into sigma_hz
    # Add noise to center, then reconstruct plus/minus with same center noise
    f_center_obs = f_center_true + rng.normal(0.0, sigma_center, size=len(z_nm))

    f_plus = f_center_obs + dZ
    f_minus = f_center_obs - dZ

    # Write CSV
    with open(path, "w", encoding="utf-8") as f:
        f.write("z_nm,f_plus_hz,f_minus_hz,sigma_hz\n")
        for zn, fp, fm in zip(z_nm, f_plus, f_minus):
            f.write(f"{zn:.1f},{fp:.6f},{fm:.6f},{sigma_center:.1f}\n")


# ----------------------------- main ---------------------------------

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(description="NV-Engine v0.23b — CSV fitter + upper limit mode for b=D0*eps")
    ap.add_argument("csv", nargs="?", default=None, help="Path to CSV file with header. If omitted, runs demo mode.")
    ap.add_argument("--d0", type=float, default=None,
                    help="Optional D0 anchor (Hz). If provided, eps=b/D0 is reported (and eps_UL).")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Fallback sigma (Hz) if CSV has no sigma columns.")
    ap.add_argument("--ref", type=str, default="max",
                    help="Reference point for differential: 'max' (default) or index (0-based) or '400nm'.")
    ap.add_argument("--z0", type=float, default=80.0, help="Default z0 (nm) if scan disabled.")
    ap.add_argument("--p", type=float, default=2.0, help="Default p if scan disabled.")
    ap.add_argument("--scan", action="store_true", help="Enable grid scan over (z0,p).")
    ap.add_argument("--z0grid", type=str, default="40,60,80,100,140,200",
                    help="z0 grid in nm: '40,60,80' or range '40:200:20'")
    ap.add_argument("--pgrid", type=str, default="1.0,1.5,2.0,2.5,3.0",
                    help="p grid: '1.0,1.5,2.0' or range '1.0:3.0:0.5'")
    ap.add_argument("--topk", type=int, default=10, help="How many top grid rows to print.")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots.")

    ap.add_argument("--ul", action="store_true",
                    help="Force upper-limit reporting (even if signal is strong).")
    ap.add_argument("--cl", type=float, default=0.95,
                    help="Confidence level for UL (supported presets: 0.90, 0.95, 0.99). Default 0.95.")
    ap.add_argument("--two-sided", action="store_true",
                    help="Use two-sided Δχ² (default is one-sided UL).")
    ap.add_argument("--no-physical", action="store_true",
                    help="Disable physical constraint b>=0 (default: enforced).")
    ap.add_argument("--auto-ul-threshold", type=float, default=9.0,
                    help="Auto UL if Δχ² < threshold (default 9).")

    args = ap.parse_args(argv)

    # ---------------- demo mode ----------------
    if args.csv is None:
        args.csv = "nv_demo.csv"
        ensure_demo_csv(args.csv, seed=7)
        if not args.scan:
            args.scan = True
        if args.d0 is None:
            args.d0 = 2.87e9
        if args.sigma is None:
            args.sigma = 250.0
        print(f"[demo] No CSV provided → created/using '{args.csv}' and running with defaults: "
              f"--scan --d0 {args.d0:g} --sigma {args.sigma:g}\n")

    cols = load_csv_columns(args.csv)
    cols_norm = {norm_key(k): k for k in cols.keys()}

    # --- z ---
    z_nm_col = pick_col(cols_norm, "z_nm", "znm", "z", "height_nm", "z(nm)")
    z_m_col = pick_col(cols_norm, "z_m", "zm", "z_meters", "z(m)")
    if z_m_col is None and z_nm_col is None:
        raise ValueError("CSV must contain z_nm (preferred) or z_m.")
    if z_m_col is not None:
        z_m = cols[z_m_col].astype(float)
        z_nm = z_m * 1e9
    else:
        z_nm = cols[z_nm_col].astype(float)
        z_m = z_nm * 1e-9

    # --- f+ / f- ---
    fplus_col = pick_col(cols_norm, "f_plus_hz", "fplus_hz", "f_plus", "fplus", "f+_hz")
    fminus_col = pick_col(cols_norm, "f_minus_hz", "fminus_hz", "f_minus", "fminus", "f-_hz")
    if fplus_col is None or fminus_col is None:
        raise ValueError("CSV must contain f_plus_hz and f_minus_hz columns.")
    f_plus = cols[fplus_col].astype(float)
    f_minus = cols[fminus_col].astype(float)
    f_center = 0.5 * (f_plus + f_minus)

    # --- sigma ---
    sigc_col = pick_col(cols_norm, "sigma_center_hz", "sigma_center", "sig_center_hz")
    sigp_col = pick_col(cols_norm, "sigma_plus_hz", "sigma_plus", "sig_plus_hz")
    sigm_col = pick_col(cols_norm, "sigma_minus_hz", "sigma_minus", "sig_minus_hz")
    sig_col = pick_col(cols_norm, "sigma_hz", "sigma", "sig_hz")

    if sigc_col is not None:
        sigma_center = cols[sigc_col].astype(float)
    elif sigp_col is not None and sigm_col is not None:
        sigp = cols[sigp_col].astype(float)
        sigm = cols[sigm_col].astype(float)
        sigma_center = 0.5 * np.sqrt(sigp**2 + sigm**2)
    elif sig_col is not None:
        sigma_center = cols[sig_col].astype(float)
    elif args.sigma is not None:
        sigma_center = np.full_like(f_center, float(args.sigma), dtype=float)
    else:
        raise ValueError("No sigma columns found. Provide --sigma <Hz> or add sigma_* columns.")

    # Sort by z
    order = np.argsort(z_m)
    z_m = z_m[order]
    z_nm = z_nm[order]
    f_center = f_center[order]
    sigma_center = sigma_center[order]

    n = len(z_m)
    if n < 3:
        raise ValueError("Need at least 3 points for a meaningful differential fit.")

    # Choose ref
    ref = args.ref.strip().lower()
    if ref == "max":
        ref_idx = int(np.argmax(z_m))
    elif ref.endswith("nm"):
        zref_nm = float(ref.replace("nm", "").strip())
        ref_idx = int(np.argmin(np.abs(z_nm - zref_nm)))
    else:
        ref_idx = int(ref)
    if not (0 <= ref_idx < n):
        raise ValueError("Reference index out of range.")

    # Differential
    y_diff, sigma_diff = make_differential(f_center, sigma_center, ref_idx=ref_idx)

    def fit_for(z0_nm: float, p: float):
        z0_m = float(z0_nm) * 1e-9
        g = g_of_z(z_m, z0_m, float(p))
        x = g - g[ref_idx]
        b_hat, se_b, chi2_fit, dof, denom = weighted_slope_fit(x, y_diff, sigma_diff)
        chi2_0 = chi2_for_b(x, y_diff, sigma_diff, b=0.0)
        dchi2 = chi2_0 - chi2_fit
        return b_hat, se_b, chi2_fit, chi2_0, dchi2, dof, denom, g

    # Scan selection
    if args.scan:
        z0_grid = parse_grid_list(args.z0grid)
        p_grid = parse_grid_list(args.pgrid)
        rows = []
        best = None
        for z0_nm in z0_grid:
            for p in p_grid:
                b_hat, se_b, chi2_fit, chi2_0, dchi2, dof, denom, _ = fit_for(z0_nm, p)
                rows.append((chi2_fit, z0_nm, p, b_hat, se_b, dchi2))
                if best is None or chi2_fit < best[0]:
                    best = (chi2_fit, z0_nm, p)

        print("NV-Engine v0.23b — GRID SCAN (top by chi²)")
        for chi2_fit, z0_nm, p, b_hat, se_b, dchi2 in topk_rows(rows, k=max(1, args.topk)):
            print(f"chi2={chi2_fit:8.3f}  z0={z0_nm:7.1f} nm  p={p:3.2f}  "
                  f"b={b_hat: .3e} ±{1.96*se_b:.2e}  Δχ²={dchi2:9.3f}")
        chi2_best, z0_nm, p = best
        z0_nm = float(z0_nm)
        p = float(p)
        print(f"\nBest: z0={z0_nm:.1f} nm, p={p:.3f} (chi2={chi2_best:.3f})\n")
    else:
        z0_nm = float(args.z0)
        p = float(args.p)

    # Final fit
    b_hat, se_b, chi2_fit, chi2_0, dchi2, dof, denom, g = fit_for(z0_nm, p)
    ci_b = (b_hat - 1.96 * se_b, b_hat + 1.96 * se_b)

    onesided = not args.two_sided
    delta = delta_chi2_for_cl(args.cl, onesided=onesided)
    physical_nonneg = not args.no_physical

    auto_ul = (dchi2 < float(args.auto_ul_threshold))
    do_ul = args.ul or auto_ul
    b_ul = upper_limit_b_profile(b_hat=b_hat, denom=denom, delta=delta, physical_nonneg=physical_nonneg)

    print("NV-Engine v0.23b — differential fit (primary parameter b = D0*eps)")
    print(f"Chosen: z0 = {z0_nm:.1f} nm, p = {p:.3f}")
    print(f"Reference: index={ref_idx}, z_ref = {z_nm[ref_idx]:.1f} nm")
    print(f"FIT : b = {b_hat:.6e} ± {1.96*se_b:.2e} (95% CI: [{ci_b[0]:.6e}, {ci_b[1]:.6e}])")
    print(f"chi2(b=0) = {chi2_0:.3f}")
    print(f"chi2(fit) = {chi2_fit:.3f} (dof={dof})")
    print(f"Δχ²       = {dchi2:.3f}")

    if do_ul:
        mode = "one-sided" if onesided else "two-sided"
        phys = "ON" if physical_nonneg else "OFF"
        print(f"\nUPPER LIMIT MODE: {'forced' if args.ul else 'auto'}")
        print(f"CL={args.cl:.2f}, {mode}, Δχ²={delta:.3f}, physical b>=0: {phys}")
        print(f"b_UL = {b_ul:.6e}")
    else:
        print("\nDETECTION MODE: Δχ² above threshold; UL not printed unless --ul is set.")

    if args.d0 is not None:
        D0_anchor = float(args.d0)
        eps_hat = b_hat / D0_anchor
        se_eps = se_b / D0_anchor
        ci_eps = (eps_hat - 1.96 * se_eps, eps_hat + 1.96 * se_eps)
        print(f"\nAnchor D0 = {D0_anchor:.6e} Hz")
        print(f"FIT : eps = {eps_hat:.6e} ± {1.96*se_eps:.2e} (95% CI: [{ci_eps[0]:.6e}, {ci_eps[1]:.6e}])")
        if do_ul:
            eps_ul = b_ul / D0_anchor
            print(f"eps_UL = {eps_ul:.6e}  (same CL as b_UL)")
    else:
        print("\nNOTE: --d0 not provided → eps (and eps_UL) not reported. b is the primary constraint.")

    # Model curve (absolute)
    y_ref = float(f_center[ref_idx])
    y_model = b_hat * (g - g[ref_idx]) + y_ref

    print()
    header = "{:>8} {:>14} {:>14} {:>12}".format("z(nm)", "f_center", "model", "resid/σ")
    print(header)
    print("-" * len(header))
    for i in range(n):
        resid = (f_center[i] - y_model[i]) / sigma_center[i]
        print("{:8.1f} {:14.3f} {:14.3f} {:12.3f}".format(z_nm[i], f_center[i], y_model[i], resid))

    if not args.no_plots:
        plt.figure()
        plt.errorbar(z_nm, f_center, yerr=sigma_center, fmt="o", capsize=3)
        plt.plot(z_nm, y_model, marker=".")
        plt.xlabel("z (nm)")
        plt.ylabel("f_center (Hz)")
        plt.title("NV-Engine v0.23b: data vs fitted model (differential)")
        plt.grid(True)

        plt.figure()
        resid_hz = f_center - y_model
        plt.errorbar(z_nm, resid_hz, yerr=sigma_center, fmt="o", capsize=3)
        plt.axhline(0.0)
        plt.xlabel("z (nm)")
        plt.ylabel("residual (Hz)")
        plt.title("Residuals")
        plt.grid(True)

        # chi2 profile plot for b
        b_span = max(abs(b_hat), b_ul, 1.0)
        b_grid = np.linspace(0.0 if physical_nonneg else (b_hat - 4*b_span),
                             b_hat + 4*b_span, 400)
        # x = g - g_ref in differential space
        x = (g - g[ref_idx])
        chi2_grid = np.array([chi2_for_b(x, y_diff, sigma_diff, b) for b in b_grid])

        plt.figure()
        plt.plot(b_grid, chi2_grid)
        plt.axvline(b_hat)
        plt.axhline(chi2_fit + delta)
        if do_ul:
            plt.axvline(b_ul)
        plt.xlabel("b (Hz)")
        plt.ylabel("chi^2")
        plt.title("chi^2 profile vs b")
        plt.grid(True)

        plt.show()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        raise SystemExit(1)

