# =============================================================================
# ucm_rotation_curve_2d_sparse_v3_7_2_0.py
# v3.6.7 — 2D rotation-curve extraction with proper 2D->1D observables
#
# PDE system (same as before):
#   (∇² - 1/L²) Φ - k0² U = ρ
#   (∇² - k0²) U - nu*(∇²)^2 U = Φ
#
# Key idea (new):
#   In 2D, "v(R) from midline derivative" is often a bad observable.
#   We provide robust 2D diagnostics + 2D->1D reduction:
#
#   MODE A: "midline"      : v(R) from Phi(R,z≈0)  (legacy/for comparison)
#   MODE B: "rho_weighted" : v(R) from <gR>_rho(z)  (recommended)
#
# Visualizations:
#   1) normalized v(R)/vmax curves for all k0
#   2) diagnostic for one k0:
#       - Phi(R,z) midplane-averaged Phi0(R)
#       - gRbar(R) (rho-weighted)
#       - optional: midline gR(R,0) for comparison
#       - 2D map: log10 |g(R,z)|  where g = sqrt(gR^2+gZ^2)
#
# Practical knobs:
#   - mode = "rho_weighted"  (recommended)
#   - nu ~ (0.06..0.12)*dR^2
#   - nsl_mid = 3..7 (only for midline-avg extraction)
# =============================================================================
# UCM-T Rotation Curves Engine
# Frozen reference implementation for RC V12 (BENCH30 OUTER)
#
# Canonical reproducible bundle (code + data + runners) is archived on Zenodo:
# https://doi.org/10.5281/zenodo.18213329
#
# This file is provided for transparency and inspection.
# Development and experiments should be done in separate files/branches.

import numpy as np
import matplotlib.pyplot as plt
#print(">>> UCM-2D engine loaded: VSmooth build (w=21)")
from scipy.sparse import lil_matrix, bmat, identity
from scipy.sparse.linalg import spsolve
from scipy.interpolate import PchipInterpolator



# -----------------------------
# Disk model
# -----------------------------
def rho_disk(R, Z, rho0=1.0, Rd=4.0, hz=0.35):
    return rho0 * np.exp(-R / Rd) * np.exp(-np.abs(Z) / hz)


def idx(i, k, Nz):
    return i * Nz + k


# -----------------------------
# Build operators
# -----------------------------
def build_laplacians(Rmax, Zmax, Nr, Nz, L, k0_u=0.0):
    R = np.linspace(0.0, Rmax, Nr)
    Z = np.linspace(0.0, Zmax, Nz)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    inv_dR2 = 1.0 / (dR * dR)
    inv_dZ2 = 1.0 / (dZ * dZ)
    invL2 = 0.0 if L > 1e12 else 1.0 / (L * L)

    N = Nr * Nz
    A = lil_matrix((N, N), dtype=float)     # (∇² - 1/L²) for Phi
    Lap = lil_matrix((N, N), dtype=float)   # ∇² for U

    def alpha_robin(Lrobin, i, k, k_extra=0.0):
        r = np.sqrt(R[i] * R[i] + Z[k] * Z[k])
        termL = 0.0 if Lrobin > 1e12 else (1.0 / Lrobin)
        return termL + float(k_extra) + (1.0 / max(r, 1e-12))

    for i in range(Nr):
        for k in range(Nz):
            p = idx(i, k, Nz)

            # symmetry at R=0
            if i == 0:
                A[p, p] = 1.0
                A[p, idx(1, k, Nz)] = -1.0
                Lap[p, p] = 1.0
                Lap[p, idx(1, k, Nz)] = -1.0
                continue

            # symmetry at z=0
            if k == 0:
                A[p, p] = 1.0
                A[p, idx(i, 1, Nz)] = -1.0
                Lap[p, p] = 1.0
                Lap[p, idx(i, 1, Nz)] = -1.0
                continue

            # Robin at outer R boundary
            if i == Nr - 1:
                # Phi open BC
                a_phi = alpha_robin(L, i, k, k_extra=0.0)
                A[p, p] = (1.0 / dR) + a_phi
                A[p, idx(i - 1, k, Nz)] = -(1.0 / dR)

                # U Yukawa-consistent open BC: alpha = k0 + 1/r
                a_u = alpha_robin(1e15, i, k, k_extra=k0_u)
                Lap[p, p] = (1.0 / dR) + a_u
                Lap[p, idx(i - 1, k, Nz)] = -(1.0 / dR)
                continue

            # Robin at outer Z boundary
            if k == Nz - 1:
                a_phi = alpha_robin(L, i, k, k_extra=0.0)
                A[p, p] = (1.0 / dZ) + a_phi
                A[p, idx(i, k - 1, Nz)] = -(1.0 / dZ)

                a_u = alpha_robin(1e15, i, k, k_extra=k0_u)
                Lap[p, p] = (1.0 / dZ) + a_u
                Lap[p, idx(i, k - 1, Nz)] = -(1.0 / dZ)
                continue

            # interior: axisymmetric Laplacian
            Ri = R[i]
            aR = inv_dR2 - 1.0 / (2.0 * Ri * dR)
            cR = inv_dR2 + 1.0 / (2.0 * Ri * dR)
            bR = -2.0 * inv_dR2

            aZ = inv_dZ2
            cZ = inv_dZ2
            bZ = -2.0 * inv_dZ2

            Lap[p, p] = (bR + bZ)
            Lap[p, idx(i - 1, k, Nz)] = aR
            Lap[p, idx(i + 1, k, Nz)] = cR
            Lap[p, idx(i, k - 1, Nz)] = aZ
            Lap[p, idx(i, k + 1, Nz)] = cZ

            A[p, p] = (bR + bZ - invL2)
            A[p, idx(i - 1, k, Nz)] = aR
            A[p, idx(i + 1, k, Nz)] = cR
            A[p, idx(i, k - 1, Nz)] = aZ
            A[p, idx(i, k + 1, Nz)] = cZ

    return R, Z, A.tocsr(), Lap.tocsr()


# -----------------------------
# Solve coupled system
# -----------------------------
def solve_twofield(Rmax=100.0, Zmax=8.0, Nr=260, Nz=240,
                   L=20.0, k0=0.0, Rd=4.0, hz=0.35,
                   nu=0.0):
    R, Z, A, Lap = build_laplacians(Rmax, Zmax, Nr, Nz, L, k0_u=k0)

    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    rho = rho_disk(RR, ZZ, Rd=Rd, hz=hz).reshape(-1)

    N = Nr * Nz
    I = identity(N, format="csr")

    Bu = Lap - (k0 * k0) * I
    if nu and nu > 0.0:
        # numerical stabilizer: -nu*(∇²)^2 U
        Lap2 = (Lap @ Lap).tocsr()
        Bu = Bu - float(nu) * Lap2

    M = bmat([[A, (-k0 * k0) * I],
              [(-1.0) * I, Bu]], format="csr")

    rhs = np.concatenate([rho, np.zeros_like(rho)])
    sol = spsolve(M, rhs)

    Phi = sol[:N].reshape((Nr, Nz))
    U = sol[N:].reshape((Nr, Nz))
    return R, Z, Phi, U


# -----------------------------
# Gradients and observables
# -----------------------------
def grad_RZ(Phi, dR, dZ):
    """Compute dPhi/dR and dPhi/dZ with stable central + one-sided at edges."""
    dPhi_dR = np.zeros_like(Phi)
    dPhi_dZ = np.zeros_like(Phi)

    dPhi_dR[1:-1, :] = (Phi[2:, :] - Phi[:-2, :]) / (2.0 * dR)
    dPhi_dR[0, :] = (Phi[1, :] - Phi[0, :]) / dR
    dPhi_dR[-1, :] = (Phi[-1, :] - Phi[-2, :]) / dR

    dPhi_dZ[:, 1:-1] = (Phi[:, 2:] - Phi[:, :-2]) / (2.0 * dZ)
    dPhi_dZ[:, 0] = (Phi[:, 1] - Phi[:, 0]) / dZ
    dPhi_dZ[:, -1] = (Phi[:, -1] - Phi[:, -2]) / dZ

    return dPhi_dR, dPhi_dZ


def phi_midplane_avg(Phi, nsl=5):
    nsl = int(max(1, nsl))
    nsl = min(nsl, Phi.shape[1])
    return np.mean(Phi[:, :nsl], axis=1)


def v_from_midline(R, Phi, nsl_mid=5, *, grad_method: str = "fd"):
    """
    Legacy-ish: build v(R) from Phi averaged over lowest z-layers,
    then finite-difference derivative in R (central).
    This is for comparison; in 2D it can show ripple.
    """
    Phi0 = phi_midplane_avg(Phi, nsl=nsl_mid)
    dR = R[1] - R[0]
    dPhi = dphi_dr_1d(R, Phi0, method=("pchip" if str(grad_method).lower().strip() in ("pchip","pchip1d") else "fd"))
    gR = -dPhi
    v = np.sqrt(np.maximum(R * np.abs(gR), 0.0))
    return v, gR, Phi0


def dphi_dr_1d(R: np.ndarray, Phi_eff: np.ndarray, method: str = "pchip"):
    """Stable 1D derivative dPhi/dR using shape-preserving interpolation.
    Falls back to finite-difference if SciPy interpolator is unavailable.
    """
    R = np.asarray(R, float).ravel()
    Phi_eff = np.asarray(Phi_eff, float).ravel()
    dR = float(R[1] - R[0]) if R.size > 1 else 1.0

    m = str(method or "pchip").lower().strip()
    if R.size < 5:
        return np.gradient(Phi_eff, dR)

    if m in ("pchip", "pchip1d", "shape"):
        try:
            f = PchipInterpolator(R, Phi_eff, extrapolate=True)
            return f.derivative()(R)
        except Exception:
            return np.gradient(Phi_eff, dR)

    # conservative fallback
    return np.gradient(Phi_eff, dR)



def _odd_window(w: int) -> int:
    w = int(max(1, w))
    if w % 2 == 0:
        w += 1
    return w


def smooth_1d_box(x: np.ndarray, w: int) -> np.ndarray:
    """Simple low-pass via moving average with reflect padding."""
    w = _odd_window(w)
    if w <= 1:
        return x
    x = np.asarray(x, float)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kern = np.ones(w, float) / float(w)
    return np.convolve(xp, kern, mode="valid")


def smooth_1d_savgol(x: np.ndarray, w: int, poly: int = 2) -> np.ndarray:
    """Savitzky–Golay smoothing (if scipy.signal is available), else falls back to box."""
    w = _odd_window(w)
    if w <= 3:
        return x
    try:
        from scipy.signal import savgol_filter
        poly = int(max(1, min(poly, w - 2)))
        return savgol_filter(np.asarray(x, float), window_length=w, polyorder=poly, mode="interp")
    except Exception:
        return smooth_1d_box(np.asarray(x, float), w)


def smooth_1d_adaptive_box(x: np.ndarray, w0: int, w1: int, i_start: int) -> np.ndarray:
    """
    Adaptive moving average: window grows linearly from w0 to w1 starting at i_start.
    Designed to suppress high-frequency ripple on the outer tail without distorting inner rise.
    """
    x = np.asarray(x, float)
    n = x.size
    w0 = _odd_window(w0)
    w1 = _odd_window(max(w1, w0))
    i_start = int(np.clip(i_start, 0, max(0, n - 1)))
    out = np.empty_like(x)

    # pre-fill inner with constant window
    out[:] = smooth_1d_box(x, w0)

    if n <= 1 or w1 == w0 or i_start >= n - 1:
        return out

    for i in range(i_start, n):
        t = 0.0 if n - 1 == i_start else (i - i_start) / float(n - 1 - i_start)
        wi = int(round(w0 + t * (w1 - w0)))
        wi = _odd_window(wi)
        half = wi // 2
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(x[lo:hi])
    return out


def v_from_rho_weighted(R, Z, Phi, Rd=4.0, hz=0.35,
                        *, grad_method: str = "fd2d",
                           obs_smooth: str = "none",
                        smooth_w0: int = 11,
                        smooth_w1: int = 31,
                        smooth_tail_frac: float = 0.7,
                        sg_poly: int = 2,
                        force_attractive: bool = False,
                        force_attractive_eps: float = 0.0, zcut_factor: float = 6.0):
    """
    Recommended 2D->1D observable:
      gR(R,z) from Phi
      gRbar(R) = ∫ gR rho dz / ∫ rho dz
      v(R) = sqrt( R * |gRbar| )

    Additional stabilization:
      - Optionally apply a 1D low-pass filter to gRbar(R) (or adaptive tail smoothing)
        to suppress residual numerical ripple ("sawtooth") at large R.
    """
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    # --- smooth Phi BEFORE derivatives (helps but may not fully remove tail ripple)
    Phi_s = Phi.copy()

    # smooth along Z (main effect)
    for i in range(Phi_s.shape[0]):
        Phi_s[i, :] = smooth_box(Phi_s[i, :], w=11)

    # light smooth along R
    for k in range(Phi_s.shape[1]):
        Phi_s[:, k] = smooth_box(Phi_s[:, k], w=5)

    # --- choose how we compute the radial acceleration observable ---
    gm = str(grad_method or "fd2d").lower().strip()

    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    rho = rho_disk(RR, ZZ, Rd=Rd, hz=hz)
    z_cut = float(zcut_factor) * hz
    mz = (Z <= z_cut)
    rho_c = rho[:, mz]
    denom = np.sum(rho_c, axis=1) + 1e-30

    if gm in ("pchip", "pchip1d", "phi_pchip", "phi1d"):
        # 1D reduction FIRST (Phi_eff), then stable derivative via PCHIP
        Phi_eff = np.sum(Phi_s[:, mz] * rho_c, axis=1) / denom
        dPhi_eff = dphi_dr_1d(R, Phi_eff, method="pchip")
        gRbar_raw = -dPhi_eff

        # for diagnostics only (keep shapes consistent)
        dPhi_dR, dPhi_dZ = grad_RZ(Phi_s, dR, dZ)
        gR = -dPhi_dR
        gZ = -dPhi_dZ
        g = np.sqrt(gR * gR + gZ * gZ)

    else:
        # default: 2D derivative FIRST, then rho-weighted reduction
        dPhi_dR, dPhi_dZ = grad_RZ(Phi_s, dR, dZ)
        gR = -dPhi_dR
        gZ = -dPhi_dZ
        g = np.sqrt(gR * gR + gZ * gZ)

        gR_c = gR[:, mz]
        gRbar_raw = np.sum(gR_c * rho_c, axis=1) / denom

    # --- optional: enforce inward (attractive) radial field to suppress sign-flip cusps
    # Numerical ripple can push gRbar(R) through zero; using abs() then creates sharp cusps in v(R).
    # When enabled, we clamp gRbar to a small negative floor and compute v from (-gRbar).
    if force_attractive:
        arr = np.asarray(gRbar_raw, float)
        scale = float(np.median(np.abs(arr))) if arr.size else 0.0
        eps = float(force_attractive_eps) if (force_attractive_eps and force_attractive_eps > 0.0) else (
            1e-8 * scale if scale > 0.0 else 1e-12
        )
        gRbar_raw = np.where(gRbar_raw > -eps, -eps, gRbar_raw)

    # --- observable-level smoothing (optional)
    obs_smooth = str(obs_smooth or "none").lower().strip()
    if obs_smooth in ("none", "off", "0"):
        gRbar = gRbar_raw
    elif obs_smooth in ("box", "ma", "moving", "moving_average"):
        gRbar = smooth_1d_box(gRbar_raw, smooth_w0)
    elif obs_smooth in ("savgol", "sgolay", "savitzky"):
        gRbar = smooth_1d_savgol(gRbar_raw, smooth_w0, poly=sg_poly)
    elif obs_smooth in ("adaptive", "adaptive_box", "tail_box"):
        i0 = int(max(0, min(len(R) - 1, round(float(smooth_tail_frac) * (len(R) - 1)))))
        gRbar = smooth_1d_adaptive_box(gRbar_raw, smooth_w0, smooth_w1, i0)
    else:
        # unknown method -> be conservative
        gRbar = gRbar_raw

    # ensure inward field after smoothing as well (avoids residual sign flips)
    if force_attractive:
        arr = np.asarray(gRbar, float)
        scale = float(np.median(np.abs(arr))) if arr.size else 0.0
        eps = float(force_attractive_eps) if (force_attractive_eps and force_attractive_eps > 0.0) else (
            1e-8 * scale if scale > 0.0 else 1e-12
        )
        gRbar = np.where(gRbar > -eps, -eps, gRbar)

    if force_attractive:
        v = np.sqrt(np.maximum(R * (-gRbar), 0.0))
    else:
        v = np.sqrt(np.maximum(R * np.abs(gRbar), 0.0))
    return v, gRbar, g, gR, gZ


def saw_index(y: np.ndarray) -> float:
    """Median |2nd diff| normalized by median |y| (robust ripple indicator)."""
    y = np.asarray(y, float).ravel()
    if y.size < 5:
        return float("nan")
    d2 = np.diff(y, n=2)
    num = float(np.median(np.abs(d2)))
    den = float(np.median(np.abs(y))) + 1e-30
    return num / den


def provall_index(v_tail: np.ndarray) -> float:
    """min(v_tail)/median(v_tail) — small values signal deep 'dips'."""
    v_tail = np.asarray(v_tail, float).ravel()
    if v_tail.size < 3:
        return float("nan")
    med = float(np.median(v_tail)) + 1e-30
    return float(np.min(v_tail)) / med


def tail_slope(R, v, R1, R2, frac=1e-4):
    R = np.asarray(R).ravel()
    v = np.asarray(v).ravel()
    vmax = float(np.max(v)) if v.size else 0.0
    if vmax <= 0:
        return np.nan, 0
    m = (R >= R1) & (R <= R2) & (R > 0) & (v > frac * vmax)
    n = int(np.sum(m))
    if n < 12:
        return np.nan, n
    s = np.gradient(np.log(v[m]), np.log(R[m]))
    return float(np.mean(s)), n


# -----------------------------
# Plateau metrics (NEW)
# -----------------------------
def smooth_box(y, w=7):
    """Simple box smoother (odd window). Keeps length."""
    y = np.asarray(y, float)
    w = int(max(1, w))
    if w == 1:
        return y.copy()
    if w % 2 == 0:
        w += 1
    k = w // 2
    pad = np.pad(y, (k, k), mode="edge")
    kernel = np.ones(w, float) / float(w)
    return np.convolve(pad, kernel, mode="valid")


def log_slope_profile(R, v, vmin_frac=1e-4, smooth_w=7):
    """
    s(R) = d ln v / d ln R
    Returns:
      s (same length as R), mask_valid
    """
    R = np.asarray(R, float)
    v = np.asarray(v, float)

    vmax = float(np.max(v)) if v.size else 0.0
    if vmax <= 0:
        return np.full_like(R, np.nan), np.zeros_like(R, dtype=bool)
    
    # --- EXTRA smoothing on v(R) itself to kill "sawtooth" (envelope/trend)
    v_env = smooth_box(v, w=121)      # main trend (81 points)
    v_s   = smooth_box(v_env, w=31)  # light additional smoothing before slope

    m = (R > 0) & np.isfinite(R) & np.isfinite(v_s) & (v_s > vmin_frac * vmax)
    s = np.full_like(R, np.nan)

    # compute slope only on valid points (using gradients in log space)
    lr = np.log(np.maximum(R[m], 1e-30))
    lv = np.log(np.maximum(v_s[m], 1e-30))
    if lr.size >= 5:
        s_m = np.gradient(lv, lr)
        if smooth_w and smooth_w > 1:
            s_m = smooth_box(s_m, w=smooth_w)
        s[m] = s_m

    return s, m


def longest_true_run(mask):
    """
    Find longest contiguous run of True in a boolean mask.
    Returns (i0, i1, length) where i1 is inclusive.
    If none, returns (-1,-1,0).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return -1, -1, 0
    best_len = 0
    best_i0 = -1
    cur_len = 0
    cur_i0 = 0
    for i, ok in enumerate(mask):
        if ok:
            if cur_len == 0:
                cur_i0 = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_i0 = cur_i0
        else:
            cur_len = 0
    if best_len <= 0:
        return -1, -1, 0
    return best_i0, best_i0 + best_len - 1, best_len


def plateau_detector(R, v, Rmin=24.0, Rmax=80.0,
                     s_max=0.25, min_width=6.0,
                     vmin_frac=1e-3, smooth_w=31):
    """
    Detect plateau as longest region where |s(R)| < s_max
    within [Rmin, Rmax], and width >= min_width.

    Returns dict with:
      ok, R0, R1, width, s_mean, s_std, npts
      plus 's_profile' for optional plotting
    """
    R = np.asarray(R, float)
    v = np.asarray(v, float)

    s, valid = log_slope_profile(R, v, vmin_frac=vmin_frac, smooth_w=smooth_w)

    wmask = valid & np.isfinite(s) & (R >= Rmin) & (R <= Rmax) & (np.abs(s) <= s_max)
    i0, i1, n = longest_true_run(wmask)

    if n <= 0:
        return dict(ok=False, R0=np.nan, R1=np.nan, width=0.0,
                    s_mean=np.nan, s_std=np.nan, npts=0, s_profile=s)

    R0 = float(R[i0])
    R1 = float(R[i1])
    width = float(R1 - R0)

    if width < float(min_width):
        return dict(ok=False, R0=R0, R1=R1, width=width,
                    s_mean=np.nan, s_std=np.nan, npts=n, s_profile=s)

    seg = s[i0:i1+1]
    return dict(ok=True, R0=R0, R1=R1, width=width,
                s_mean=float(np.nanmean(seg)), s_std=float(np.nanstd(seg)),
                npts=int(n), s_profile=s)


def plateau_slope(R, v, r1_frac=0.6, r2_frac=0.9):
    """
    Estimate slope of v(R) in the outer region.
    """
    Rmax = R.max()
    mask = (R > r1_frac * Rmax) & (R < r2_frac * Rmax)

    if np.sum(mask) < 5:
        return np.nan

    coeffs = np.polyfit(R[mask], v[mask], 1)
    return coeffs[0]


def parse_list(s: str, cast=float):
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(cast(part))
    return out


def run_for_L(*, mode: str, L: float, Rmax: float, Zmax: float, Nr: int, Nz: int,
              Rd: float, hz: float, nu: float,
              k0_list: list[float],
              win1: tuple[float, float], win2: tuple[float, float],
              plat_win: tuple[float, float], s_max: float, min_width: float, smooth_w: int,
              diag_k0: float, no_plots: bool) -> None:
    """
    mode: 'rho_weighted' or 'midline'
    Scan over k0_list and print one table for this L.
    If diag_k0 in k0_list, produce diagnostic plots for that k0 (unless no_plots).
    """
    print("=== v3.7.1 CLEAN — IR two-field (2D) + proper observables + plateau detector ===")
    print(f"mode={mode}")
    print(f"L={L:.1f}, Rmax={Rmax:.1f}, Zmax={Zmax:.1f}, Nr={Nr}, Nz={Nz}")
    print(f"nu={nu:.3e}, Rd={Rd}, hz={hz}")
    print(f"tail1={win1}, tail2={win2}")
    print("{:>6} {:>10} {:>6}   {:>10} {:>6}   {:>7} {:>7} {:>8} {:>9}   {:>7} {:>7}   {:>12} {:>12}".format(
        "k0", "slope1", "N1", "slope2", "N2", "R0", "R1", "dRplat", "splat", "saw_v", "prov", "vmax", "max|U|"
    ))
    print("-" * 110)

    diag_done = False

    for k0 in k0_list:
        # Solve fields
        R, Z, Phi, U = solve_twofield(Rmax=Rmax, Zmax=Zmax, Nr=Nr, Nz=Nz,
                                     L=L, k0=float(k0), Rd=Rd, hz=hz, nu=nu)

        # Observable
        if mode == "rho_weighted":
            v, gRbar, g, gR, gZ = v_from_rho_weighted(R, Z, Phi, Rd=Rd, hz=hz, grad_method=args.grad_method)
        elif mode == "midline":
            v, gRbar, Phi0 = v_from_midline(R, Phi, nsl_mid=3, grad_method=("pchip" if args.grad_method=="pchip" else "fd"))
        else:
            raise ValueError("mode must be 'rho_weighted' or 'midline'")

        # Legacy tail slopes (kept as diagnostics)
        s1, n1 = tail_slope(R, v, *win1, frac=1e-3)
        s2, n2 = tail_slope(R, v, *win2, frac=1e-3)

        # Plateau detector (primary metric)
        plat = plateau_detector(R, v,
                                Rmin=plat_win[0], Rmax=plat_win[1],
                                s_max=s_max, min_width=min_width,
                                vmin_frac=1e-3, smooth_w=smooth_w)
        dRplat = plat["width"] if plat["ok"] else 0.0
        splat = plat["s_mean"] if plat["ok"] else float("nan")
        R0 = plat["R0"] if plat["ok"] else 0.0
        R1 = plat["R1"] if plat["ok"] else 0.0

        vmax = float(np.max(v)) if np.size(v) else float("nan")
        umax = float(np.max(np.abs(U))) if np.size(U) else float("nan")

        m_tail = (R >= float(win2[0])) & (R <= float(win2[1])) & np.isfinite(R) & np.isfinite(v)
        v_tail = v[m_tail]
        saw_v = saw_index(v_tail)
        prov = provall_index(v_tail)

        print("{:6.2f} {:10.3f} {:6d}   {:10.3f} {:6d}   {:7.2f} {:7.2f} {:8.2f} {:9.3f}   {:7.3f} {:7.3f}   {:12.3e} {:12.3e}".format(
            float(k0), s1, int(n1), s2, int(n2),
            float(R0), float(R1), float(dRplat),
            (float(splat) if np.isfinite(splat) else float("nan")),
            (float(saw_v) if np.isfinite(saw_v) else float("nan")),
            (float(prov) if np.isfinite(prov) else float("nan")),
            float(vmax), float(umax)
        ))

        # Diagnostics for one k0 (plots + debug)
        if (not no_plots) and (not diag_done) and (abs(float(k0) - float(diag_k0)) < 1e-12):
            diag_done = True

            # --- debug line on slope profile
            sprof = plat["s_profile"]
            w = (R >= plat_win[0]) & (R <= plat_win[1]) & np.isfinite(sprof)
            if np.any(w):
                print(f"    debug: k0={k0:.2f}  min|s|={np.nanmin(np.abs(sprof[w])):.3f}  "
                      f"mean|s|={np.nanmean(np.abs(sprof[w])):.3f}  "
                      f"frac(|s|<{s_max})={np.mean(np.abs(sprof[w]) < s_max):.2f}")

            # --- plots
            plt.figure()
            plt.title(f"v(R) ({mode}), L={L:g}, k0={k0:g}")
            plt.plot(R, v, label="v(R)")
            if plat["ok"]:
                plt.axvspan(plat["R0"], plat["R1"], alpha=0.2, label=f"plateau ΔR={plat['width']:.2f}")
            plt.xlabel("R"); plt.ylabel("v"); plt.grid(True); plt.legend()

            plt.figure()
            plt.title(f"s(R)=d ln v/d ln R, L={L:g}, k0={k0:g}")
            plt.plot(R, sprof, label="s(R)")
            plt.axhline(+s_max, linestyle="--")
            plt.axhline(-s_max, linestyle="--")
            plt.axvline(plat_win[0], linestyle=":")
            plt.axvline(plat_win[1], linestyle=":")
            if plat["ok"]:
                plt.axvspan(plat["R0"], plat["R1"], alpha=0.2, label=f"<s>={plat['s_mean']:.3f}")
            plt.xlabel("R"); plt.ylabel("s"); plt.grid(True); plt.legend()

            # 2D map of log10|g|
            if mode == "rho_weighted":
                gmag = np.sqrt(gR**2 + gZ**2)
                eps = 1e-30
                plt.figure()
                plt.title(f"log10|g(R,Z)|, L={L:g}, k0={k0:g}")
                plt.imshow(np.log10(np.abs(gmag) + eps),
                           origin="lower", aspect="auto",
                           extent=[Z.min(), Z.max(), R.min(), R.max()])
                plt.xlabel("Z"); plt.ylabel("R")
                plt.colorbar(label="log10|g|")
                
            # =============================
            # FIGURE 1: visualization
            # =============================
            R_plot = R
            Z_plot = Z
            g_plot = gR
            v_plot = v

            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

            # Общий стиль
            for ax in axs:
                ax.tick_params(direction='in', length=4, width=1)
                ax.grid(False)

            # --- (A) 2D field ---
            ax = axs[0]
            im = ax.imshow(
                np.log10(np.abs(g_plot)),
                extent=[R_plot.min(), R_plot.max(), Z_plot.min(), Z_plot.max()],
                origin='lower',
                aspect='auto',
                cmap='magma'
            )

            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label(r'$\log_{10}|g_R|$')

            ax.set_xlabel(r'$R$ [arb. units]')
            ax.set_ylabel(r'$Z$ [arb. units]')
            ax.set_title('(A) Acceleration field')

            # --- (B) Rotation curve ---
            ax2 = axs[1]
            ax2.plot(R_plot, v_plot, lw=2, color='black')

            R1 = 0.4 * R_plot.max()
            R2 = 0.7 * R_plot.max()
            ax2.axvspan(R1, R2, color='gray', alpha=0.25)

            ax2.set_xlabel(r'$R$ [arb. units]')
            ax2.set_ylabel(r'$v(R)$')
            ax2.set_title('(B) Rotation curve')

            ax2.text(
                0.5 * R_plot.max(),
                0.9 * np.max(v_plot),
                r'$v(R)=\sqrt{R\langle g_R\rangle}$',
                fontsize=10
            )

            plt.tight_layout()
            plt.savefig("figure1.pdf", dpi=300, bbox_inches='tight')
            plt.savefig("figure1.png", dpi=300, bbox_inches='tight')
            plt.show()

    print("-" * 110)
    print()


def create_phase_diagram(*,
                         Rmax: float = 100.0, Zmax: float = 8.0,
                         Nr: int = 260, Nz: int = 240,
                         Rd: float = 4.0, hz: float = 0.35,
                         nu: float = 1.491e-02,
                         k0_min: float = 0.02, k0_max: float = 0.30, Nk0: int = 12,
                         L_min: float = 5.0, L_max: float = 40.0, NL: int = 12,
                         plat_win: tuple[float, float] = (30.0, 75.0),
                         s_max: float = 0.30, min_width: float = 6.0, smooth_w: int = 15,
                         out_prefix: str = "figure2_phase_diagram"):
    """Create a phase diagram over (k0, L).

    Metric (color): detected plateau width ΔR within plat_win using plateau_detector().
    This produces a clean, journal-ready heatmap without changing the physics code.
    """
    k0_vals = np.linspace(k0_min, k0_max, Nk0)
    L_vals  = np.linspace(L_min, L_max, NL)
    phase_map = np.zeros((len(L_vals), len(k0_vals)), dtype=float)

    for i, L in enumerate(L_vals):
        for j, k0 in enumerate(k0_vals):
            R, Z, Phi, U = solve_twofield(Rmax=Rmax, Zmax=Zmax, Nr=Nr, Nz=Nz,
                                         L=float(L), k0=float(k0), Rd=Rd, hz=hz, nu=nu)

            v, gRbar, g, gR, gZ = v_from_rho_weighted(R, Z, Phi, Rd=Rd, hz=hz, grad_method="fd2d")

            plat = plateau_detector(R, v,
                                    Rmin=plat_win[0], Rmax=plat_win[1],
                                    s_max=s_max, min_width=min_width,
                                    vmin_frac=1e-3, smooth_w=smooth_w)

            phase_map[i, j] = plat["width"] if plat["ok"] else 0.0

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    im = ax.imshow(
        phase_map,
        origin='lower',
        aspect='auto',
        extent=[k0_vals.min(), k0_vals.max(), L_vals.min(), L_vals.max()],
        cmap='viridis'
    )

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'Plateau width $\Delta R$')

    ax.set_xlabel(r'$k_0$')
    ax.set_ylabel(r'$L$')
    ax.set_title('Phase diagram: plateau width')

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches='tight')
    plt.show()

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="UCM-Engine 2D rotation curves — clean release (v3.7.1)")
    ap.add_argument("--mode", default="rho_weighted", choices=["rho_weighted", "midline"],
                    help="Observable mode (default: rho_weighted).")
    ap.add_argument("--grad-method", default="fd2d", choices=["fd2d", "pchip"],
                    help="How to compute dPhi/dR for observables: fd2d (default) or pchip (1D Phi_eff + PCHIP).")
    ap.add_argument("--scan", action="store_true",
                    help="Enable scan over L list (otherwise runs single L).")
    ap.add_argument("--L", type=float, default=20.0, help="Single L value (if --scan not set).")
    ap.add_argument("--L-list", type=str, default="20",
                    help="Comma list of L values for --scan, e.g. '10,15,20,30'.")
    ap.add_argument("--k0-list", type=str, default="0.02,0.10,0.15,0.20,0.25,0.30,0.35,0.40",
                    help="Comma list of k0 values.")
    ap.add_argument("--diag-k0", type=float, default=0.20,
                    help="k0 value to plot diagnostics for (must be in k0-list to trigger).")
    ap.add_argument("--no-plots", action="store_true", help="Disable diagnostic plots.")
    ap.add_argument("--phase-diagram", action="store_true",
                    help="Generate the phase diagram (Figure 2) and exit.")

    # Grid / geometry
    ap.add_argument("--Rmax", type=float, default=100.0)
    ap.add_argument("--Zmax", type=float, default=8.0)
    ap.add_argument("--Nr", type=int, default=260)
    ap.add_argument("--Nz", type=int, default=240)

    # Model params
    ap.add_argument("--Rd", type=float, default=4.0)
    ap.add_argument("--hz", type=float, default=0.35)
    ap.add_argument("--nu", type=float, default=1.491e-02)

    # Metrics
    ap.add_argument("--tail1", type=str, default="24,48")
    ap.add_argument("--tail2", type=str, default="48,80")
    ap.add_argument("--plat-win", type=str, default="30,75")
    ap.add_argument("--s-max", type=float, default=0.30)
    ap.add_argument("--min-width", type=float, default=6.0)
    ap.add_argument("--smooth-w", type=int, default=15)

    args = ap.parse_args(argv)

    win1 = tuple(parse_list(args.tail1, float)) if args.tail1 else (24.0, 48.0)
    win2 = tuple(parse_list(args.tail2, float)) if args.tail2 else (48.0, 80.0)
    plat_win = tuple(parse_list(args.plat_win, float)) if args.plat_win else (30.0, 75.0)
    if len(win1) != 2 or len(win2) != 2 or len(plat_win) != 2:
        raise ValueError("tail1/tail2/plat-win must be 'a,b'.")
    if args.phase_diagram:
        create_phase_diagram(Rmax=args.Rmax, Zmax=args.Zmax, Nr=args.Nr, Nz=args.Nz,
                             Rd=args.Rd, hz=args.hz, nu=args.nu,
                             plat_win=plat_win, s_max=args.s_max,
                             min_width=args.min_width, smooth_w=args.smooth_w)
        return



    k0_list = parse_list(args.k0_list, float)
    if not k0_list:
        raise ValueError("Empty k0-list.")

    if args.scan:
        L_list = parse_list(args.L_list, float)
        if not L_list:
            raise ValueError("Empty L-list.")
        for L in L_list:
            run_for_L(mode=args.mode, L=float(L),
                      Rmax=args.Rmax, Zmax=args.Zmax, Nr=args.Nr, Nz=args.Nz,
                      Rd=args.Rd, hz=args.hz, nu=args.nu,
                      k0_list=k0_list,
                      win1=win1, win2=win2,
                      plat_win=plat_win, s_max=args.s_max,
                      min_width=args.min_width, smooth_w=args.smooth_w,
                      diag_k0=args.diag_k0, no_plots=args.no_plots)
    else:
        run_for_L(mode=args.mode, L=float(args.L),
                  Rmax=args.Rmax, Zmax=args.Zmax, Nr=args.Nr, Nz=args.Nz,
                  Rd=args.Rd, hz=args.hz, nu=args.nu,
                  k0_list=k0_list,
                  win1=win1, win2=win2,
                  plat_win=plat_win, s_max=args.s_max,
                  min_width=args.min_width, smooth_w=args.smooth_w,
                  diag_k0=args.diag_k0, no_plots=args.no_plots)


if __name__ == "__main__":
    main()
