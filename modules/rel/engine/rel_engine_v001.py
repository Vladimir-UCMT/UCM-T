from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def omega_forward(
    k: tuple[float, float, float],
    v0: tuple[float, float, float],
    c0: float,
) -> float:
    """Forward branch: ω = v0·k + c0|k| (Eq. 16)."""
    kn = _norm(k)
    return _dot(v0, k) + c0 * kn


def group_velocity(
    k: tuple[float, float, float],
    v0: tuple[float, float, float],
    c0: float,
) -> tuple[float, float, float]:
    """Group velocity: vg = v0 + c0 k/|k| (Eq. 17)."""
    kn = _norm(k)
    if kn <= 0.0:
        raise ValueError("k must be non-zero for group_velocity")
    return (v0[0] + c0 * k[0] / kn, v0[1] + c0 * k[1] / kn, v0[2] + c0 * k[2] / kn)


def loop_phase_from_circulation(omega: float, chi: float, c0: float, circulation: float) -> float:
    """ΔΦ_loop = (ω/(χ c0^2)) ∮ v0·dl  (Eq. 27)."""
    if chi == 0.0 or c0 == 0.0:
        raise ValueError("chi and c0 must be non-zero")
    return omega * circulation / (chi * c0 * c0)


def sagnac_phase(omega: float, chi: float, c0: float, omega_dot_area: float) -> float:
    """ΔΦ_Sag = (2ω/(χ c0^2)) Ω·A  (Eq. 29)."""
    if chi == 0.0 or c0 == 0.0:
        raise ValueError("chi and c0 must be non-zero")
    return 2.0 * omega * omega_dot_area / (chi * c0 * c0)

def acoustic_ds2_1d(dt: float, dx: float, v0: float, c0: float, rho0: float = 1.0) -> float:
    """
    Acoustic interval in 1D (Eq. 24):
      ds^2 = (rho0/c0) * ( -c0^2 dt^2 + (dx - v0 dt)^2 )
    """
    if c0 == 0.0:
        raise ValueError("c0 must be non-zero")
    return (rho0 / c0) * (-(c0 * c0) * (dt * dt) + (dx - v0 * dt) * (dx - v0 * dt))

def acoustic_metric_g_1d(v0: float, c0: float, rho0: float = 1.0) -> list[list[float]]:
    """g_{μν} for (t,x) in 1D (Eq. 22)."""
    if c0 == 0.0:
        raise ValueError("c0 must be non-zero")
    pref = rho0 / c0
    return [
        [pref * (-(c0 * c0 - v0 * v0)), pref * (-v0)],
        [pref * (-v0), pref * 1.0],
    ]


def acoustic_metric_ginv_1d(v0: float, c0: float, rho0: float = 1.0) -> list[list[float]]:
    """g^{μν} for (t,x) in 1D (Eq. 23)."""
    if c0 == 0.0 or rho0 == 0.0:
        raise ValueError("c0 and rho0 must be non-zero")
    pref = 1.0 / (rho0 * c0)
    return [
        [pref * (-1.0), pref * (-v0)],
        [pref * (-v0), pref * (c0 * c0 - v0 * v0)],
    ]


def _mat2_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]

def null_speeds_1d(v0: float, c0: float) -> tuple[float, float]:
    """Characteristic speeds from ds^2=0: dx/dt = v0 ± c0."""
    return (v0 - c0, v0 + c0)


def is_null_1d(dt: float, dx: float, v0: float, c0: float, rho0: float = 1.0, tol: float = 1e-12) -> bool:
    """Check ds^2≈0 using Eq.24 implementation."""
    return abs(acoustic_ds2_1d(dt=dt, dx=dx, v0=v0, c0=c0, rho0=rho0)) <= tol

def _unit(n: tuple[float, float, float]) -> tuple[float, float, float]:
    nn = _norm(n)
    if nn <= 0.0:
        raise ValueError("direction must be non-zero")
    return (n[0] / nn, n[1] / nn, n[2] / nn)


def null_speeds_along(
    n: tuple[float, float, float],
    v0: tuple[float, float, float],
    c0: float,
) -> tuple[float, float]:
    """Speeds along direction n-hat: v0·n̂ ± c0."""
    nh = _unit(n)
    proj = _dot(v0, nh)
    return (proj - c0, proj + c0)


def find_horizon_x(xs: list[float], vs: list[float], c0: float) -> float:
    """Find x_H where v0 crosses c0 by linear interpolation (v0(x_H)=c0)."""
    if len(xs) != len(vs) or len(xs) < 2:
        raise ValueError("xs and vs must have same length >= 2")
    for i in range(len(xs) - 1):
        v1, v2 = vs[i], vs[i + 1]
        if (v1 - c0) == 0.0:
            return xs[i]
        if (v1 - c0) * (v2 - c0) <= 0.0:
            # linear interpolation between (x1,v1) and (x2,v2)
            x1, x2 = xs[i], xs[i + 1]
            if v2 == v1:
                return x1
            t = (c0 - v1) / (v2 - v1)
            return x1 + t * (x2 - x1)
    raise ValueError("no horizon crossing found")

def interp1_linear(xs: list[float], ys: list[float], x: float) -> float:
    """Piecewise-linear interpolation for monotonic/unsorted xs (uses nearest segment)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have same length >= 2")
    # find best bracketing segment
    best_i = None
    best_span = float("inf")
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
        if lo <= x <= hi:
            span = hi - lo
            if span < best_span:
                best_span = span
                best_i = i
    if best_i is None:
        # fallback: nearest segment midpoint
        best_i = 0
        best_d = float("inf")
        for i in range(len(xs) - 1):
            xm = 0.5 * (xs[i] + xs[i + 1])
            d = abs(xm - x)
            if d < best_d:
                best_d = d
                best_i = i
    x1, x2 = xs[best_i], xs[best_i + 1]
    y1, y2 = ys[best_i], ys[best_i + 1]
    if x2 == x1:
        return float(y1)
    t = (x - x1) / (x2 - x1)
    return float(y1 + t * (y2 - y1))


def omega_h(xs: list[float], vs: list[float], x_h: float) -> float:
    """Ω_H ≈ dv0/dx at x_h using nearest segment slope (Eq. 30)."""
    if len(xs) != len(vs) or len(xs) < 2:
        raise ValueError("xs and vs must have same length >= 2")
    # find nearest interval
    best_i = 0
    best_d = float("inf")
    for i in range(len(xs) - 1):
        xm = 0.5 * (xs[i] + xs[i + 1])
        d = abs(xm - x_h)
        if d < best_d:
            best_d = d
            best_i = i
    dx = xs[best_i + 1] - xs[best_i]
    if dx == 0.0:
        raise ValueError("duplicate xs points")
    return (vs[best_i + 1] - vs[best_i]) / dx

def analyze_profile_1d(inp: dict) -> dict:
    """
    Unified profile analysis.
    Input supports:
      - horizon: {xs,vs,c0}
      - null speeds: {x,xs,vs,c0}
    Returns a dict with whatever can be computed.
    """
    xs = [float(z) for z in inp["xs"]]
    vs = [float(z) for z in inp["vs"]]
    c0 = float(inp["c0"])

    out = {"input": {"xs": xs, "vs": vs, "c0": c0}}

    # horizon-related
    try:
        xh = find_horizon_x(xs, vs, c0=c0)
        Om = omega_h(xs, vs, xh)
        out.update({
            "horizon_x": xh,
            "Omega_H": Om,
            "T_H_coeff": hawking_temperature(Om),
        })
    except Exception:
        pass

    # null-speeds at point
    if "x" in inp:
        x = float(inp["x"])
        v_at = interp1_linear(xs, vs, x)
        s_minus, s_plus = null_speeds_1d(v_at, c0)
        out.update({
            "x": x,
            "v0_at_x": v_at,
            "dxdt_minus": s_minus,
            "dxdt_plus": s_plus,
        })

    return out


def hawking_temperature(
    omega_h_val: float,
    hbar: float | None = None,
    k_b: float | None = None,
) -> float:
    """
    T_H = (ħ / (2π k_B)) * Ω_H  (Eq. 31).
    If hbar/k_b not provided, returns (1/(2π)) * Ω_H (i.e., coefficient-only).
    """
    if hbar is None or k_b is None:
        return omega_h_val / (2.0 * math.pi)
    return (hbar * omega_h_val) / (2.0 * math.pi * k_b)

def eikonal_phase(action_line: float, omega: float, T: float) -> float:
    """Eikonal phase/action: S = ∮ p·dx − ω T  (Eq. 25)."""
    return float(action_line) - float(omega) * float(T)


def _ham_F(k: tuple[float, float, float], v0: tuple[float, float, float], c0: float, omega: float) -> float:
    # F(k) = 1/2(ω − v0·k)^2 − 1/2 c0^2 |k|^2   (inside ∇_k in Eq. 26)
    vk = _dot(v0, k)
    kn2 = _dot(k, k)
    return 0.5 * (omega - vk) * (omega - vk) - 0.5 * (c0 * c0) * kn2


def canonical_momentum(
    k: tuple[float, float, float],
    v0: tuple[float, float, float],
    c0: float,
    omega: float,
) -> tuple[float, float, float]:
    """p = ∇_k[ 1/2(ω − v0·k)^2 − 1/2 c0^2 |k|^2 ]  (Eq. 26)."""
    vk = _dot(v0, k)
    fac = -(omega - vk)  # derivative of 1/2(omega - v0·k)^2 w.r.t k gives (omega - v0·k)(-v0)
    c2 = c0 * c0
    return (
        fac * v0[0] - c2 * k[0],
        fac * v0[1] - c2 * k[1],
        fac * v0[2] - c2 * k[2],
    )


def _selftest() -> None:
    # simple numeric check (not a physics test suite)
    k = (3.0, 4.0, 0.0)  # |k|=5
    v0 = (10.0, 0.0, 0.0)
    c0 = 2.0
    w = omega_forward(k, v0, c0)
    vg = group_velocity(k, v0, c0)
        # acoustic interval checks (Eq. 24), 1D
    ds2 = acoustic_ds2_1d(dt=1.0, dx=0.0, v0=0.0, c0=2.0, rho0=1.0)
    # expected: (1/2)*(-4*1 + 0) = -2
    assert abs(ds2 + 2.0) < 1e-12

    ds2b = acoustic_ds2_1d(dt=1.0, dx=2.0, v0=2.0, c0=2.0, rho0=1.0)
    # dx - v0 dt = 0 => same as above
    assert abs(ds2b + 2.0) < 1e-12

    # expected: v0·k=30, c0|k|=10 => ω=40
    assert abs(w - 40.0) < 1e-12
    # expected: vg=(10,0,0) + 2*(3/5,4/5,0) = (11.2,1.6,0)
    assert abs(vg[0] - 11.2) < 1e-12
    assert abs(vg[1] - 1.6) < 1e-12
    assert abs(vg[2] - 0.0) < 1e-12
        # horizon & Omega_H checks (Eq. 30): v(x)=1+x, c0=2 => x_H=1, dv/dx=1
    xs = [0.0, 1.0, 2.0]
    vs = [1.0, 2.0, 3.0]
    xh = find_horizon_x(xs, vs, c0=2.0)
    assert abs(xh - 1.0) < 1e-12
    Om = omega_h(xs, vs, xh)
    assert abs(Om - 1.0) < 1e-12


    # loop phase checks (Eq. 27, 29)
    chi = 0.5
    circulation = 7.0
    phi_loop = loop_phase_from_circulation(omega=4.0, chi=chi, c0=2.0, circulation=circulation)
    assert abs(phi_loop - 14.0) < 1e-12

    phi_sag = sagnac_phase(omega=4.0, chi=chi, c0=2.0, omega_dot_area=3.0)
    assert abs(phi_sag - 12.0) < 1e-12

        # Hawking analogue temperature proportionality (Eq. 31)
    th = hawking_temperature(omega_h_val=2.0)
    assert abs(th - (2.0 / (2.0 * math.pi))) < 1e-12

    # Eq.25: S = I - omega*T
    S = eikonal_phase(action_line=10.0, omega=2.0, T=3.0)
    assert abs(S - 4.0) < 1e-12

    # Eq.26: compare analytic ∇_k F with finite differences
    k = (0.3, -0.4, 0.5)
    v0 = (1.2, -0.7, 0.1)
    c0 = 1.8
    omega = 2.3
    p = canonical_momentum(k, v0, c0, omega)
    eps = 1e-6
    for i in range(3):
        k1 = [k[0], k[1], k[2]]
        k2 = [k[0], k[1], k[2]]
        k1[i] += eps
        k2[i] -= eps
        num = (_ham_F((k1[0], k1[1], k1[2]), v0, c0, omega) - _ham_F((k2[0], k2[1], k2[2]), v0, c0, omega)) / (2.0 * eps)
        assert abs(p[i] - num) < 1e-6

    # Eq.22-23: g * ginv = I (numeric check)
    v0 = 0.3
    c0 = 2.0
    rho0 = 1.5
    g = acoustic_metric_g_1d(v0, c0, rho0)
    gi = acoustic_metric_ginv_1d(v0, c0, rho0)
    I = _mat2_mul(g, gi)
    assert abs(I[0][0] - 1.0) < 1e-12
    assert abs(I[0][1] - 0.0) < 1e-12
    assert abs(I[1][0] - 0.0) < 1e-12
    assert abs(I[1][1] - 1.0) < 1e-12

    # Null characteristics from metric (ds^2=0): dx/dt = v0 ± c0
    v0 = 0.7
    c0 = 2.0
    s1, s2 = null_speeds_1d(v0, c0)
    assert abs(s1 - (v0 - c0)) < 1e-12
    assert abs(s2 - (v0 + c0)) < 1e-12
    # check ds^2==0 for dx = (v0+c0) dt
    assert is_null_1d(dt=1.0, dx=(v0 + c0) * 1.0, v0=v0, c0=c0, rho0=1.0, tol=1e-9)

    # Directional null speeds: v0·n̂ ± c0
    v0v = (1.0, 2.0, 0.0)
    n = (3.0, 4.0, 0.0)  # n̂=(0.6,0.8,0)
    c0 = 2.0
    s1, s2 = null_speeds_along(n, v0v, c0)
    # v0·n̂ = 1*0.6 + 2*0.8 = 2.2
    assert abs(s1 - 0.2) < 1e-12
    assert abs(s2 - 4.2) < 1e-12


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description="REL engine (v0.0.1) demo skeleton.")
    ap.add_argument("--demo", action="store_true", help="Run demo mode.")
    ap.add_argument("--outdir", default="", help="Optional outdir for engine-native artifacts.")
    ap.add_argument("--selftest", action="store_true", help="Run internal self-test and exit.")
    ap.add_argument("--calc-horizon", default="", help="Path to JSON with {xs,vs,c0} to compute x_H, Omega_H, T_H_coeff.")
    ap.add_argument("--out", default="", help="Optional output JSON path for calc-horizon result.")
    ap.add_argument("--calc-null-speeds", default="", help="Path to JSON with {x,xs,vs,c0} to compute v0(x) and v0±c0.")
    ap.add_argument("--calc-profile", default="", help="Path to JSON with {xs,vs,c0} and optional {x}; runs unified analysis.")

    args = ap.parse_args()

    if args.selftest:
        _selftest()
        
        print("[rel] selftest: OK")
        return 0
    
    if args.calc_horizon:
        inp = json.loads(Path(args.calc_horizon).read_text(encoding="utf-8-sig"))
        xs = [float(x) for x in inp["xs"]]
        vs = [float(v) for v in inp["vs"]]
        c0 = float(inp["c0"])
        xh = find_horizon_x(xs, vs, c0=c0)
        Om = omega_h(xs, vs, xh)
        Th = hawking_temperature(Om)

        inp = json.loads(Path(args.calc_horizon).read_text(encoding="utf-8-sig"))
        res = analyze_profile_1d(inp)

        outp = {
            "engine": "rel_engine_v001",
            "timestamp_utc": now_iso(),
            **res,
            "units_note": "T_H_coeff = Omega_H/(2*pi); SI requires hbar,k_B",
        }

        if args.out:
            Path(args.out).write_text(json.dumps(outp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            print(json.dumps(outp, ensure_ascii=False))
        return 0

    if args.calc_null_speeds:
        inp = json.loads(Path(args.calc_null_speeds).read_text(encoding="utf-8-sig"))
        x = float(inp["x"])
        xs = [float(z) for z in inp["xs"]]
        vs = [float(z) for z in inp["vs"]]
        c0 = float(inp["c0"])

        v_at = interp1_linear(xs, vs, x)
        s_minus, s_plus = null_speeds_1d(v_at, c0)

        inp = json.loads(Path(args.calc_null_speeds).read_text(encoding="utf-8-sig"))
        res = analyze_profile_1d(inp)

        outp = {
            "engine": "rel_engine_v001",
            "timestamp_utc": now_iso(),
            **res,
        }
        
        if args.out:
            Path(args.out).write_text(json.dumps(outp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            print(json.dumps(outp, ensure_ascii=False))
        return 0

    if args.calc_profile:
        inp = json.loads(Path(args.calc_profile).read_text(encoding="utf-8-sig"))
        res = analyze_profile_1d(inp)

        outp = {
            "engine": "rel_engine_v001",
            "timestamp_utc": now_iso(),
            **res,
            "units_note": "T_H_coeff = Omega_H/(2*pi); SI requires hbar,k_B",
        }

        if args.out:
            Path(args.out).write_text(json.dumps(outp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            print(json.dumps(outp, ensure_ascii=False))
        return 0


    # Demo: produce a tiny engine-native artifact (NOT the results contract).
    payload = {
        "engine": "rel_engine_v001",
        "timestamp_utc": now_iso(),
        "mode": "demo" if args.demo else "noop",
        "notes": "This is a placeholder engine skeleton.",
    }
        # lightweight demo calc: simple linear profile v(x)=1+x, c0=2
    xs = [0.0, 1.0, 2.0]
    vs = [1.0, 2.0, 3.0]
    xh = find_horizon_x(xs, vs, c0=2.0)
    Om = omega_h(xs, vs, xh)
    Th_coeff = hawking_temperature(Om)  # coefficient-only (no SI constants)

    payload["calc"] = {
        "profile": {"xs": xs, "vs": vs, "c0": 2.0},
        "horizon_x": xh,
        "Omega_H": Om,
        "T_H_coeff": Th_coeff,
        "units_note": "T_H_coeff = Omega_H/(2*pi); SI requires hbar,k_B",
    }

    # second demo profile: v(x)=2x, c0=2  -> x_H=1, Omega_H=2
    xs2 = [0.0, 1.0, 2.0]
    vs2 = [0.0, 2.0, 4.0]
    xh2 = find_horizon_x(xs2, vs2, c0=2.0)
    Om2 = omega_h(xs2, vs2, xh2)
    Th2 = hawking_temperature(Om2)

    payload["calc2"] = {
        "profile": {"xs": xs2, "vs": vs2, "c0": 2.0},
        "horizon_x": xh2,
        "Omega_H": Om2,
        "T_H_coeff": Th2,
    }

    if args.outdir:
        outdir = Path(args.outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "rel_engine_demo.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(payload, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
