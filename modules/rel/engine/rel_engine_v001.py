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

    # loop phase checks (Eq. 27, 29)
    chi = 0.5
    circulation = 7.0
    phi_loop = loop_phase_from_circulation(omega=4.0, chi=chi, c0=2.0, circulation=circulation)
    assert abs(phi_loop - 14.0) < 1e-12

    phi_sag = sagnac_phase(omega=4.0, chi=chi, c0=2.0, omega_dot_area=3.0)
    assert abs(phi_sag - 12.0) < 1e-12

def acoustic_ds2_1d(dt: float, dx: float, v0: float, c0: float, rho0: float = 1.0) -> float:
    """
    Acoustic interval in 1D (Eq. 24):
      ds^2 = (rho0/c0) * ( -c0^2 dt^2 + (dx - v0 dt)^2 )
    """
    if c0 == 0.0:
        raise ValueError("c0 must be non-zero")
    return (rho0 / c0) * (-(c0 * c0) * (dt * dt) + (dx - v0 * dt) * (dx - v0 * dt))



def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description="REL engine (v0.0.1) demo skeleton.")
    ap.add_argument("--demo", action="store_true", help="Run demo mode.")
    ap.add_argument("--outdir", default="", help="Optional outdir for engine-native artifacts.")
    ap.add_argument("--selftest", action="store_true", help="Run internal self-test and exit.")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        
        print("[rel] selftest: OK")
        return 0

    # Demo: produce a tiny engine-native artifact (NOT the results contract).
    payload = {
        "engine": "rel_engine_v001",
        "timestamp_utc": now_iso(),
        "mode": "demo" if args.demo else "noop",
        "notes": "This is a placeholder engine skeleton.",
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
