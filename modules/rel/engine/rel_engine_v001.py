from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import math

def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


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
    return (v0[0] + c0 * k[0]/kn, v0[1] + c0 * k[1]/kn, v0[2] + c0 * k[2]/kn)


def _selftest() -> None:
    # simple numeric check (not a physics test suite)
    k = (3.0, 4.0, 0.0)      # |k|=5
    v0 = (10.0, 0.0, 0.0)
    c0 = 2.0
    w = omega_forward(k, v0, c0)
    vg = group_velocity(k, v0, c0)

    # expected: v0·k=30, c0|k|=10 => ω=40
    assert abs(w - 40.0) < 1e-12
    # expected: vg=(10,0,0) + 2*(3/5,4/5,0) = (11.2,1.6,0)
    assert abs(vg[0] - 11.2) < 1e-12
    assert abs(vg[1] - 1.6) < 1e-12
    assert abs(vg[2] - 0.0) < 1e-12


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
