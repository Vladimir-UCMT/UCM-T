# tools/shared_env.py
from __future__ import annotations

from typing import Mapping, Optional, Dict
import os


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, str(default))).strip())
    except Exception:
        return float(default)


def phase0_params(env: Optional[Mapping[str, str]] = None) -> Dict[str, float]:
    """
    Canonical Phase0 shared-medium parameters, read from environment variables.

    Returns dict with keys: c0, rho_inf, kappa, kappa_s
    """
    if env is None:
        env = os.environ

    return {
        "c0": _env_float(env, "UCM_C0", 2.0),
        "rho_inf": _env_float(env, "UCM_RHO_INF", 0.0),
        "kappa": _env_float(env, "UCM_KAPPA", 0.0),
        "kappa_s": _env_float(env, "UCM_KAPPA_S", 0.0),
    }
