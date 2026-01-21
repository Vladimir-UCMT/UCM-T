# tools/contract_meta.py
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    # tools/contract_meta.py -> parents[1] = repo root
    return Path(__file__).resolve().parents[1]


def _git(args: list[str]) -> str:
    try:
        rr = _repo_root()
        proc = subprocess.run(
            ["git", *args],
            cwd=str(rr),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            return ""
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def contract_meta(*, wrapper_version: str | None = None) -> dict:
    """
    Stable metadata for results_global.json (publication/repro level).
    Safe on machines without git: returns empty strings for git fields.
    """
    meta = {
        "ucmt_repo": _git(["config", "--get", "remote.origin.url"]),
        "ucmt_commit": _git(["rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if wrapper_version:
        meta["wrapper_version"] = wrapper_version
    return meta
