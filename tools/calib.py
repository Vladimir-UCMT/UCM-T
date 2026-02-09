# tools/calib.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set


def repo_root() -> Path:
    # tools/calib.py -> parents[1] = repo root
    return Path(__file__).resolve().parents[1]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _split_list(s: str) -> List[str]:
    return [x.strip().lower() for x in (s or "").split(",") if x.strip()]


def _mods_all() -> List[str]:
    return ["nv", "casimir", "rel", "rc", "rd"]


def _calc_skip(only: Optional[str], skip: Optional[str]) -> str:
    allm = set(_mods_all())
    if only:
        only_set = set(_split_list(only))
        unknown = sorted(only_set - allm)
        if unknown:
            raise SystemExit(f"Unknown modules in --only: {unknown}. Allowed: {_mods_all()}")
        skip_set = allm - only_set
        return ",".join(sorted(skip_set))
    if skip:
        skip_set = set(_split_list(skip))
        unknown = sorted(skip_set - allm)
        if unknown:
            raise SystemExit(f"Unknown modules in --skip: {unknown}. Allowed: {_mods_all()}")
        return ",".join(sorted(skip_set))
    return ""


def _runs_root_default() -> Path:
    # 1) explicit env
    if os.environ.get("UCM_RUNS_ROOT", "").strip():
        return Path(os.environ["UCM_RUNS_ROOT"]).expanduser().resolve()

    # 2) common Windows default
    p = Path(r"C:\UCM\RUNS")
    if os.name == "nt" and p.exists():
        return p

    # 3) fallback inside repo
    return repo_root() / "RUNS"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: List[str], *, cwd: Path, env: dict, title: str) -> subprocess.CompletedProcess:
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
def _run_live(cmd: List[str], *, cwd: Path, env: dict, title: str) -> subprocess.CompletedProcess:
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    # Live output to console (no capture) to avoid "hang" feeling.
    return subprocess.run(cmd, cwd=str(cwd), env=env)


def _latest_collect_dir(run_dir: Path) -> Optional[Path]:
    # core_calib_collect creates CORE_CALIB_COLLECT_<timestamp> in out-root
    cands = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("CORE_CALIB_COLLECT_")]
    if not cands:
        return None
    # name sort works because timestamp is YYYYMMDD_HHMMSS
    return sorted(cands, key=lambda p: p.name)[-1]


def cmd_healthcheck(args: argparse.Namespace) -> int:
    rr = repo_root()
    runs_root = _ensure_dir(Path(args.runs_root).expanduser().resolve() if args.runs_root else _runs_root_default())

    if args.outdir:
        run_dir = Path(args.outdir).expanduser().resolve()
    else:
        run_dir = runs_root / f"CALIB_HEALTH_{_ts()}"

    _ensure_dir(run_dir)

    # Phase0 env (canonical names)
    env = os.environ.copy()
    env["UCM_C0"] = str(args.phase0_c0)
    env["UCM_RHO_INF"] = str(args.phase0_rho_inf)
    env["UCM_KAPPA"] = str(args.phase0_kappa)
    env["UCM_KAPPA_S"] = str(args.phase0_kappa_s)

    # ultrafast: convenience preset (skip RC + RD no-run) unless user overrides via --only/--skip
    fast_mode = bool(args.fast or args.ultrafast)

    if args.ultrafast and not args.only and not args.skip:
        skip = _calc_skip(None, "rc")
    else:
        skip = _calc_skip(args.only, args.skip)

    # 1) run
    run_cmd = [sys.executable, "-X", "utf8", str(rr / "tools" / "run_calib_all.py"), "--outdir", str(run_dir)]
    if skip:
        run_cmd += ["--skip", skip]
    if fast_mode:
        # fast = do not run RD engine (adapter uses --no-run)
        run_cmd += ["--rd-no-run"]

    p1 = _run_live(run_cmd, cwd=rr, env=env, title="RUN (run_calib_all)")

    # Note: run_calib_all historically returns 0 even if module reported error.
    # We'll enforce correctness by collect+check below.

    # 2) collect into this run_dir (so artifacts stay together)
    collect_cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(rr / "tools" / "core_calib_collect.py"),
        "--run-dir",
        str(run_dir),
        "--out-root",
        str(run_dir),
        "--aliases",
        str(rr / "tools" / "core_calib_aliases.json"),
    ]
    p2 = _run(collect_cmd, cwd=rr, env=env, title="COLLECT (core_calib_collect)")
    print(p2.stdout)
    if p2.returncode != 0:
        print("[fail] collect failed")
        return 2

    def _parse_done_path(stdout: str, key: str) -> Optional[Path]:
        # examples:
        # [done] out_dir: C:\...\CORE_CALIB_COLLECT_...
        # [done] merged:  C:\...\core_calib_merged.csv
        for line in stdout.splitlines():
            line = line.strip()
            prefix = f"[done] {key}:"
            if line.startswith(prefix):
                return Path(line[len(prefix):].strip())
        return None

    collect_dir = _parse_done_path(p2.stdout, "out_dir")
    merged = _parse_done_path(p2.stdout, "merged")

    if not collect_dir or not collect_dir.exists():
        print("[fail] could not parse collect out_dir from output")
        return 2
    if not merged or not merged.exists():
        print("[fail] could not parse merged path from output")
        return 2

    # 3) check
    check_cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(rr / "tools" / "core_calib_check.py"),
        "--merged",
        str(merged),
    ]
    p3 = _run(check_cmd, cwd=rr, env=env, title="CHECK (core_calib_check)")
    print(p3.stdout)

    if p3.returncode != 0:
        print("[fail] check failed")
        return 2

    print("\n[ok] healthcheck passed")
    print(f"[ok] run_dir:  {run_dir}")
    print(f"[ok] merged:   {merged}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.calib", description="UCM-T calibration pipeline CLI.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    hc = sub.add_parser("healthcheck", help="Run -> collect -> check (one command).")
    hc.add_argument("--outdir", default=None, help="Run directory (default: auto in runs root).")
    hc.add_argument(
        "--runs-root",
        default=None,
        help="Root directory for auto runs (default: env UCM_RUNS_ROOT, else C:\\UCM\\RUNS, else ./RUNS).",
    )
    hc.add_argument("--skip", default="", help="Comma list to skip: nv,casimir,rel,rc,rd")
    hc.add_argument("--only", default=None, help="Comma list: run only these modules (overrides --skip).")
    hc.add_argument("--fast", action="store_true", help="Fast mode (currently: --rd-no-run).")
    hc.add_argument(
        "--ultrafast",
        action="store_true",
        help="Ultrafast preset: implies --fast and skips RC (no network/download).",
    )

    # Phase0
    hc.add_argument("--phase0-c0", type=float, default=2.0)
    hc.add_argument("--phase0-rho-inf", type=float, default=0.0)
    hc.add_argument("--phase0-kappa", type=float, default=0.0)
    hc.add_argument("--phase0-kappa-s", type=float, default=0.0)

    hc.set_defaults(func=cmd_healthcheck)

    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
