# tools/core_calib_collect.py
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ModuleScan:
    module: str
    module_dir: Path
    wrapper_status_path: Optional[Path]
    results_global_path: Optional[Path]
    contract_ok: bool
    status: str
    error: str


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_first(module_dir: Path, rel_candidates: List[Path]) -> Optional[Path]:
    for rel in rel_candidates:
        p = module_dir / rel
        if p.exists():
            return p
    return None


def _scan_module_dir(module_dir: Path) -> ModuleScan:
    module = module_dir.name

    # Prefer canonical locations
    wrapper_status = _find_first(module_dir, [
        Path("wrapper_status.json"),
        Path("results") / "wrapper_status.json",
    ])

    results_global = _find_first(module_dir, [
        Path("results") / "results_global.json",
        Path("results_global.json"),  # legacy
    ])

    contract_ok = False
    status = "missing"
    error = ""

    if wrapper_status and wrapper_status.exists():
        try:
            ws = _read_json(wrapper_status)
            status = str(ws.get("status", "unknown"))
            contract_ok = (status == "ok")
            if status == "error":
                error = str(ws.get("error", ""))
        except Exception as e:
            status = "bad_wrapper_status_json"
            error = f"{type(e).__name__}: {e}"

    return ModuleScan(
        module=module,
        module_dir=module_dir,
        wrapper_status_path=wrapper_status,
        results_global_path=results_global,
        contract_ok=contract_ok,
        status=status,
        error=error,
    )


def _flatten_primitives(d: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only top-level scalar metrics suitable for CSV."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None or isinstance(v, (int, float, bool)):
            out[k] = v
        elif isinstance(v, str):
            # keep only short single-line strings (e.g. bench tags), drop logs
            if ("\n" not in v) and ("\r" not in v) and (len(v) <= 120):
                out[k] = v
    return out


def _csv_safe(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

def _load_aliases(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """
    Return: { module_name: { alias_key -> preferred_key } }
    Uses tools/core_calib_aliases.json structure:
      modules.<module>.preferred_outputs.<preferred> = [aliases...]
    """
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"aliases file not found: {p}")
    data = _read_json(p)
    modules = data.get("modules", {})
    out: Dict[str, Dict[str, str]] = {}
    for mod, block in modules.items():
        pref_out = (block or {}).get("preferred_outputs", {}) or {}
        rev: Dict[str, str] = {}
        for preferred, aliases in pref_out.items():
            for a in (aliases or []):
                rev[str(a)] = str(preferred)
        out[str(mod)] = rev
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect results_global.json across module runs (no optimization).")
    ap.add_argument("--run-dir", required=True, help="Run directory containing module subfolders (e.g. CALIB_SMOKE_V23).")
    ap.add_argument(
        "--out-root",
        default=r"C:\UCM\RUNS",
        help=r"Where to create CORE_CALIB_COLLECT_<timestamp> (default: C:\UCM\RUNS).",
    )
    ap.add_argument(
    "--aliases",
    default=None,
    help="Path to core_calib_aliases.json (optional). Used to normalize rg__ keys.",
)

    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    out_root = Path(args.out_root).expanduser()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"CORE_CALIB_COLLECT_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    alias_rev = _load_aliases(args.aliases)
    out_root = Path(args.out_root).expanduser()
    
    scans: List[ModuleScan] = []
    rows: List[Dict[str, Any]] = []
    all_keys: List[str] = []

    for child in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        scan = _scan_module_dir(child)
        scans.append(scan)

        row: Dict[str, Any] = {
            "module": scan.module,
            "module_dir": str(scan.module_dir),
            "wrapper_status": str(scan.wrapper_status_path) if scan.wrapper_status_path else "",
            "results_global": str(scan.results_global_path) if scan.results_global_path else "",
            "contract_ok": scan.contract_ok,
            "status": scan.status,
            "error": scan.error,
        }

        if scan.results_global_path and scan.results_global_path.exists():
            try:
                rg = _read_json(scan.results_global_path)
                flat = _flatten_primitives(rg)
                rev = alias_rev.get(scan.module, {})
                for k, v in flat.items():
                    k2 = rev.get(k, k)  # normalize
                    key = f"rg__{k2}"
                    if key not in row:  # prefer first occurrence (avoid overwrite)
                        row[key] = v
            except Exception as e:
                row["status"] = "bad_results_global_json"
                row["error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    # Build header: fixed first, then union of all rg__ keys
    fixed = ["module", "module_dir", "wrapper_status", "results_global", "contract_ok", "status", "error"]
    rg_keys = sorted({k for r in rows for k in r.keys() if k.startswith("rg__")})
    header = fixed + rg_keys

    merged_csv = out_dir / "core_calib_merged.csv"
    with merged_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_safe(r.get(k)) for k in header})

    report_md = out_dir / "core_calib_report.md"
    ok_modules = [s.module for s in scans if s.contract_ok]
    bad_modules = [s for s in scans if not s.contract_ok]

    with report_md.open("w", encoding="utf-8") as f:
        f.write("# CORE_CALIB_COLLECT report\n\n")
        f.write(f"- run_dir: {run_dir}\n")
        f.write(f"- out_dir: {out_dir}\n")
        f.write(f"- modules found: {len(scans)}\n")
        f.write(f"- contract_ok: {len(ok_modules)} -> {', '.join(ok_modules) if ok_modules else '(none)'}\n\n")
        if bad_modules:
            f.write("## Non-contract_ok / missing\n\n")
            for s in bad_modules:
                f.write(f"- {s.module}: status={s.status}, error={s.error}\n")

    manifest = out_dir / "core_calib_manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": "core_calib_collect_v1",
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "created_local": ts,
                "modules": [
                    {
                        "module": s.module,
                        "module_dir": str(s.module_dir),
                        "wrapper_status": str(s.wrapper_status_path) if s.wrapper_status_path else None,
                        "results_global": str(s.results_global_path) if s.results_global_path else None,
                        "aliases_path": str(Path(args.aliases).expanduser().resolve()) if args.aliases else None,

                        "contract_ok": s.contract_ok,
                        "status": s.status,
                        "error": s.error,
                    }
                    for s in scans
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[done] out_dir: {out_dir}")
    print(f"[done] merged:  {merged_csv}")
    print(f"[done] report:  {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
