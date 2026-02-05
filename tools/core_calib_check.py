# tools/core_calib_check.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "ok")


def main() -> int:
    ap = argparse.ArgumentParser(description="Consistency check for CORE_CALIB_COLLECT merged CSV.")
    ap.add_argument("--merged", required=True, help="Path to core_calib_merged.csv")
    args = ap.parse_args()

    merged = Path(args.merged).expanduser().resolve()
    if not merged.exists():
        raise SystemExit(f"merged not found: {merged}")

    rows = _read_csv(merged)
    if not rows:
        raise SystemExit("merged csv is empty")

    # Basic contract_ok check
    bad = [r for r in rows if not _truthy(r.get("contract_ok", "")) or r.get("status", "") != "ok"]
    if bad:
        print("[error] non-ok modules:")
        for r in bad:
            print(f"  - {r.get('module')}: contract_ok={r.get('contract_ok')} status={r.get('status')} error={r.get('error')}")
        return 2

    print(f"[ok] modules={len(rows)} all contract_ok")

    # Inventory rg__ keys per module
    all_rg: Set[str] = set()
    per_mod: Dict[str, Set[str]] = {}
    for r in rows:
        mod = r.get("module", "?")
        keys = {k for k, v in r.items() if k.startswith("rg__") and (v is not None) and (str(v).strip() != "")}
        per_mod[mod] = keys
        all_rg |= keys

    print(f"[info] total rg__ keys present (non-empty): {len(all_rg)}")
    for mod in sorted(per_mod.keys()):
        keys = sorted(per_mod[mod])
        print(f"[info] {mod}: {len(keys)} keys")
        # print a short preview
        preview = ", ".join(keys[:12])
        if len(keys) > 12:
            preview += ", ..."
        print(f"       {preview}")

    # Candidate shared parameters (by CSV headers, not substring search)
    candidates = ["rg__c0", "rg__rho_inf", "rg__kappa", "rg__kappa_s"]
    columns = set(rows[0].keys())

    present = [c for c in candidates if c in columns]
    missing = [c for c in candidates if c not in columns]

    print("[check] candidate shared params columns (by header):")
    print(f"  present: {', '.join(present) if present else '(none)'}")
    print(f"  missing: {', '.join(missing) if missing else '(none)'}")

    def _norm_num(s: str):
        t = str(s).strip()
        if t == "":
            return None
        try:
            # normalize numeric formatting (2.1110 == 2.111)
            return round(float(t), 12)
        except Exception:
            return t  # fall back to raw string

    # --- hard requirements for Phase 0: shared medium params ---
    required = candidates

    for req_key in required:
        if req_key not in columns:
            print(f"[fail] required column missing: {req_key}")
            return 2

        # ensure non-empty for all modules and consistent across modules
        per_module_val = {}
        missing_mods = []
        for r in rows:
            mod = r.get("module", "?")
            raw = r.get(req_key, "")
            norm = _norm_num(raw)
            if norm is None:
                missing_mods.append(mod)
            else:
                per_module_val[mod] = (raw, norm)

        if missing_mods:
            missing_mods = sorted(set(missing_mods))
            print(f"[fail] required shared param {req_key} is empty in modules: {missing_mods}")
            return 2

        norms = {v[1] for v in per_module_val.values()}
        if len(norms) != 1:
            print(f"[fail] required shared param {req_key} differs across modules:")
            for mod in sorted(per_module_val.keys()):
                raw, norm = per_module_val[mod]
                print(f"  - {mod}: {raw} (norm={norm})")
            return 2

        the_val = next(iter(norms))
        mods = sorted(per_module_val.keys())
        print(f"[ok] required shared param {req_key} consistent across modules ({len(mods)}): {the_val}")



    return 0


if __name__ == "__main__":
    raise SystemExit(main())
