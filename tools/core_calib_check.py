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

    # Candidate shared parameters (presence only)
    candidates = ["rg__c0", "rg__rho_inf", "rg__kappa", "rg__kappa_s"]
    present = [c for c in candidates if c in merged.read_text(encoding="utf-8")]
    missing = [c for c in candidates if c not in present]

    print("[check] candidate shared params columns (presence only):")
    print(f"  present: {', '.join(present) if present else '(none)'}")
    print(f"  missing: {', '.join(missing) if missing else '(none)'}")
        # --- hard requirements for Phase 0: shared medium params ---
    # Require rg__c0 to be non-empty in at least 2 modules (expected: nv + rel)
    req_key = "rg__c0"
    mods_with_key = [m for m, keys in per_mod.items() if req_key in keys]  # per_mod already stores non-empty rg__ keys
    mods_with_key = sorted(set(mods_with_key))

    if len(mods_with_key) < 5:
        print(f"[fail] required shared param {req_key} must be non-empty in >=5 modules, got {len(mods_with_key)}: {mods_with_key}")
        return 2
    else:
        print(f"[ok] required shared param {req_key} present in modules: {mods_with_key}")
    req_key = "rg__rho_inf"
    mods_with_key = [m for m, keys in per_mod.items() if req_key in keys]
    mods_with_key = sorted(set(mods_with_key))

    if len(mods_with_key) < 5:
        print(f"[fail] required shared param {req_key} must be non-empty in >=5 modules, got {len(mods_with_key)}: {mods_with_key}")
        return 2
    else:
        print(f"[ok] required shared param {req_key} present in modules: {mods_with_key}")


    req_key = "rg__kappa"
    mods_with_key = [m for m, keys in per_mod.items() if req_key in keys]
    mods_with_key = sorted(set(mods_with_key))

    if len(mods_with_key) < 5:
        print(f"[fail] required shared param {req_key} must be non-empty in >=5 modules, got {len(mods_with_key)}: {mods_with_key}")
        return 2
    else:
        print(f"[ok] required shared param {req_key} present in modules: {mods_with_key}")
        
    req_key = "rg__kappa_s"
    mods_with_key = [m for m, keys in per_mod.items() if req_key in keys]
    mods_with_key = sorted(set(mods_with_key))

    if len(mods_with_key) < 5:
        print(f"[fail] required shared param {req_key} must be non-empty in >=5 modules, got {len(mods_with_key)}: {mods_with_key}")
        return 2
    else:
        print(f"[ok] required shared param {req_key} present in modules: {mods_with_key}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
