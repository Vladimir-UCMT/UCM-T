from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description="REL engine (v0.0.1) demo skeleton.")
    ap.add_argument("--demo", action="store_true", help="Run demo mode.")
    ap.add_argument("--outdir", default="", help="Optional outdir for engine-native artifacts.")
    args = ap.parse_args()

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
