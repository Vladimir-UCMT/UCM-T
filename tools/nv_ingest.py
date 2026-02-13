# tools/nv_ingest.py
# NV Data Intake v1: DC211 lock-in CW ODMR raw TXT -> standardized CSV + meta

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(s: str) -> Optional[float]:
    s = str(s).strip()
    if not s:
        return None
    # tolerate decimal comma in metadata (e.g., 0,5)
    s2 = s.replace(",", ".")
    try:
        return float(s2)
    except Exception:
        return None


def _parse_record_time(s: str) -> Optional[str]:
    """Parse 'dd.mm.yyyy hh:mm:ss' into ISO8601 (local, unknown tz)."""
    s = s.strip()
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.isoformat()
        except Exception:
            pass
    return None


@dataclass
class ParsedTxt:
    meta: Dict[str, Any]
    rows: List[Tuple[float, float, float]]  # (freq_mhz, ch1_v, ch2_v)


def parse_dc211_lockin_txt(path: Path) -> ParsedTxt:
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()

    meta: Dict[str, Any] = {
        "source_file": str(path),
        "source_basename": path.name,
        "source_sha256": _sha256(path),
        "parser": "nv_ingest_dc211_lockin_txt_v1",
    }

    # --- header scan ---
    for line in txt[:200]:
        line = line.strip()
        if line.lower().startswith("record performed:"):
            t = line.split(":", 1)[1].strip()
            meta["record_performed"] = t
            iso = _parse_record_time(t)
            if iso:
                meta["record_performed_iso"] = iso
        elif line.lower().startswith("x channel name:"):
            meta["x_channel_name"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("comments:"):
            meta["comments"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("time constant:"):
            meta["time_constant"] = _safe_float(line.split(":", 1)[1].strip())
        elif line.lower().startswith("field step:"):
            meta["field_step"] = _safe_float(line.split(":", 1)[1].strip())
        elif line.lower().startswith("sweep count:"):
            meta["sweep_count"] = int(_safe_float(line.split(":", 1)[1].strip()) or 0)

    # --- locate data table ---
    start_idx: Optional[int] = None
    for i, line in enumerate(txt):
        l = line.lower()
        if "frequency" in l and "channel1" in l and "channel2" in l:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not locate data header line with Frequency/Channel1/Channel2")

    # data begins after the dashed separator following header
    data_idx: Optional[int] = None
    for j in range(start_idx + 1, min(start_idx + 10, len(txt))):
        t = txt[j].strip()
        if t and set(t) <= set("-\t "):
            data_idx = j + 1
            break
    if data_idx is None:
        # fallback: assume next line after header is separator, then data
        data_idx = start_idx + 2

    rows: List[Tuple[float, float, float]] = []
    for line in txt[data_idx:]:
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 3:
            continue
        if not _FLOAT_RE.match(parts[0].replace(",", ".")):
            # end of numeric block
            if rows:
                break
            continue
        f = _safe_float(parts[0])
        c1 = _safe_float(parts[1])
        c2 = _safe_float(parts[2])
        if f is None or c1 is None or c2 is None:
            continue
        rows.append((float(f), float(c1), float(c2)))

    if not rows:
        raise ValueError("No numeric data rows parsed")

    # summary stats
    freqs = [r[0] for r in rows]
    meta["n_rows"] = len(rows)
    meta["freq_mhz_min"] = min(freqs)
    meta["freq_mhz_max"] = max(freqs)
    if len(freqs) >= 3:
        diffs = [round(freqs[i + 1] - freqs[i], 12) for i in range(len(freqs) - 1)]
        diffs_nz = [d for d in diffs if d != 0]
        if diffs_nz:
            meta["freq_step_mhz_median"] = median(diffs_nz)

    return ParsedTxt(meta=meta, rows=rows)


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def infer_b_mt(meta: Dict[str, Any]) -> float:
    c = str(meta.get("comments", "") or "").lower()
    # heuristic: any 'mf' (magnetic field) in comments => ~4 mT
    if " mf" in f" {c}" or "with mf" in c or "маг" in c:
        return 4.0
    return 0.0


def write_csv(path: Path, rows: List[Tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq_mhz", "channel1_v", "channel2_v"])
        for f_mhz, c1, c2 in rows:
            w.writerow([f"{f_mhz:.10g}", f"{c1:.10g}", f"{c2:.10g}"])


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rel_if_under(p: Path, root: Path) -> str:
    """Return relative path if under root, otherwise absolute string."""
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="NV ingest: DC211 lock-in CW ODMR raw TXT -> CSV + meta (v1)")
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Input raw .txt (repeatable)")
    ap.add_argument("--out-root", required=True, help="Dataset root to write into (creates raw/csv/meta)")
    ap.add_argument("--dataset-id", default=None, help="Dataset identifier (stored in manifest/meta)")
    ap.add_argument("--am-hz", type=float, default=680.0, help="AM modulation frequency (Hz), stored in meta")
    ap.add_argument(
        "--primary-channel",
        choices=["channel1", "channel2"],
        default="channel1",
        help="Which channel is intended as primary signal",
    )
    ap.add_argument(
        "--b-mt",
        dest="b_mt",
        action="append",
        default=None,
        help="Magnetic field in mT per input (repeatable). If omitted, inferred from Comments.",
    )
    ap.add_argument("--copy-raw", action="store_true", help="Copy raw inputs into out-root/raw (default: on)")
    ap.add_argument("--no-copy-raw", dest="copy_raw", action="store_false")
    ap.set_defaults(copy_raw=True)
    ap.add_argument("--write-manifest", action="store_true", help="Write/overwrite dataset manifest.json (default: on)")
    ap.add_argument("--no-manifest", dest="write_manifest", action="store_false")
    ap.set_defaults(write_manifest=True)

    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    raw_dir = out_root / "raw"
    csv_dir = out_root / "csv"
    meta_dir = out_root / "meta"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    inputs = [Path(p).expanduser().resolve() for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    # map b_mt
    b_list: List[Optional[float]] = []
    if args.b_mt:
        vals = [(_safe_float(x) if x is not None else None) for x in args.b_mt]
        if len(vals) == 1 and len(inputs) > 1:
            b_list = vals * len(inputs)
        elif len(vals) == len(inputs):
            b_list = vals
        else:
            raise SystemExit("--b-mt must be provided once (applies to all) or exactly once per --in")
    else:
        b_list = [None] * len(inputs)

    records: List[Dict[str, Any]] = []

    for idx, p in enumerate(inputs):
        parsed = parse_dc211_lockin_txt(p)

        b_mt = float(b_list[idx]) if b_list[idx] is not None else float(infer_b_mt(parsed.meta))

        # Build stable stem
        stem = _slug(p.stem)

        # copy raw if requested
        raw_copy_path = raw_dir / p.name
        if args.copy_raw:
            # avoid creating duplicates when input is already under out-root/raw
            try:
                if raw_copy_path.resolve() != p.resolve():
                    if raw_copy_path.exists():
                        if _sha256(raw_copy_path) != _sha256(p):
                            raw_copy_path.write_bytes(p.read_bytes())
                    else:
                        raw_copy_path.write_bytes(p.read_bytes())
            except Exception:
                # best-effort copy
                if not raw_copy_path.exists():
                    raw_copy_path.write_bytes(p.read_bytes())

        csv_path = csv_dir / f"{stem}__cw_odmr.csv"
        write_csv(csv_path, parsed.rows)

        meta = dict(parsed.meta)
        meta.update(
            {
                "schema": "nv_dc211_lockin_odmr_v1",
                "dataset_id": args.dataset_id,
                "am_hz": float(args.am_hz),
                "primary_channel": args.primary_channel,
                "b_mt": b_mt,
                "raw_path": _rel_if_under(raw_copy_path if args.copy_raw else p, out_root),
                "csv_path": _rel_if_under(csv_path, out_root),
            }
        )

        # Quick sanity marker: frequency where primary channel is minimal
        try:
            rows = parsed.rows
            if args.primary_channel == "channel1":
                j = min(range(len(rows)), key=lambda k: rows[k][1])
            else:
                j = min(range(len(rows)), key=lambda k: rows[k][2])
            meta["rough_min_freq_mhz"] = float(rows[j][0])
        except Exception:
            pass

        meta_path = meta_dir / f"{stem}__meta.json"
        write_json(meta_path, meta)

        rec = {
            "item_id": f"{args.dataset_id or 'NV_DATASET'}::{stem}",
            "b_mt": b_mt,
            "csv": _rel_if_under(csv_path, out_root),
            "meta": _rel_if_under(meta_path, out_root),
            "raw": _rel_if_under(raw_copy_path if args.copy_raw else p, out_root),
        }
        records.append(rec)

        print(f"[ok] {p.name} -> {csv_path.name} (b_mt={b_mt})")

    if args.write_manifest:
        manifest = {
            "schema": "nv_ingest_manifest_v1",
            "dataset_id": args.dataset_id,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "n_records": len(records),
            "records": records,
        }
        write_json(out_root / "manifest.json", manifest)
        print(f"[ok] manifest: {out_root / 'manifest.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
