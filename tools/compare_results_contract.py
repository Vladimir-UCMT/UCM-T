#!/usr/bin/env python3
"""Compare and validate UCM-T results contract artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


ID_CANDIDATES = ["event_id", "galaxy", "item_id", "item", "name", "id"]
NULL_TOKENS = {"", "nan", "null", "none", "na"}
ZERO_WIDTH_PREFIX = "\ufeff\u200b\u200c\u200d\u2060"


@dataclass
class RunArtifacts:
    run_dir: Path
    global_path: Optional[Path] = None
    items_path: Optional[Path] = None
    used_fallback: bool = False


@dataclass
class RunData:
    artifacts: RunArtifacts
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    global_data: Optional[Dict[str, Any]] = None
    global_signature: Optional[str] = None
    global_keys: List[str] = field(default_factory=list)
    items_rows: List[Dict[str, str]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    id_column: Optional[str] = None
    numeric_columns: List[str] = field(default_factory=list)
    duplicate_ids: List[str] = field(default_factory=list)
    all_null_metric_rows: List[str] = field(default_factory=list)
    non_numeric_values: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ComparisonData:
    warnings: List[str] = field(default_factory=list)
    column_added: List[str] = field(default_factory=list)
    column_removed: List[str] = field(default_factory=list)
    ids_only_left: List[str] = field(default_factory=list)
    ids_only_right: List[str] = field(default_factory=list)
    numeric_diffs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare UCM-T results contract artifacts.")
    parser.add_argument("--left", required=True, help="Left run directory")
    parser.add_argument("--right", help="Right run directory")
    parser.add_argument("--check-only", action="store_true", help="Validate only the left run")
    parser.add_argument("--out", help="Optional path to write markdown or JSON report")
    parser.add_argument("--top-n", type=int, default=5, help="Top-N item diffs per column")
    return parser.parse_args()


def is_null(value: Optional[str]) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in NULL_TOKENS


def parse_number(value: Optional[str]) -> Tuple[bool, Optional[float]]:
    if is_null(value):
        return True, None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return False, None
    if math.isnan(parsed):
        return True, None
    return True, parsed


def find_artifacts(run_dir: Path) -> RunArtifacts:
    preferred_global = run_dir / "results" / "results_global.json"
    preferred_items = run_dir / "results" / "results_items.csv"
    if preferred_global.exists() or preferred_items.exists():
        return RunArtifacts(run_dir=run_dir, global_path=preferred_global, items_path=preferred_items)

    fallback_global = run_dir / "results_global.json"
    fallback_items = run_dir / "results_items.csv"
    return RunArtifacts(
        run_dir=run_dir,
        global_path=fallback_global,
        items_path=fallback_items,
        used_fallback=True,
    )


def canonical_json_signature(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_global_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str], Optional[str]]:
    if not path.exists():
        return None, None, [], f"Missing results_global.json at {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return None, None, [], f"Failed to parse JSON ({path}): {exc}"
    if not isinstance(data, dict):
        return None, None, [], f"Global JSON must be a dict ({path})"
    keys = sorted(data.keys())
    signature = canonical_json_signature(data)
    return data, signature, keys, None


def read_items_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str], Optional[str]]:
    if not path.exists():
        return [], [], f"Missing results_items.csv at {path}"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                raw_fieldnames = next(reader)
            except StopIteration:
                return [], [], f"CSV missing header row ({path})"
            fieldnames = [normalize_header_name(name) for name in raw_fieldnames]
            dict_reader = csv.DictReader(handle, fieldnames=fieldnames)
            rows = [dict(row) for row in dict_reader]
    except csv.Error as exc:
        return [], [], f"CSV parse error ({path}): {exc}"
    return rows, fieldnames, None


def normalize_header_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    return str(name).lstrip(ZERO_WIDTH_PREFIX)


def choose_id_column(columns: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in ID_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def analyze_items(rows: List[Dict[str, str]], columns: List[str], run_data: RunData) -> None:
    if not rows:
        run_data.warnings.append("CSV has zero data rows")
    id_column = choose_id_column(columns)
    run_data.id_column = id_column
    if id_column is None:
        run_data.errors.append("No id-like column found")
        return

    numeric_columns: List[str] = []
    non_numeric_values: Dict[str, List[str]] = {}
    for column in columns:
        if column == id_column:
            continue
        any_numeric = False
        for row in rows:
            ok, parsed = parse_number(row.get(column))
            if parsed is not None:
                any_numeric = True
                break
        if any_numeric:
            numeric_columns.append(column)

    if not numeric_columns:
        run_data.errors.append("No numeric metric columns detected")
        return

    duplicate_ids: List[str] = []
    id_seen: Dict[str, int] = {}
    all_null_metric_rows: List[str] = []

    for row in rows:
        row_id = str(row.get(id_column, "")).strip()
        if row_id:
            id_seen[row_id] = id_seen.get(row_id, 0) + 1
        metrics_null = True
        for column in numeric_columns:
            ok, parsed = parse_number(row.get(column))
            if not ok:
                non_numeric_values.setdefault(column, []).append(row_id or "<missing-id>")
            if parsed is not None:
                metrics_null = False
        if metrics_null:
            all_null_metric_rows.append(row_id or "<missing-id>")

    duplicate_ids = sorted([item_id for item_id, count in id_seen.items() if count > 1])
    run_data.numeric_columns = sorted(numeric_columns)
    run_data.duplicate_ids = duplicate_ids
    run_data.all_null_metric_rows = all_null_metric_rows
    run_data.non_numeric_values = {key: sorted(ids) for key, ids in non_numeric_values.items()}

    if duplicate_ids:
        run_data.warnings.append(f"Duplicate item IDs detected ({len(duplicate_ids)})")
    if all_null_metric_rows:
        run_data.warnings.append(f"Rows with all-null metrics ({len(all_null_metric_rows)})")
    if non_numeric_values:
        run_data.warnings.append("Non-numeric values detected in numeric columns")


def validate_run(run_dir: Path) -> RunData:
    artifacts = find_artifacts(run_dir)
    run_data = RunData(artifacts=artifacts)

    if artifacts.used_fallback:
        run_data.warnings.append("Using legacy root-level results files; preferred location is results/")

    global_data, signature, keys, error = read_global_json(artifacts.global_path)
    if error:
        run_data.errors.append(error)
    else:
        run_data.global_data = global_data
        run_data.global_signature = signature
        run_data.global_keys = keys

    rows, columns, error = read_items_csv(artifacts.items_path)
    if error:
        run_data.errors.append(error)
    else:
        run_data.items_rows = rows
        run_data.columns = columns
        analyze_items(rows, columns, run_data)

    return run_data


def build_id_map(rows: List[Dict[str, str]], id_column: str) -> Dict[str, Dict[str, str]]:
    mapped: Dict[str, Dict[str, str]] = {}
    for row in rows:
        row_id = str(row.get(id_column, "")).strip()
        if row_id:
            mapped[row_id] = row
    return mapped


def compute_numeric_diffs(
    left: RunData,
    right: RunData,
    top_n: int,
) -> Dict[str, Dict[str, Any]]:
    if not (left.id_column and right.id_column):
        return {}
    left_ids = build_id_map(left.items_rows, left.id_column)
    right_ids = build_id_map(right.items_rows, right.id_column)
    common_ids = sorted(set(left_ids) & set(right_ids))
    column_intersection = sorted(set(left.numeric_columns) & set(right.numeric_columns))
    diffs: Dict[str, Dict[str, Any]] = {}

    for column in column_intersection:
        values: List[Tuple[str, float, float, float]] = []
        for item_id in common_ids:
            left_ok, left_val = parse_number(left_ids[item_id].get(column))
            right_ok, right_val = parse_number(right_ids[item_id].get(column))
            if left_ok and right_ok and left_val is not None and right_val is not None:
                delta = right_val - left_val
                values.append((item_id, left_val, right_val, delta))
        if not values:
            continue
        deltas = [entry[3] for entry in values]
        values_sorted = sorted(values, key=lambda item: (-abs(item[3]), item[0]))
        diffs[column] = {
            "mean_delta": mean(deltas),
            "median_delta": median(deltas),
            "max_abs_delta": max(abs(item) for item in deltas),
            "top_diffs": [
                {
                    "item_id": item_id,
                    "left": left_val,
                    "right": right_val,
                    "delta": delta,
                }
                for item_id, left_val, right_val, delta in values_sorted[:top_n]
            ],
        }

    return diffs


def compare_runs(left: RunData, right: RunData, top_n: int) -> ComparisonData:
    comparison = ComparisonData()

    left_columns = set(left.columns)
    right_columns = set(right.columns)
    comparison.column_added = sorted(right_columns - left_columns)
    comparison.column_removed = sorted(left_columns - right_columns)

    if left.id_column and right.id_column:
        left_ids = build_id_map(left.items_rows, left.id_column)
        right_ids = build_id_map(right.items_rows, right.id_column)
        comparison.ids_only_left = sorted(set(left_ids) - set(right_ids))
        comparison.ids_only_right = sorted(set(right_ids) - set(left_ids))

    comparison.numeric_diffs = compute_numeric_diffs(left, right, top_n)

    return comparison


def format_list(items: List[str]) -> str:
    if not items:
        return "(none)"
    return ", ".join(items)


def render_run_section(label: str, run: RunData) -> List[str]:
    artifacts = run.artifacts
    lines = [f"## {label}"]
    lines.append(f"- Run directory: {artifacts.run_dir}")
    lines.append(f"- results_global.json: {artifacts.global_path}")
    lines.append(f"- results_items.csv: {artifacts.items_path}")
    if run.global_signature:
        lines.append(f"- Global JSON signature: `{run.global_signature}`")
    if run.global_keys:
        lines.append(f"- Global JSON keys: {format_list(run.global_keys)}")
    if run.columns:
        lines.append(f"- CSV columns: {format_list(sorted(run.columns))}")
    if run.id_column:
        lines.append(f"- ID column: `{run.id_column}`")
    if run.numeric_columns:
        lines.append(f"- Numeric metric columns: {format_list(run.numeric_columns)}")
    if run.duplicate_ids:
        lines.append(f"- Duplicate IDs: {format_list(run.duplicate_ids)}")
    if run.all_null_metric_rows:
        lines.append(f"- Rows with all-null metrics: {format_list(run.all_null_metric_rows)}")
    if run.non_numeric_values:
        for column in sorted(run.non_numeric_values):
            ids = run.non_numeric_values[column]
            lines.append(f"- Non-numeric values in {column}: {format_list(ids)}")
    if run.warnings:
        lines.append(f"- Warnings: {format_list(run.warnings)}")
    if run.errors:
        lines.append(f"- Errors: {format_list(run.errors)}")
    return lines


def render_comparison_section(comparison: ComparisonData) -> List[str]:
    lines = ["## Comparison"]
    lines.append(f"- Columns added on right: {format_list(comparison.column_added)}")
    lines.append(f"- Columns removed on right: {format_list(comparison.column_removed)}")
    lines.append(f"- Items only in left: {format_list(comparison.ids_only_left)}")
    lines.append(f"- Items only in right: {format_list(comparison.ids_only_right)}")
    if comparison.numeric_diffs:
        lines.append("- Numeric column diffs:")
        for column in sorted(comparison.numeric_diffs):
            diff = comparison.numeric_diffs[column]
            lines.append(
                f"  - {column}: mean={diff['mean_delta']:.6g}, "
                f"median={diff['median_delta']:.6g}, max_abs={diff['max_abs_delta']:.6g}"
            )
            for entry in diff["top_diffs"]:
                lines.append(
                    "    - "
                    f"{entry['item_id']}: left={entry['left']:.6g}, "
                    f"right={entry['right']:.6g}, delta={entry['delta']:.6g}"
                )
    else:
        lines.append("- Numeric column diffs: (none)")
    if comparison.warnings:
        lines.append(f"- Warnings: {format_list(comparison.warnings)}")
    return lines


def render_markdown_report(
    left: RunData,
    right: Optional[RunData],
    comparison: Optional[ComparisonData],
) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines = ["# Results Contract Comparator Report", f"Generated: {timestamp}", ""]
    lines.extend(render_run_section("Left run", left))
    if right:
        lines.append("")
        lines.extend(render_run_section("Right run", right))
    if comparison:
        lines.append("")
        lines.extend(render_comparison_section(comparison))
    lines.append("")
    return "\n".join(lines)


def render_json_report(
    left: RunData,
    right: Optional[RunData],
    comparison: Optional[ComparisonData],
) -> Dict[str, Any]:
    def run_payload(run: RunData) -> Dict[str, Any]:
        return {
            "run_dir": str(run.artifacts.run_dir),
            "results_global": str(run.artifacts.global_path),
            "results_items": str(run.artifacts.items_path),
            "global_signature": run.global_signature,
            "global_keys": run.global_keys,
            "columns": sorted(run.columns),
            "id_column": run.id_column,
            "numeric_columns": run.numeric_columns,
            "duplicate_ids": run.duplicate_ids,
            "all_null_metric_rows": run.all_null_metric_rows,
            "non_numeric_values": run.non_numeric_values,
            "warnings": run.warnings,
            "errors": run.errors,
        }

    payload: Dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "left": run_payload(left),
    }
    if right:
        payload["right"] = run_payload(right)
    if comparison:
        payload["comparison"] = {
            "column_added": comparison.column_added,
            "column_removed": comparison.column_removed,
            "ids_only_left": comparison.ids_only_left,
            "ids_only_right": comparison.ids_only_right,
            "numeric_diffs": comparison.numeric_diffs,
            "warnings": comparison.warnings,
        }
    return payload


def main() -> int:
    args = parse_args()
    left_dir = Path(args.left).expanduser().resolve()
    right_dir = Path(args.right).expanduser().resolve() if args.right else None

    if args.check_only and right_dir:
        raise SystemExit("--check-only cannot be used with --right")
    if not args.check_only and not right_dir:
        raise SystemExit("Provide --right or use --check-only")

    left = validate_run(left_dir)
    right = validate_run(right_dir) if right_dir else None

    comparison = compare_runs(left, right, args.top_n) if right else None

    report_markdown = render_markdown_report(left, right, comparison)
    print(report_markdown)

    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".json":
            payload = render_json_report(left, right, comparison)
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            out_path.write_text(report_markdown, encoding="utf-8")

    has_errors = bool(left.errors) or (right and right.errors)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
