# UCM-T Calibration API (v1)

This document *freezes* the calibration pipeline interface:
folder layout, required artifacts, statuses, and versioning rules.

If we change anything here, we bump the API version (v2, v3, ...).

---

## 1) Run directory layout (required)

A calibration run root is provided by `--outdir`.

Expected structure:

<OUTDIR>/
├─ nv/
│  └─ results/
│     ├─ results_global.json
│     ├─ results_items.csv
│     └─ wrapper_status.json
├─ casimir/
│  └─ results/...
├─ rc/
│  └─ results/...
├─ rd/
│  └─ results/...
└─ calib_summary.{json,csv}

Notes:
- Modules may add extra artifacts (logs, figures, posteriors).
- The pipeline only relies on the required files above.

---

## 2) Required module artifacts

### 2.1 results_global.json (required)

Minimal required keys:

- module (string)
- timestamp_utc (ISO string)
- status ("ok" | "error")
- engine_returncode (int)
- n_items (int)

Optional keys:
- ucmt_repo, commit_hash, dataset_id, engine_name, engine_version, params, metrics, notes

### 2.2 results_items.csv (required)

Minimal required columns:

- item_id
- status            ("ok" | "fail" | "skip")
- score             (module-defined scalar)
- metric_value      (numeric scalar used for cross-domain calibration)
- summary           (short text)

Rules:
- CSV must have at least 1 data row.
- At least one row must contain numeric `metric_value`.

### 2.3 wrapper_status.json (required for pipeline-facing status)

Schema: `ucm_wrapper_status_v1`

Minimal keys:

- schema: "ucm_wrapper_status_v1"
- status: "ok" | "error"
- error: string (empty if ok)
- published_from: string (e.g., "results_global.json")

Rule:
- wrapper_status.json MUST be written only after results are published.
- If publishing fails → wrapper_status.json MUST be "error".

---

## 3) Pipeline summary artifacts

### 3.1 calib_summary.csv (required)

Columns:

- module
- status
- returncode
- has_items_csv
- outdir
- error

### 3.2 calib_summary.json (required)

Minimal keys:

- timestamp_utc
- outdir
- mode
- results (array of per-module objects)

---

## 4) Status codes (pipeline)

Per-module `status` values used by `run_calib_all.py`:

- run: "ok" / "error" / "missing_results" / "bad_global_json" / "bad_items_csv" / "bad_wrapper_status"
- --dry-run: "dry_ok" / "missing_pilot"
- --check-engines: "check_ok" / "missing_pilot" / "missing_engine"
- --check-contract: "contract_ok" / "missing_results" / "bad_global_json" / "bad_items_csv" / "bad_wrapper_status" / "error"

---

## 5) Versioning rules

- Any breaking change to:
  - required file names or locations,
  - required keys/columns,
  - status semantics
  → bumps API version (v2, v3, ...), and the pipeline must be tagged.

Recommended tags:
- calib-v1, calib-v2, ...
