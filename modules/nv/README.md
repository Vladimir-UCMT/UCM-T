# NV Module

This module contains tools related to **NV-center based laboratory analysis**
within the UCM-T framework.

---

## Structure

```
modules/nv/
  bench/
    NV_DC211_LOCKIN_CW_ODMR_AM680_V1/
  engine/
    nv_engine_v023.py
  pilot_nv.py
  README.md
```

- `bench/`  
  Small, commit-friendly **ingestion references** (raw→CSV).  
  Converter CLI: `python -m tools.nv_ingest`.

- `engine/`  
  Domain physics code for NV-center fitting and analysis.  
  Kept independent from pipeline / orchestration logic.

- `pilot_nv.py`  
  A lightweight **wrapper/adapter** that runs the NV engine and publishes
  artifacts according to the UCM-T **results contract**.

---

## Outputs (Results Contract)

The wrapper always creates `results/` under the run output directory:

- `results/results_global.json` — global run summary (JSON)
- `results/results_items.csv` — per-item metrics table (CSV)
- `results/wrapper_status.json` — wrapper execution status (JSON)

### `results_global.json` (key fields)

- `schema`: `ucm_results_contract_v1`
- `module`: `nv`
- `timestamp_utc`
- `status`: `ok` or `error`
- `engine_returncode`
- `n_items`
- `wrapper_version`
- Repro metadata (from `tools/contract_meta.py`):
  - `ucmt_repo`, `ucmt_commit`, `python_version`, `platform`

### `results_items.csv` (required columns)

The calibration validator expects the items CSV to follow a standard schema.
For NV the wrapper publishes a minimal single-row table with:

- `item_id`
- `status` (`ok` / `fail`)
- `score` (numeric)
- `metric_value` (numeric)
- `summary`

This keeps NV compatible with the unified calibration pipeline even when
running in demo mode.

### `wrapper_status.json`

Wrapper status payload:

- `schema`: `ucm_wrapper_status_v1`
- `status`
- `returncode` (wrapper publish returncode; `0` on successful publish)
- `has_items_csv`
- `error` (if any)
- `published_from`

---

## Current status

- NV wrapper runs the engine in demo mode and produces contract-compliant outputs.
- UTF-8 handling is enforced for Windows compatibility.
- Fully integrated into the unified calibration launcher (smoke test passes).

This module is considered **operational** at the pipeline level.
Further work may extend structured metric extraction from real ODMR fit outputs.

## Bench sanity (real data)
NV wrapper reads bench dataset via $env:UCM_NV_BENCH_DATASET_ID (default: NV_DC211_LOCKIN_CW_ODMR_AM680_V1).

