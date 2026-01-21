# Results Contract (UCM-T)

This document defines a lightweight **results contract** used across UCM-T modules.
It does not impose a repository structure; it specifies only the minimal metadata
needed for reproducibility and cross-domain comparison.

---

## Minimal result set (recommended)

A reproducible run should produce (recommended location: `results/`):

1. `results/results_global.json` — run-level summary (single file)
2. `results/results_items.csv` — item-level table (rows = events/objects/tests)
3. `results/wrapper_status.json` — wrapper execution status (publish success/failure)
4. `run_manifest.txt` (or `.md`) — human-readable reproduction notes

`run_manifest` should include:
- exact commands used to run,
- dependency notes,
- where inputs came from (links/DOI),
- where outputs are stored (paths or external archive link),
- how to validate correctness (key numbers/figures).

---

## results_global.json

### Required fields (minimal)

- `schema`: `ucm_results_contract_v1`
- `module`: short module id (e.g., `nv`, `casimir`, `rc`, `rd`)
- `timestamp_utc`: ISO8601 timestamp in UTC
- `status`: `ok` or `error`
- `engine_returncode`: integer engine/process return code
- `n_items`: integer number of item rows written to `results_items.csv`

### Recommended fields (repro metadata)

These fields are typically injected by a helper (e.g., `tools/contract_meta.py`):

- `ucmt_repo`: git remote URL (if available)
- `ucmt_commit`: git commit hash (if available)
- `python_version`: runtime Python version
- `platform`: OS/platform string
- `wrapper_version`: wrapper/pipeline version tag (e.g., `calib-v2.3`)

### Optional fields

- `error`: short error string (present when `status="error"`)
- `stdout_tail`, `stderr_tail`: short tails for debugging (optional)

---

## results_items.csv

### Required columns (pipeline-compatible minimal set)

- `item_id` — unique row id (event/object/test id)
- `status` — `ok` or `fail`
- `score` — numeric (module-defined meaning)
- `metric_value` — numeric primary metric (module-defined meaning)
- `summary` — short human-readable summary

Modules may add extra columns. Each module must document the meaning of `score`,
`metric_value`, and any additional columns.

---

## wrapper_status.json

Wrapper status is written to `results/wrapper_status.json` and describes whether the
wrapper successfully **published** contract artifacts (even if the engine failed).

### Recommended fields

- `schema`: `ucm_wrapper_status_v1`
- `status`: wrapper status (`ok` / `error`)
- `returncode`: wrapper publish return code (0 means wrapper finished publishing)
- `has_items_csv`: boolean
- `error`: string (empty if none)
- `published_from`: script name (e.g., `pilot_rc.py`)

---

## Examples

### Example: results/results_global.json

```json
{
  "schema": "ucm_results_contract_v1",
  "module": "nv",
  "timestamp_utc": "2026-01-21T12:34:56+00:00",
  "status": "ok",
  "engine_returncode": 0,
  "n_items": 1,
  "ucmt_commit": "abc123def456",
  "python_version": "3.12.7",
  "platform": "Windows-10-10.0.19045-SP0",
  "wrapper_version": "calib-v2.3"
}

###Example: results/results_items.csv

item_id,status,score,metric_value,summary
DEMO,ok,1.0,0.0,NV demo run
__error__,fail,0.0,1.0,FileNotFoundError: NV engine not found

###Example: results/wrapper_status.json

{
  "schema": "ucm_wrapper_status_v1",
  "status": "ok",
  "returncode": 0,
  "has_items_csv": true,
  "error": "",
  "published_from": "pilot_nv.py"
}

##Notes

This contract is intentionally minimal and flexible.
Large artifacts (e.g., full posteriors) should be stored externally when appropriate.



