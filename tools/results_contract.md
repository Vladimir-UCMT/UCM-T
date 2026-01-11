# Results Contract (UCM-T)

This document defines a lightweight **results contract** used across UCM-T modules.
It does not impose a repository structure; it specifies only the minimal metadata
needed for reproducibility and cross-domain comparison.

---

## Minimal result set (recommended)

A reproducible run should produce:

1. `results_global.json` — run-level summary (single file)
2. `results_items.csv` — item-level table (rows = events/objects/tests)
3. `run_manifest.txt` (or `.md`) — human-readable reproduction notes

Modules may add additional outputs (figures, logs, posteriors), but these three
files provide a common denominator for comparison.

---

## 1) `results_global.json` (run-level summary)

**Purpose:** capture the run configuration and aggregate metrics.

Recommended fields:

- `ucmt_repo`: repository identifier (e.g., `Vladimir-UCMT/UCM-T`)
- `module`: module path (e.g., `modules/rotation-curves`)
- `engine_name`: name of engine/script used
- `engine_version`: version string if available
- `commit_hash`: git commit hash (if applicable)
- `timestamp_utc`: ISO timestamp
- `dataset_id`: dataset name/version (or DOI)
- `n_items`: number of items evaluated
- `metrics`: dictionary of aggregate metrics (e.g., mean chi2, pass rate)
- `params`: dictionary of key parameters (model/regime/hyperparameters)
- `notes`: free text (optional)

---

## 2) `results_items.csv` (item-level table)

**Purpose:** comparable per-object/per-event results.

Required columns (minimal):
- `item_id` — object/event identifier
- `status` — `ok` / `fail` / `skip`
- `score` — primary scalar score (module-defined; documented in module README)
- `summary` — short note or key parameter/value (optional but helpful)

Recommended additional columns:
- `score_alt` — secondary score if needed
- `n_data` — number of data points used
- `runtime_s` — runtime per item (if applicable)
- module-specific parameters (documented in module README)

---

## 3) `run_manifest.txt` (or `.md`)

**Purpose:** a short human-readable reproduction recipe.

Include:
- exact commands used to run,
- dependency notes,
- where inputs came from (links/DOI),
- where outputs are stored (paths or external archive link),
- how to validate correctness (key numbers/figures).

---

## Notes

- This contract is intentionally minimal and flexible.
- Each module must document the meaning of `score` and any module-specific columns.
- Large artifacts (e.g., full posteriors) should be stored externally when appropriate.
