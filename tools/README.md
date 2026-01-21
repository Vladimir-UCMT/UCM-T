# Tools: Calibration Pipeline

This folder contains helpers and scripts for running the unified calibration
wrappers and validating their outputs against the UCM-T results contract.

Contract specification (with examples): `tools/results_contract.md`.

---

## Quick smoke test

Run the end-to-end smoke test for all integrated modules:

```powershell
powershell -ExecutionPolicy Bypass -File tools/scripts/run_calib_smoke_v23.ps1

Outputs are written under C:\UCM\RUNS\... and each module publishes artifacts into:

results/results_global.json

results/results_items.csv

results/wrapper_status.json

###Results contract utilities

Contract definition: tools/results_contract.md

Comparator (validate/compare): tools/compare_results_contract.py

Contract metadata helper: tools/contract_meta.py

###Examples (copy-paste templates)
####results/results_global.json
{
  "schema": "ucm_results_contract_v1",
  "module": "rc",
  "timestamp_utc": "2026-01-21T12:34:56+00:00",
  "status": "ok",
  "engine_returncode": 0,
  "n_items": 30,
  "ucmt_repo": "https://github.com/<org>/<repo>.git",
  "ucmt_commit": "abc123def456",
  "python_version": "3.12.7",
  "platform": "Windows-10-10.0.19045-SP0",
  "wrapper_version": "calib-v2.3"
}

####results/results_items.csv
item_id,status,score,metric_value,summary
ITEM_001,ok,1.0,0.123,Example item row
__error__,fail,0.0,1.0,RuntimeError: example failure

####results/wrapper_status.json
{
  "schema": "ucm_wrapper_status_v1",
  "status": "ok",
  "returncode": 0,
  "has_items_csv": true,
  "error": "",
  "published_from": "pilot_rc.py"
}

##Helper: contract metadata

tools/contract_meta.py injects reproducibility metadata into results_global.json
(repo URL, commit hash, python version, platform).

Wrappers typically do:

from tools.contract_meta import contract_meta
global_payload.update(contract_meta(wrapper_version="calib-v2.3"))