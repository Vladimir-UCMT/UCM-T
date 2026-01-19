# UCM-T Calibration Pipeline — Status Snapshot
**Date:** 2026-01-19  
**Baseline run:** CALIB_SMOKE_RC_PUBLISH6  
**Repo tag:** calib-v2

---

## 1. Scope of this snapshot

This document фиксирует состояние калибровочного конвейера UCM-T
на момент, когда все модули (NV, Casimir, RC, RD) успешно прошли
контрактную проверку (`--check-contract`).

Цель — обеспечить воспроизводимость, возврат к рабочему состоянию
и безопасное продолжение разработки без потери прогресса.

---

## 2. Final status (all green)

All modules report `contract_ok`:

| Module   | Status        |
|----------|---------------|
| NV       | contract_ok   |
| Casimir  | contract_ok   |
| RC       | contract_ok   |
| RD       | contract_ok   |

Verified run directory:
C:\UCM\RUNS\CALIB_SMOKE_RC_PUBLISH6


---

## 3. Calibration pipeline entry point

Unified launcher:
```powershell
python tools\run_calib_all.py --outdir <RUN_DIR>

4. Implemented validation modes
4.1 --dry-run
python tools\run_calib_all.py --outdir <RUN_DIR> --dry-run


Purpose:

Do NOT run engines

Check pilot script presence

Create output directory structure

Produce calib_summary.csv

Status: dry_ok

4.2 --check-engines
python tools\run_calib_all.py --outdir <RUN_DIR> --check-engines


Purpose:

Do NOT run computations

Validate existence of critical engine/runner files

Catch path/config errors early (esp. RC)

Statuses:

check_ok

missing_engine

4.3 --check-contract (key result of this stage)
python tools\run_calib_all.py --outdir <RUN_DIR> --check-contract


Validates:

results/results_global.json existence and readability

results/results_items.csv existence

numeric metric column (metric_value or normalized score/value)

schema consistency

Statuses:

contract_ok

missing_results

bad_items_csv

bad_global_json

This is the final reproducibility gate.

5. Results contract (de facto standard)
results_items.csv

Required columns:

item_id, status, score, metric_value, summary

results_global.json

Minimal required fields:

{
  "module": "...",
  "timestamp_utc": "...",
  "status": "ok|error",
  "engine_returncode": 0,
  "n_items": N
}

wrapper_status.json
{
  "schema": "ucm_wrapper_status_v1",
  "status": "ok|error",
  "error": "",
  "published_from": "results_global.json"
}

6. RC adapter: resolved issues

Key fix: modules/rotation-curves/pilot_rc.py

Resolved problems:

pilot_results.csv was produced but not published into results/

wrapper_status.json could be overwritten to error after success

undefined variables (pilot_csv / pilot_results_csv)

inconsistent success/error paths

Current guarantees:

results_global.json and results_items.csv are always published on success

wrapper_status.json = ok only after successful publication

error-contract published on any failure

no silent exceptions

RC is now a first-class, contract-compliant module.

7. What must NOT be changed casually

tools/run_calib_all.py (dispatcher + validation logic)

results/ directory structure

filenames of contract artifacts

meaning of contract statuses

Any changes require a new calibration version/tag.

8. Repository state

Relevant files stabilized:

tools/run_calib_all.py

modules/rotation-curves/pilot_rc.py

CALIBRATION_PIPELINE.md

Tag:

calib-v2

9. Recommended next steps (not started yet)

Freeze public contract in CALIBRATION_API.md

Improve RC results_items.csv richness (per-galaxy rows)

Cross-domain calibration using NV / Casimir as anchors

No new work started beyond this snapshot.


---
