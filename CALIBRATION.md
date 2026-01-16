# Calibration pipeline (UCM-T)

This repo contains multiple domain modules integrated into a single calibration pipeline.
Each module provides a pilot adapter that writes results in the UCM results contract format:

<OUTDIR>/<module>/results/
results_global.json
results_items.csv


A unified launcher runs all modules and produces a summary.

---

## Quick start (local)

From the repository root:

### 1) Sanity check: Python syntax
```bash
python -m compileall -q .
2) Run the unified calibration launcher
python -X utf8 tools/run_calib_all.py --outdir _TMP/CALIB_CHECK


This runs (in order): NV, Casimir, Rotation Curves (RC), Ringdown (RD).

3) Inspect summary
type _TMP/CALIB_CHECK\calib_summary.csv


Expected: all modules report status=ok and items=True.

Notes

Ringdown (RD) status is taken from results/wrapper_status.json (preferred),
and results/results_global.json is treated as a secondary source.

Pilot adapters are designed to never crash the pipeline: on failure they still
publish contract files with status="error".