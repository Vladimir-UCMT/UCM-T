# Modules

# UCM-T Calibration Modules

This directory contains calibration-ready modules integrated into the
UCM-T calibration pipeline.

Each module exposes a **pilot adapter** that:
- runs the underlying engine (or publishes existing results),
- writes outputs into a standardized `results/` directory,
- conforms to the **UCM Results Contract** (`results_global.json`,
  `results_items.csv`),
- never crashes the pipeline (errors are reported via contract files).

---

## Integrated modules (status: 2026-01-16)

### 1. Rotation Curves (RC)
Path:
modules/rotation-curves/

makefile
Копировать код

Adapter:
modules/rotation-curves/pilot_rc.py

yaml
Копировать код

Status:
- ✅ Integrated into calibration pipeline
- ✅ Produces valid results contract
- ✅ Robust to engine failures (always writes numeric metric)

Notes:
- Uses SPARC rotation curve data
- Calls `pilot_sparc_ucmfit_v4_*` runner
- Explicitly passes engine path to avoid environment-dependent failures

---

### 2. Ringdown (RD)
Path:
modules/ringdown/

makefile
Копировать код

Adapter:
modules/ringdown/pilot_rd.py

yaml
Копировать код

Status:
- ✅ Integrated into calibration pipeline
- ✅ Produces valid results contract
- ✅ Supports `--no-run` (publish-only mode)

Notes:
- Wraps CVN ringdown engine
- Copies engine outputs into standardized `results/` layout
- Warnings about missing events are expected and documented

---

### 3. NV-center ODMR (NV)
Path:
modules/nv/

yaml
Копировать код

Status:
- ✅ Integrated (pilot adapter present)
- ✅ Contract-compatible outputs

---

### 4. Casimir module
Path:
modules/casimir/

yaml
Копировать код

Status:
- ✅ Integrated (pilot adapter present)
- ✅ Contract-compatible outputs

---

## Results Contract

Each module writes:

<outdir>/results/
├─ results_global.json
└─ results_items.csv

markdown
Копировать код

Design principles:
- `results_items.csv` **always contains at least one numeric metric**
- On failure, `status="error"` is written instead of raising exceptions
- Contracts are validated with:
tools/compare_results_contract.py

yaml
Копировать код

---

## Current state summary

- All four physical domains (RC, RD, NV, Casimir) are integrated
- All modules pass contract validation
- Calibration pipeline is stable and reproducible
- Next step: introduce a unified launcher (`run_calib_all.py`)