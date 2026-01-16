# Modules

# UCM-T Calibration Modules
This directory contains **domain-specific modules** associated with UCM-T.

Each module represents an **independent physical domain**
(e.g. galactic dynamics, ringdown physics, laboratory experiments),
while adhering to common repository principles:

- reproducibility
- operational testability
- explicit separation between *physical engines* and *pipeline adapters*

---

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


Adapter:
modules/rotation-curves/pilot_rc.py


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


Adapter:
modules/ringdown/pilot_rd.py


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


Status:
- ✅ Integrated (pilot adapter present)
- ✅ Contract-compatible outputs

---

### 4. Casimir module
Path:
modules/casimir/


Status:
- ✅ Integrated (pilot adapter present)
- ✅ Contract-compatible outputs

---

## Results Contract

Each module writes:

<outdir>/results/
├─ results_global.json
└─ results_items.csv


Design principles:
- `results_items.csv` **always contains at least one numeric metric**
- On failure, `status="error"` is written instead of raising exceptions
- Contracts are validated with:
tools/compare_results_contract.py

---

## Current state summary

- All four physical domains (RC, RD, NV, Casimir) are integrated
- All modules pass contract validation
- Calibration pipeline is stable and reproducible
- Next step: introduce a unified launcher (`run_calib_all.py`)

- **Rotation Curves** — [`modules/rotation-curves/`](rotation-curves/)  
  Galactic rotation curves: calibration pipelines, benchmarks, and external datasets.

- **Ringdown** — [`modules/ringdown/`](ringdown/)  
  Ringdown modeling and inference pipelines for compact objects.

- **NV** — [`modules/nv/`](nv/)  
  NV-center based laboratory analysis and fitting tools.

- **Casimir** — [`modules/casimir/`](casimir/)  
  Casimir energy and force calculations within the UCM framework.

---

## Module structure (recommended)

A module may contain any internal structure it needs.
The following layout is recommended but not mandatory:

module/
engine/ # domain physics code (no pipeline logic)
pilot_*.py # lightweight wrapper for the calibration pipeline
README.md # scope, status, and usage notes
links.md # external persistent resources (Zenodo, releases)

Unified calibration launcher

All integrated calibration modules can be executed with a single command.

Run all modules:

python -X utf8 tools/run_calib_all.py --outdir <OUTDIR>


This sequentially runs:

NV (ODMR)

Casimir

Rotation Curves (RC)

Ringdown (RD)

Each module writes its own results contract to:

<OUTDIR>/<module>/results/


A global summary is written to:

<OUTDIR>/calib_summary.json
<OUTDIR>/calib_summary.csv


Publish-only mode for Ringdown (skip engine run):

python -X utf8 tools/run_calib_all.py --outdir <OUTDIR> --rd-no-run
