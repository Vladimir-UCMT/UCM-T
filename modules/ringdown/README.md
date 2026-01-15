# Ringdown Module

This module contains tools for **ringdown modeling and inference**
within the UCM-T framework.

The focus of the module is the analysis of compact-object ringdown signals,
including benchmark runs and real-event pipelines.

---

## Structure

ringdown/
engine/
core/
pilot_cvn_rd.py
...
bench/
RD_BENCH_3.txt
RD_BENCH_3_NEW.txt
data/
cvn/
...
README.md

yaml
Копировать код

- `engine/`  
  Contains the core ringdown physics and inference code.
  This is the primary scientific implementation and may evolve independently.

- `bench/`  
  Local benchmark notes and comparison artifacts.
  These files are **not part of the calibration pipeline** and are ignored by git.

- `data/`  
  Small reference datasets required for local testing.
  Full datasets and outputs are hosted externally.

---

## Current status

- Ringdown engine is **operational and validated**
- Benchmark runs and reference comparisons exist
- Partial results contract support is implemented

---

## Pipeline integration

At present, the ringdown module is **not yet connected**
to the unified calibration launcher.

Planned integration will introduce a lightweight adapter:

pilot_rd.py

arduino
Копировать код

which will:
- run selected ringdown scenarios
- export results using the standard UCM-T results contract
- allow inclusion in multi-domain calibration runs

No changes to the underlying physics engine are required.