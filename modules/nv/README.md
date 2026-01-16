# NV Module

This module contains tools related to **NV-center based laboratory analysis**
within the UCM-T framework.

---

## Structure

nv/
engine/
nv_engine_v023.py
pilot_nv.py
README.md


- `engine/`  
  Contains the **domain physics code** for NV-center fitting and analysis.
  This code is kept independent from any pipeline or orchestration logic.

- `pilot_nv.py`  
  A **lightweight adapter** that runs the NV engine and exports results
  according to the UCM-T *results contract*:



results/
results_global.json
results_items.csv


---

## Current status

- NV engine runs in demo mode and produces real fit results
- UTF-8 handling is enforced for Windows compatibility
- Fully integrated into the calibration launcher

This module is considered **operational** at the pipeline level.
Further work may extend structured metric extraction.
