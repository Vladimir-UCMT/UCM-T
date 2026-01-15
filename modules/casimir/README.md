# Casimir Module

This module implements **Casimir energy and force calculations**
within the Unified Compressible-Medium (UCM) framework.

---

## Structure

casimir/
engine/
casimir_ucm.py
pilot_casimir.py
README.md

pgsql
Копировать код

- `engine/`  
  Contains pure physical functions for Casimir energy and force evaluation.
  The code is intentionally kept minimal and side-effect free.

- `pilot_casimir.py`  
  A pipeline adapter that evaluates Casimir quantities on fixed
  reference parameters and exports results using the UCM-T
  results contract.

---

## Current status

- Real Casimir force and energy are computed
- Deterministic smoke metrics are exported
- Fully integrated into the calibration pipeline

This module serves as a **micro-/quantum-scale calibration anchor**
within the multi-domain UCM-T system.
