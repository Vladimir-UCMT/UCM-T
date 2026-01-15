# Modules

This directory contains **domain-specific modules** associated with UCM-T.

Each module represents an **independent physical domain**
(e.g. galactic dynamics, ringdown physics, laboratory experiments),
while adhering to common repository principles:

- reproducibility
- operational testability
- explicit separation between *physical engines* and *pipeline adapters*

---

## Available modules

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
