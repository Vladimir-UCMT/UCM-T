# Modules

This directory contains domain-specific modules associated with UCM-T.
Each module is designed to evolve independently while adhering to the
repository principles: reproducibility, operational testability, and
external hosting for large datasets/outputs.

## Available modules

- **Rotation Curves** — [modules/rotation-curves/](rotation-curves/)  
  Galactic rotation curves: calibration pipelines, benchmarks, and external links.
  
  - **Ringdown** — [modules/ringdown/](ringdown/)  
  Ringdown modeling, inference pipelines, and reproducibility links.


## Conventions (lightweight)

A module may contain any structure it needs. Recommended minimal files:
- `README.md` — scope and status
- `links.md` — persistent external resources (Zenodo / releases)
- small demo artifacts only (full outputs archived externally)
