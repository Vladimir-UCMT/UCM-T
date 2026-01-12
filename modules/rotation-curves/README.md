# Rotation Curves Module

This module collects UCM-T related work on **galactic rotation curves**:
models, calibration pipelines, and reproducible benchmarks.

## Reproducibility

The canonical reproducible benchmark for this module is **RC V12 (BENCH30 OUTER)**.
See `reproducibility.md` for the exact reproduction procedure, expected outputs, and validation rules.
Results contract
Run outputs follow the project-level results contract: `tools/results_contract.md`.

## Scope

- Operational modeling of rotation curves within the UCM-T framework
- Calibration against public catalogs (e.g., SPARC) where applicable
- Reproducible benchmark runs and comparison protocols

## Data policy

Large datasets and benchmark outputs must not be committed to Git.
Use external archives (e.g., Zenodo) and provide persistent links.

## Status

Bootstrap stage.
This folder currently provides a stable entry point for future additions:
scripts, documentation, and links to external reproducibility bundles.

## Planned contents (non-binding)

- `links.md` — persistent links to datasets / Zenodo bundles / releases
- `runs/` — small demo runs only (full results archived externally)
- `tools/` — helpers specific to rotation-curve calibration
