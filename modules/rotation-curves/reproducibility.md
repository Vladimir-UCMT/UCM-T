### Expected behavior
The script runs the engine over the BENCH30 OUTER set.  
All outputs are written to the `results/` directory.
Note
This benchmark is executed from the canonical Zenodo bundle.
The engine code in this GitHub repository is provided for reference and inspection
and is not intended to be run standalone from this repository.

## Validation
After the run completes, verify that the following files exist in `results/`:
`results_global.json`, `results_items.csv`.

These files are the primary benchmark summary artifacts and are designed to be comparable across
engine versions under the same results contract.

## Results format and comparability
Benchmark outputs follow the project-level results contract defined in
`tools/results_contract.md`. This contract specifies the canonical output structure that enables
consistent comparison across modules and engine versions.

## Status and scope
RC V12 is a fixed, reproducible baseline. It is not an evolving development branch. Any improvements
or extensions must be released as new Zenodo versions and linked explicitly. Earlier exploratory
runs are considered superseded by RC V12.

## Citation
If you use this benchmark or reference bundle in a paper, please cite:
Yakovlev, V. (2026). *UCM Engine: Rotation Curve Benchmark (BENCH30 OUTER V12)*. Zenodo.
https://doi.org/10.5281/zenodo.18213329
