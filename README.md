# UCM-T
## Start here

- Manifesto (EN): `docs/manifesto_en.md`
- Manifesto (RU): `docs/manifesto_ru.md`
- Modules: `modules/`
- Results contract: `tools/results_contract.md`
- Results contract comparator: `tools/compare_results_contract.py`
- Contract metadata helper: `tools/contract_meta.py`
- Smoke test launcher: `tools/scripts/run_calib_smoke_v23.ps1`
- Latest calibration baseline: Release `calib-v2.3.1` (see GitHub Releases)

## Calibration pipeline (quick check)

To verify that the unified calibration wrappers are working end-to-end, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/scripts/run_calib_smoke_v23.ps1
Outputs are written under C:\UCM\RUNS\... and each module publishes artifacts into:

results/results_global.json

results/results_items.csv

results/wrapper_status.json

**Unified Compressible Medium Theory (UCM-T)**  
Research hub for models, calibration pipelines, and reproducible benchmarks.

## Navigation

- Documentation: [`docs/`](docs/)  
  Methodological manifesto (RU/EN) and lightweight project conventions.

- Modules: [`modules/`](modules/)  
  Domain-specific workspaces (e.g., rotation curves) with reproducibility links.

  - Tools: [`tools/`](tools/)  
  Reproducibility bundle template, results contract, and the results contract comparator.

This repository serves as a central coordination point for:
- theoretical models based on the Unified Compressible Medium framework,
- calibration pipelines across different physical domains,
- reproducible computational benchmarks and validation datasets.

The project is organized around the principle of **operational testability**:
any theoretical statement is meaningful only if it can be mapped to
a concrete calibration or falsification procedure.

> This repository does not enforce a fixed internal structure.
> Individual modules, engines, and datasets may evolve independently,
> provided they comply with reproducibility and documentation requirements.

## Phase 0: shared medium params (UCM_*)

UCM-T calibration wrappers export the following shared “medium parameters” into `results_global.json`
(for all smoke modules: nv, casimir, rc, rd, rel). Values are taken from environment variables.

### Environment variables

- `UCM_C0` — default `2.0`
- `UCM_RHO_INF` — default `0.0`
- `UCM_KAPPA` — default `0.0`
- `UCM_KAPPA_S` — default `0.0`

These parameters are written to `results_global.json` as:
`c0`, `rho_inf`, `kappa`, `kappa_s` (and become `rg__c0`, `rg__rho_inf`, `rg__kappa`, `rg__kappa_s` after collect/merge).

### PowerShell quick run (current session)

```powershell
$env:UCM_C0="2.0"
$env:UCM_RHO_INF="0.0"
$env:UCM_KAPPA="0.0"
$env:UCM_KAPPA_S="0.0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\scripts\run_calib_smoke_v23.ps1

Notes:

$env:NAME="value" sets an env var for the current PowerShell process/session.

-ExecutionPolicy Bypass runs this script without changing the system-wide execution policy.

### KK healthcheck (one command)

```powershell
cd C:\UCM\UCM-T
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\scripts\healthcheck_calib_smoke.ps1


License: MIT
