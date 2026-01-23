# REL module

Relativistic extension ("rel-engine") module for UCM-T calibration pipeline.

Status: **work in progress**, but integrated into calibration smoke (contract_ok).

## What it does (today)

The engine implements small, testable building blocks derived from the paper:

- Dispersion (forward branch) and group velocity (Eq. 16–17)
- Loop / Sagnac phase (Eq. 27, 29)
- 1D acoustic interval (Eq. 24)
- Horizon locator and surface gravity analogue Ω_H (Eq. 30)
- Hawking analogue temperature coefficient T_H = Ω_H/(2π) (Eq. 31, coefficient-only by default)

## Files

- `engine/rel_engine_v001.py` — engine core
- `pilot_rel.py` — wrapper that produces **results contract** outputs for the calibration pipeline

## Quick start

### Self-test (recommended)
```bash
python modules/rel/engine/rel_engine_v001.py --selftest

##Demo (prints JSON)

python modules/rel/engine/rel_engine_v001.py --demo
Demo includes two simple profiles showing that Ω_H and T_H scale as expected.

##Real calc: horizon from JSON (engine mode)
Create input JSON:

{"xs":[0,1,2],"vs":[0,2,4],"c0":2}

Run:

python modules/rel/engine/rel_engine_v001.py --calc-horizon in.json
Optional file output:

python modules/rel/engine/rel_engine_v001.py --calc-horizon in.json --out out.json
Note: input is read with utf-8-sig to tolerate Windows UTF-8 BOM.

##Wrapper usage (calibration pipeline)

The wrapper always emits:

results/results_global.json

results/results_items.csv

results/wrapper_status.json

Default behavior (smoke-safe): runs engine --demo.

To run "real" horizon calculation through the wrapper, set env var:

REL_INPUT_JSON — path to a JSON file with {xs,vs,c0}

Example (PowerShell):

$env:REL_INPUT_JSON = (Resolve-Path .\in.json).Path
python modules/rel/pilot_rel.py --outdir .\RUN --tag REL_REAL
Remove-Item Env:REL_INPUT_JSON
When available, the wrapper copies horizon_x, Omega_H, T_H_coeff into results_global.json.



Example (PowerShell, null speeds):
```powershell
$env:REL_INPUT_JSON = (Resolve-Path .\in_nullspeeds.json).Path
$env:REL_MODE = "null_speeds"
python modules/rel/pilot_rel.py --outdir .\RUN --tag REL_NS
Remove-Item Env:REL_INPUT_JSON
Remove-Item Env:REL_MODE


### Проверка
```powershell
cd C:\UCM\UCM-T
python -m py_compile modules/rel/engine/rel_engine_v001.py

Wrapper mode selection:
- `REL_MODE=horizon` (default when `REL_INPUT_JSON` is set) → engine `--calc-horizon`
- `REL_MODE=null_speeds` → engine `--calc-null-speeds` (expects `{x,xs,vs,c0}`)
- `REL_MODE=profile → engine --calc-profile (unified: horizon + null-speeds if x provided)

Example (PowerShell, null speeds):
```powershell
$env:REL_INPUT_JSON = (Resolve-Path .\in_nullspeeds.json).Path
$env:REL_MODE = "null_speeds"
python modules/rel/pilot_rel.py --outdir .\RUN --tag REL_NS
Remove-Item Env:REL_INPUT_JSON
Remove-Item Env:REL_MODE

## Quick check: profile mode (horizon)

```powershell
New-Item -ItemType Directory -Force C:\UCM\TMP_REL | Out-Null
python -c "import json; json.dump({'xs':[0,1,2],'vs':[0,2,4],'c0':2}, open(r'C:\UCM\TMP_REL\in_horizon.json','w',encoding='utf-8'), ensure_ascii=False)"

$env:REL_INPUT_JSON = "C:\UCM\TMP_REL\in_horizon.json"
$env:REL_MODE = "profile"
python -X utf8 -u C:\UCM\UCM-T\modules\rel\pilot_rel.py --outdir C:\UCM\TMP_REL\RUN_REL_HOR --tag REL_HOR
Remove-Item Env:REL_INPUT_JSON,Env:REL_MODE -ErrorAction SilentlyContinue
Expected in results/results_global.json:

horizon_x = 1.0

Omega_H = 2.0

T_H_coeff = 1/pi

### Optional dispersion inputs (paper v2)

For `REL_MODE=profile` you may pass (optional) parameters:
- `rho_inf` (or `rho_infty`)
- `kappa`
- `kappa_s`
- `eps` (default 0.01)

If provided, engine reports:
- `l_kappa`, `l_s`, `k_max_for_eps`
