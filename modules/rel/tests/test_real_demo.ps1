$ErrorActionPreference = "Stop"

$demo = Join-Path $PSScriptRoot "..\examples\real_demo.json"
$demo = (Resolve-Path $demo).Path

$env:REL_INPUT_JSON = $demo
$env:REL_MODE = "profile"

python -X utf8 -u (Join-Path $PSScriptRoot '..\pilot_rel.py') --outdir C:\UCM\TMP_REL\RUN_REL_REAL_DEMO --tag REL_REAL_DEMO | Out-Host

python -c "import json,math; p=r'C:\UCM\TMP_REL\RUN_REL_REAL_DEMO\results\results_global.json'; d=json.load(open(p,'r',encoding='utf-8')); \
assert abs(d.get('horizon_x')-1.0)<1e-12; \
assert abs(d.get('l_kappa')-1.0)<1e-12; \
assert abs(d.get('phase_loop_coeff')-1.25)<1e-12; \
assert abs(d.get('phase_loop')-3.75)<1e-12; \
assert abs(d.get('phase_sagnac')-30.0)<1e-12; \
assert abs(d.get('phase_ab')-12.5)<1e-12; \
assert abs(d.get('Omega0_over_c0')-3.0)<1e-12; \
print('OK: real demo')"

Remove-Item Env:REL_INPUT_JSON,Env:REL_MODE -ErrorAction SilentlyContinue
