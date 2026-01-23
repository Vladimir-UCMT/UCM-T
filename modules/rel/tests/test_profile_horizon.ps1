$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force C:\UCM\TMP_REL | Out-Null
python -c "import json; json.dump({'xs':[0,1,2],'vs':[0,2,4],'c0':2}, open(r'C:\UCM\TMP_REL\in_horizon.json','w',encoding='utf-8'), ensure_ascii=False)"

$env:REL_INPUT_JSON = "C:\UCM\TMP_REL\in_horizon.json"
$env:REL_MODE = "profile"

python -X utf8 -u (Join-Path $PSScriptRoot "..\pilot_rel.py") --outdir C:\UCM\TMP_REL\RUN_REL_HOR --tag REL_HOR | Out-Host

python -c "import json,math; p=r'C:\UCM\TMP_REL\RUN_REL_HOR\results\results_global.json'; d=json.load(open(p,'r',encoding='utf-8')); assert d.get('horizon_x')==1.0; assert d.get('Omega_H')==2.0; assert abs(d.get('T_H_coeff')-(1/math.pi))<1e-12; print('OK: profile horizon metrics')"

Remove-Item Env:REL_INPUT_JSON -ErrorAction SilentlyContinue
Remove-Item Env:REL_MODE -ErrorAction SilentlyContinue
