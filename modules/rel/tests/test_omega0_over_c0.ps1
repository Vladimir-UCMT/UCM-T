$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force C:\UCM\TMP_REL | Out-Null
python -c "import json; json.dump({'xs':[0,1,2],'vs':[0,2,4],'c0':2,'Omega0':6}, open(r'C:\UCM\TMP_REL\in_omega0.json','w',encoding='utf-8'), ensure_ascii=False)"

$env:REL_INPUT_JSON = "C:\UCM\TMP_REL\in_omega0.json"
$env:REL_MODE = "profile"

python -X utf8 -u (Join-Path $PSScriptRoot '..\pilot_rel.py') --outdir C:\UCM\TMP_REL\RUN_REL_OMEGA0 --tag REL_OMEGA0 | Out-Host

python -c "import json; p=r'C:\UCM\TMP_REL\RUN_REL_OMEGA0\results\results_global.json'; d=json.load(open(p,'r',encoding='utf-8')); \
assert abs(d.get('Omega0_over_c0')-3.0)<1e-12; print('OK: Omega0_over_c0')"

Remove-Item Env:REL_INPUT_JSON,Env:REL_MODE -ErrorAction SilentlyContinue
