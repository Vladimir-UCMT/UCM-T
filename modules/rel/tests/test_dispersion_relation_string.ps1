$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force C:\UCM\TMP_REL | Out-Null
python -c "import json; json.dump({'xs':[0,1,2],'vs':[0,2,4],'c0':2,'rho_inf':1,'kappa':4,'kappa_s':16,'eps':0.01}, open(r'C:\UCM\TMP_REL\in_disp.json','w',encoding='utf-8'), ensure_ascii=False)"

$env:REL_INPUT_JSON = "C:\UCM\TMP_REL\in_disp.json"
$env:REL_MODE = "profile"

python -X utf8 -u (Join-Path $PSScriptRoot '..\pilot_rel.py') --outdir C:\UCM\TMP_REL\RUN_REL_DISP2 --tag REL_DISP2 | Out-Host

python -c "import json; p=r'C:\UCM\TMP_REL\RUN_REL_DISP2\results\results_global.json'; d=json.load(open(p,'r',encoding='utf-8')); \
assert d.get('dispersion_relation','').startswith('w^2=c0^2 k^2'); print('OK: dispersion relation string')"

Remove-Item Env:REL_INPUT_JSON,Env:REL_MODE -ErrorAction SilentlyContinue

