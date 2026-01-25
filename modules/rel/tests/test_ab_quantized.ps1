$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force C:\UCM\TMP_REL | Out-Null
python -c "import json; json.dump({'xs':[0,1,2],'vs':[0,2,4],'c0':2,'omega':10,'chi':2,'n':5,'Gamma0':2}, open(r'C:\UCM\TMP_REL\in_abq.json','w',encoding='utf-8'), ensure_ascii=False)"

$env:REL_INPUT_JSON = "C:\UCM\TMP_REL\in_abq.json"
$env:REL_MODE = "profile"

python -X utf8 -u (Join-Path $PSScriptRoot '..\pilot_rel.py') --outdir C:\UCM\TMP_REL\RUN_REL_ABQ --tag REL_ABQ | Out-Host

python -c "import json; p=r'C:\UCM\TMP_REL\RUN_REL_ABQ\results\results_global.json'; d=json.load(open(p,'r',encoding='utf-8')); \
assert abs(d.get('Gamma')-10.0)<1e-12; \
assert abs(d.get('phase_ab')-12.5)<1e-12; \
print('OK: AB quantized')"

Remove-Item Env:REL_INPUT_JSON,Env:REL_MODE -ErrorAction SilentlyContinue
