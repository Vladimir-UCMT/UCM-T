# UCM-T calibration smoke-test (v2.3)
$ErrorActionPreference = "Stop"

# This script lives in: <repo>\tools\scripts\
# Repo root is two levels up.
$REPO = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $REPO

$OUT = "C:\UCM\RUNS\CALIB_SMOKE_V23"
mkdir $OUT -Force | Out-Null

python (Join-Path $REPO "tools\run_calib_all.py") --outdir $OUT --rd-no-run
python (Join-Path $REPO "tools\run_calib_all.py") --outdir $OUT --check-contract

Write-Host ""
Write-Host "[done] Smoke run output: $OUT"
Write-Host "[done] Summary: $OUT\calib_summary.csv"
