# UCM-T calibration smoke-test (v2.3)
# Runs all modules quickly and validates results contract.
# RD is executed in engine --dry_run mode via wrapper flag --rd-no-run.

$ErrorActionPreference = "Stop"

$REPO = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO = Split-Path -Parent $REPO  # .../UCM-T

cd $REPO

$OUT = "C:\UCM\RUNS\CALIB_SMOKE_V23"
mkdir $OUT -Force | Out-Null

python tools\run_calib_all.py --outdir $OUT --rd-no-run
python tools\run_calib_all.py --outdir $OUT --check-contract

Write-Host ""
Write-Host "[done] Smoke run output: $OUT"
Write-Host "[done] Summary: $OUT\calib_summary.csv"
