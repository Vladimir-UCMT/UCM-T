# tools/scripts/healthcheck_ultrafast_zip.ps1
# One-button: run ultrafast healthcheck and zip the resulting run_dir

param(
  [string]$RunsRoot = "C:\UCM\RUNS",
  [string]$OutDir = "",
  [string]$UCM_C0 = "2.111",
  [string]$UCM_RHO_INF = "0.123",
  [string]$UCM_KAPPA = "0.00456",
  [string]$UCM_KAPPA_S = "0.0789"
)

$ErrorActionPreference = "Stop"

# Repo root = two levels up from this script (tools/scripts -> repo)
$REPO = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $REPO

if (-not $OutDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutDir = Join-Path $RunsRoot ("CALIB_HEALTH_ULTRAFAST_" + $stamp)
}

Write-Host "[hc] repo:     $REPO"
Write-Host "[hc] RunsRoot: $RunsRoot"
Write-Host "[hc] OutDir:   $OutDir"
Write-Host "[hc] Phase0:   UCM_C0=$UCM_C0 UCM_RHO_INF=$UCM_RHO_INF UCM_KAPPA=$UCM_KAPPA UCM_KAPPA_S=$UCM_KAPPA_S"

python -X utf8 -m tools.calib healthcheck `
  --ultrafast `
  --runs-root $RunsRoot `
  --outdir $OutDir `
  --phase0-c0 $UCM_C0 `
  --phase0-rho-inf $UCM_RHO_INF `
  --phase0-kappa $UCM_KAPPA `
  --phase0-kappa-s $UCM_KAPPA_S

if ($LASTEXITCODE -ne 0) {
  throw "healthcheck ultrafast failed (exit=$LASTEXITCODE)"
}

# Zip run_dir (everything is inside now, including CORE_CALIB_COLLECT_*)
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\scripts\zip_run_dir.ps1 -RunDir $OutDir

Write-Host "[hc] OK: ultrafast healthcheck + zip done"
exit 0
