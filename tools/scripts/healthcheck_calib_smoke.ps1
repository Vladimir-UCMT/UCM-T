# tools/scripts/healthcheck_calib_smoke.ps1
# One-button health check for calibration smoke + collect/check

param(
  [string]$RunDir = "C:\UCM\RUNS\CALIB_SMOKE_V23",
  [string]$UCM_C0 = "2.0",
  [string]$UCM_RHO_INF = "0.0",
  [string]$UCM_KAPPA = "0.0",
  [string]$UCM_KAPPA_S = "0.0"
)

$ErrorActionPreference = "Stop"

function Invoke-External {
  param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$false)][string[]]$Args = @()
  )
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "External command failed: $Exe $($Args -join ' ') (exit=$LASTEXITCODE)"
  }
}

Write-Host "[hc] repo: $(Get-Location)"
Write-Host "[hc] RunDir: $RunDir"
Write-Host "[hc] Phase0 env: UCM_C0=$UCM_C0 UCM_RHO_INF=$UCM_RHO_INF UCM_KAPPA=$UCM_KAPPA UCM_KAPPA_S=$UCM_KAPPA_S"

# Set env vars for this session/process (children inherit)
$env:UCM_C0      = $UCM_C0
$env:UCM_RHO_INF = $UCM_RHO_INF
$env:UCM_KAPPA   = $UCM_KAPPA
$env:UCM_KAPPA_S = $UCM_KAPPA_S

# 1) Smoke run (run in separate PowerShell with Bypass so ExecutionPolicy doesn't block)
Write-Host "[hc] step 1/3: smoke"
Invoke-External "powershell.exe" @(
  "-NoProfile",
  "-ExecutionPolicy", "Bypass",
  "-File", ".\tools\scripts\run_calib_smoke_v23.ps1"
)

# 2) Collect
Write-Host "[hc] step 2/3: collect"
$collect = & python -X utf8 .\tools\core_calib_collect.py --run-dir $RunDir --aliases .\tools\core_calib_aliases.json
if ($LASTEXITCODE -ne 0) {
  throw "collect failed (exit=$LASTEXITCODE)"
}
$collect | ForEach-Object { Write-Host $_ }

$outDir = ($collect | Select-String "^\[done\] out_dir:" | ForEach-Object { $_.Line.Split(":",2)[1].Trim() })
if (-not $outDir) { throw "Could not parse out_dir from collect output" }

$merged = Join-Path $outDir "core_calib_merged.csv"
if (-not (Test-Path $merged)) { throw "Merged CSV not found: $merged" }

# 3) Check
Write-Host "[hc] step 3/3: check"
Invoke-External "python" @("-X","utf8",".\tools\core_calib_check.py","--merged",$merged)

Write-Host "[hc] OK: smoke + collect + check passed"
exit 0
