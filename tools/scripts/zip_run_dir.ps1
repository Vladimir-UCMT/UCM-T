param(
  [Parameter(Mandatory=$true)][string]$RunDir,
  [string]$ZipPath = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $RunDir)) { throw "RunDir not found: $RunDir" }

if (-not $ZipPath) {
  $parent = Split-Path -Parent $RunDir
  $name = Split-Path -Leaf $RunDir
  $ZipPath = Join-Path $parent ($name + ".zip")
}

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

Compress-Archive -Path (Join-Path $RunDir "*") -DestinationPath $ZipPath -Force
Write-Host "[ok] zipped: $ZipPath"
