$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runA = Join-Path $repoRoot "_fixtures/run_a"
$runB = Join-Path $repoRoot "_fixtures/run_b"
$runCsvBom = Join-Path $repoRoot "_fixtures/run_csv_bom"
$outReport = Join-Path $repoRoot "_fixtures/comparison_report.md"

python (Join-Path $repoRoot "compare_results_contract.py") --left $runA --right $runB --out $outReport
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python (Join-Path $repoRoot "compare_results_contract.py") --left $runA --check-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python (Join-Path $repoRoot "compare_results_contract.py") --left $runCsvBom --check-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
