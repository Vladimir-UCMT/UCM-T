# UCM-T Calibration Pipeline (v2)

This repo contains a unified calibration pipeline that runs multiple domain modules
(NV, Casimir, Rotation Curves, Ringdown) and publishes comparable outputs in a common format.

See also: `CALIBRATION_API.md` (frozen contract and status semantics).

## Quick start (Windows / PowerShell)

From repo root:

```powershell
cd C:\UCM\UCM-T
mkdir C:\UCM\RUNS\CALIB -Force
python tools\run_calib_all.py --outdir C:\UCM\RUNS\CALIB
Where outputs go
The pipeline creates one folder per module:

C:\UCM\RUNS\CALIB\nv\

C:\UCM\RUNS\CALIB\casimir\

C:\UCM\RUNS\CALIB\rc\

C:\UCM\RUNS\CALIB\rd\

Each module publishes results in:

<module>\results\results_global.json

<module>\results\results_items.csv

<module>\results\wrapper_status.json

After the run, the pipeline writes:

C:\UCM\RUNS\CALIB\calib_summary.csv

C:\UCM\RUNS\CALIB\calib_summary.json

Common modes
Dry run (only check pilots exist + create folders):

python tools\run_calib_all.py --outdir C:\UCM\RUNS\CALIB --dry-run
Check engines (also verify key engine files exist; currently RC):


python tools\run_calib_all.py --outdir C:\UCM\RUNS\CALIB --check-engines
Check contract for an existing run directory (no execution):

python tools\run_calib_all.py --outdir C:\UCM\RUNS\CALIB --check-contract
Summary CSV (calib_summary.csv)
Columns:

module — nv / casimir / rc / rd

status — see CALIBRATION_API.md (e.g., ok / error / missing_results / contract_ok / bad_items_csv ...)

returncode — process return code for normal runs (0 means the script did not crash)

has_items_csv — whether results_items.csv exists

outdir — module output folder

error — short error reason (if any)

Typical troubleshooting
status = missing_results
The module did not publish:

results/results_global.json

and/or results/results_items.csv

Check the module output folder for logs and wrapper status:

<module>\results\wrapper_status.json

<module>\stdout_tail (in calib_summary.json)

status = error
Open:

<module>\results\wrapper_status.json

<module>\results\results_global.json

They contain the most direct error message and (for wrappers) traceback tail.

Reproducible baseline
The pipeline baseline is tagged: calib-v2
## Smoke test

Fast end-to-end validation:

```powershell
powershell -ExecutionPolicy Bypass -File tools/scripts/run_calib_smoke_v23.ps1

Reference notes: benchmarks/CALIB_SMOKE_V23/README.md.



