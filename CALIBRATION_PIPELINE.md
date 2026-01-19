
# UCM-T Calibration Pipeline (v1)

This repo contains a unified calibration pipeline that runs multiple domain modules
(NV, Casimir, Rotation Curves, Ringdown) and publishes comparable outputs in a common format.

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

Some modules may also write logs at module root:

<module>\stdout.txt

<module>\stderr.txt

Summary file
After the run, the pipeline writes:

C:\UCM\RUNS\CALIB\calib_summary.csv

Columns:

module — nv / casimir / rc / rd

status — ok / error / missing_results

returncode — wrapper/engine return code (0 means “did not crash”)

has_items_csv — whether results_items.csv exists

error — short error reason (if any)

Typical troubleshooting
status = missing_results
The module did not publish:

results/results_global.json

Check whether the module crashed before publishing.
Look for:

<module>\stderr.txt

<module>\stdout.txt

status = error
Open:

<module>\results\results_global.json

<module>\results\wrapper_status.json

They contain the most direct error message and (for wrappers) traceback tail.

Reproducible baseline
The pipeline baseline is tagged:

calib-v1

To return to the baseline:

git checkout calib-v1

