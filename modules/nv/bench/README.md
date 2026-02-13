# NV bench: DC211 lock-in CW ODMR (AM 680 Hz) — Data Intake v1

This bench folder contains a **minimal reproducible raw→CSV intake** example for
DC211 lock-in CW ODMR sweeps.

- **Raw format:** DC211 text export (`Frequency, MHz`, `Channel1`, `Channel2`).
- **Primary signal (convention):** `Channel1` (lock-in output).
- **AM modulation:** 680 Hz.
- Two sweeps are included:
  - **B≈0 mT** (no applied field)
  - **B≈4 mT** (magnetic field applied; inferred from `Comments: ... with MF ...`)

## Layout

```
NV_DC211_LOCKIN_CW_ODMR_AM680_V1/
  raw/   # original TXT (or copied TXT)
  csv/   # standardized CSV: freq_mhz, channel1_v, channel2_v
  meta/  # per-file meta JSON
  manifest.json
```

## Rebuild CSV from raw

From repo root:

```bash
python -m tools.nv_ingest \
  --dataset-id NV_DC211_LOCKIN_CW_ODMR_AM680_V1 \
  --out-root modules/nv/bench/NV_DC211_LOCKIN_CW_ODMR_AM680_V1 \
  --in modules/nv/bench/NV_DC211_LOCKIN_CW_ODMR_AM680_V1/raw/dc211_no_field_raw.txt \
  --in modules/nv/bench/NV_DC211_LOCKIN_CW_ODMR_AM680_V1/raw/dc211_B4mT_raw.txt
```

This rewrites `csv/`, `meta/`, and `manifest.json`.

> Note: The current `modules/nv/pilot_nv.py` wrapper is still a **contract demo**
> (it does not yet fit ODMR lines). This bench is meant as a stable ingestion
> reference for the next NV analysis step.
