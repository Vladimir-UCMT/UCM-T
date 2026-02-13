# NV bench datasets

This directory stores small benchmark datasets for the NV module used by the Calibration Kit.

## Rule
Do NOT place raw/csv/meta directly under `modules/nv/bench/`.
Each dataset must live in its own folder:

  modules/nv/bench/<DATASET_ID>/
    raw/
    csv/
    meta/
    manifest.json

## Included datasets
- NV_DC211_LOCKIN_CW_ODMR_AM680_V1
  DC211 lock-in CW ODMR (Channel1 intended), AM = 680 Hz.
  Two spectra: B≈0 mT and B≈4 mT.
  See: NV_DC211_LOCKIN_CW_ODMR_AM680_V1/manifest.json
