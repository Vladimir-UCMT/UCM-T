# Relativity module (rel)

Status: skeleton / demo.

- Engine: `modules/rel/engine/rel_engine_v001.py` (demo placeholder)
- Wrapper: `modules/rel/pilot_rel.py` publishes UCM-T results contract:
  - `results/results_global.json`
  - `results/results_items.csv`
  - `results/wrapper_status.json`

Run (standalone wrapper):

```powershell
python modules/rel/pilot_rel.py --outdir C:\UCM\RUNS\REL_DEMO
