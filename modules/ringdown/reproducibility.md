# Reproducibility — Ringdown (CVN) — Pilot

Status: **pilot / evolving** (this benchmark definition is not frozen yet).

This page describes the current reproduction procedure for the Ringdown / CVN module and the
expected run artifacts. Until a frozen release is published, this document should be treated as a
living guide.

## Scope

Goal: evaluate a ringdown model against catalog-level posterior data (e.g., per-event posteriors for
mode parameters), producing comparable summary artifacts across runs.

This module is designed to support:
- reproducible re-runs of the current pipeline,
- comparison of model variants against a baseline hypothesis,
- clean separation between code (GitHub) and frozen bundles (external archives).

## Baseline

Baseline hypothesis (current): **CONST/GR** (no deviation, or constant shifts).

Note: exact baseline definitions may evolve as the benchmark is finalized.

## Data policy

Large datasets and full benchmark outputs must not be committed to Git.

Data must be provided via:
- external archives (Zenodo or equivalent) and/or
- a documented acquisition/preparation procedure.

Canonical links to stable archives will be listed in: `links.md`.

## How to run (demo, from this repository)

Status: pilot. The demo dataset shipped in this repository is synthetic and intended only to verify
that the pipeline runs end-to-end.

From the repository root:

```bash
python -X utf8 modules/ringdown/engine/core/pilot_cvn_rd.py --bench RD_DEMO_221 --tag DEMO --score model_nll --B 200 --root modules/ringdown
Outputs are written under:

modules/ringdown/RUNS/

Expected artifacts

This pilot pipeline produces:

modules/ringdown/RUNS/<bench>/<timestamp>_<bench>_<tag>/results_global.json

modules/ringdown/RUNS/<bench>/<timestamp>_<bench>_<tag>/results_event.csv

Whenever possible, future benchmark runs will also emit artifacts compatible with the shared results
contract:

results/results_global.json

results/results_items.csv

See: tools/results_contract.md

Validation (minimal)

After a demo run completes:

Confirm the two files above exist (results_global.json, results_event.csv).

Confirm results_global.json records the run configuration (bench, tag, score, B and/or model id).

Re-run the command on the same machine and confirm the summary statistics are stable within
expected Monte-Carlo noise (pilot stage).

## Freezing policy (how this becomes “canonical”)

This reproducibility guide will be upgraded from pilot to frozen once:
1) a stable dataset bundle is published externally (DOI),
2) the runner entry point is fixed and documented,
3) reference outputs are archived,
4) acceptance tolerances are documented.

When frozen, the canonical DOI will be added to `links.md`, and this file will reference it explicitly.
