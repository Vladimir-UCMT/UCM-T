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

## How to run (current)

This module currently provides a pilot pipeline. The exact runner scripts may change while the
benchmark is being frozen.

Start from the module directory:

- `modules/ringdown/`

Then follow the current run instructions referenced from the repository-level documentation and/or
the runner scripts committed in this module when available.

If you are running from an external frozen bundle, follow the bundle’s `README` first and treat this
file as a conceptual guide.

## Expected artifacts

Whenever possible, runs should emit artifacts compatible with the project-level results contract:

- `results/results_global.json`
- `results/results_items.csv`

See: `tools/results_contract.md`

Module-specific artifacts may also be produced (names may evolve during the pilot stage), e.g.:
- per-event tables (CSV) with event-level scores or parameters,
- diagnostic plots and log files.

## Validation (minimal)

After a run completes:
1) confirm that the expected artifacts exist (see above),
2) confirm that the run declares its configuration (engine version, model variant, dataset tag),
3) confirm that the summary metrics are consistent across repeated runs on the same input bundle.

Until the benchmark is frozen, numerical values are not guaranteed to match across commits.

## Freezing policy (how this becomes “canonical”)

This reproducibility guide will be upgraded from pilot to frozen once:
1) a stable dataset bundle is published externally (DOI),
2) the runner entry point is fixed and documented,
3) reference outputs are archived,
4) acceptance tolerances are documented.

When frozen, the canonical DOI will be added to `links.md`, and this file will reference it explicitly.
