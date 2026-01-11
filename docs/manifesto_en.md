# Methodological Manifesto of UCM-T

## UCM Calibration Hub (reproducible pipeline)

This repository is a **calibration hub**, not a collection of ideas. Each module implements a reproducible pipeline:

**data → model → metric → artifacts → comparison vs baseline**

Minimum requirements for each module:
1) data or a measurement protocol,
2) a baseline hypothesis,
3) a numeric score,
4) reproducible output artifacts,
5) a reproducibility guide.

Whenever possible, runs should follow the shared results contract:
- `results/results_global.json`
- `results/results_items.csv`
See: `tools/results_contract.md`.

Core modules:
- Ringdown / CVN: `modules/ringdown/`
- Rotation curves (RC V12): `modules/rotation-curves/`


Unified Compressible Medium Theory (UCM-T) is treated not as a completed
physical theory, but as an **operational research framework**
focused on calibratable and reproducible testing.

## Core Principle

A physical statement is considered meaningful only if it can be
associated with a concrete procedure:
a calibration step, a numerical test, a comparative benchmark,
or a potentially falsifying experiment.

UCM-T deliberately avoids reliance on entities or assumptions
that are not operationally accessible,
unless they lead to observable consequences.

## Unified Calibration Pipeline

Different physical domains—such as gravitation, astrophysics,
quantum systems, and condensed media—are approached
within a unified operational framework:

- models are formulated operationally,
- parameters are calibrated against data,
- results are compared in a reproducible manner.

Agreements or discrepancies across domains
are treated as empirical facts,
not as arguments in favor of a preconceived interpretation.

## Attitude Toward Negative Results

Negative results, unstable regimes, and unsuccessful calibrations
are considered informative and worthy of documentation.

UCM-T does not aim at confirmation at any cost;
the possibility of falsification is regarded not as a weakness,
but as a necessary property of a viable theoretical framework.

## Project Status

UCM-T is under active development.
Its architecture, tools, and internal conventions
may evolve as experience and data accumulate.

This repository serves as a coordination hub
for models, computational tools,
and reproducible tests related to UCM-T.
