# Core calibration kernel (draft plan)

Goal: calibrate shared "medium parameters" across modules (rel, nv, casimir, rc, rd) using existing benchmark outputs.
Non-goal: re-run heavy engines inside the kernel. Kernel consumes results/ artifacts that already satisfy the results contract.

## 0) Preconditions (when kernel makes sense)
- Each module exposes a stable set of candidate medium parameters (name, units/norm, meaning).
- At least 2 modules share ≥1 parameter candidate.
- Each module provides a reproducible benchmark producing:
  - results/results_global.json
  - results/results_items.csv (if applicable)
  - wrapper_status.json (contract_ok)

## 1) Inputs to the kernel (first version)
### 1.1 "Observations pack" (directory-based)
Kernel input is a list of run directories, each containing:
- module tag (e.g. rel/nv/casimir/rc/rd)
- results/results_global.json (required)
- results/results_items.csv (optional)
- wrapper_status.json (required)

Example CLI:
python tools/core_calib.py --runs C:\UCM\RUNS\CALIB_SMOKE_V23

### 1.2 Optional config (YAML/JSON)
- parameter name mapping / aliases
- which parameters to solve for
- which metrics to fit / weights
- priors / bounds (optional)
- aggregation rules for items.csv (optional)

## 2) Canonical objects (data model)
### 2.1 Parameter vector (theta)
A dictionary:
theta = {
  "c0": ...,
  "rho_inf": ...,
  "kappa": ...,
  "kappa_s": ...,
  ...
}

### 2.2 Observations (per module)
From results_global.json extract a normalized set:
obs[module] = {
  "metrics": {...},
  "n_items": ...,
  "meta": {...},
}

## 3) Objective function (first version)
- Weighted least squares on selected metrics:
  J(theta) = Σ_modules Σ_metrics w * (m_pred(theta) - m_obs)^2
- For v0: we likely cannot predict all metrics without re-running engines.
  Therefore first kernel version uses a "consistency / stitching" objective:
  - enforce shared parameters equality / bounds
  - optionally fit parameters that are directly reported by modules (if any)
  - track mismatch as diagnostics, not as a full physical inference engine

## 4) Outputs (kernel results contract, proposed)
Kernel produces:
tools/core_calib_runs/<timestamp>/
  results/
    results_global.json
    results_items.csv (optional)
  wrapper_status.json
  core_calib_report.md

### 4.1 results_global.json (minimum)
- tag: "CORE_CALIB_V0"
- inputs: list of run dirs + hashes
- selected_parameters: list
- theta_best: dict
- objective_value: float
- per_module_summary:
  - module: name
  - used_metrics: [...]
  - residual_norm: ...
  - notes: ...

### 4.2 wrapper_status.json
- schema: ucm_wrapper_status_v1
- status: ok/error
- contract_ok: true/false

## 5) Minimal roadmap (micro-steps)
S0: Spec only (this doc)
S1: Collector
- scan runs/, validate contract_ok, load results_global + (optional) items
- write merged table for inspection

S2: Schema alignment
- parameter alias table (cross-module)
- metric selection rules per module

S3: Consistency kernel v0
- checks only (no optimization): detect conflicts, missing params, unit mismatches
- outputs core_calib_report.md + status ok

S4: Optional optimization v0
- solve a tiny system for a tiny set (e.g. c0 shared between rel+another module, if/when it exists)

## 6) Open questions to resolve later (do not block S1-S3)
- which parameters are truly shared across which modules (beyond names)
- units / nondimensionalization
- whether kernel is allowed to call engines (likely no for v0)
- robust weighting / outlier handling
- where "truth" comes from (benchmarks vs real data)

