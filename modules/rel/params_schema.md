# rel: params schema (inputs/outputs)

This file is a compact map of REL parameters for future cross-module calibration.
Scope: modules/rel (wrapper + engine). Focus: what is a medium parameter vs experiment/control.

## A) Inputs

### Required (profile)
- `xs` : list[float] — координаты (1D профиль)
- `vs` : list[float] — скорость потока v(x)
- `c0` : float — фоновая скорость

### Optional (dispersion / paper v2)
- `rho_inf` (alias `rho_infty`) : float
- `kappa` : float
- `kappa_s` (alias `kappas`) : float
- `eps` : float (default 0.01)

### Optional (loop phases / paper v2)
- `omega` : float
- `chi` : float
- `loop_int_vdl` : float  (user-supplied ∮ v·dl)

### Optional (Sagnac special case)
- `Omega` (alias `omega_rot`) : float
- `area` (alias `A`) : float

### Optional (AB-like special case)
- `Gamma` (alias `circulation`) : float
- Quantized mode (used only if `Gamma` not provided):
  - `n` : float/int
  - `Gamma0` (alias `gamma0`) : float

### Optional (vorticity link)
- `Omega0` (alias `vorticity`) : float

## B) Outputs (engine -> wrapper -> results_global)

### Trace/debug (wrapper)
- `engine_cmd`
- `rel_mode`
- `rel_input_json_basename`

### Horizon block
- `horizon_x`
- `Omega_H`
- `T_H_coeff`

### Dispersion block
- `l_kappa`
- `l_s`
- `k_lkappa_max_for_eps`
- `k_max_for_eps`
- `eps`
- `dispersion_relation`

### Loop phases
- `phase_loop_coeff`  (= omega/(chi*c0^2))
- `loop_int_vdl`
- `phase_loop`        (= phase_loop_coeff*loop_int_vdl)

### Special cases
- Sagnac:
  - `Omega`, `area`, `phase_sagnac` (= 2*phase_loop_coeff*(Omega*area))
- AB-like:
  - `Gamma`, `phase_ab` (= phase_loop_coeff*Gamma)
  - (quantized) `n`, `Gamma0` may also appear

### Vorticity link
- `Omega0`
- `Omega0_over_c0` (= Omega0/c0)

## C) Classification for future "medium calibration core"

### Likely "medium parameters" candidates (cross-module)
- `c0`
- `rho_inf`
- `kappa`, `kappa_s` (if other modules use same dispersion model)

### Likely "experiment / geometry / control" (not medium)
- `xs`, `vs` (profile definition)
- `omega`, `chi`
- `loop_int_vdl`, `Omega`, `area`, `Gamma`, `n`, `Gamma0`, `Omega0`

## D) Notes
- Geometry is currently provided in "collapsed" form (scalars). No contour/surface integration is performed by engine.
- Aliases exist for convenience; if we later build a global schema, we should standardize preferred names.
