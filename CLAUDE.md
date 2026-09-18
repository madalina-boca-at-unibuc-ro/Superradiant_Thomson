# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A numerical simulation of superradiant (coherent, nonlinear) Thomson scattering: an ensemble of
relativistic electrons driven by a pulsed Laguerre-Gauss laser beam, radiating a frequency-domain
Faraday tensor field that is coherently summed onto a 2D observation screen.

The `md_helpers_proposed/` directory is the scientific specification this code implements (units,
LG field, equations of motion, screen geometry, Fourier-transform forms, non-linear Thomson
frequencies, angular-momentum flux). `md_helpers_implemented/` documents, module by module, how
each spec item was actually implemented and where its tests live — read the matching file there
before changing a module's physics or conventions. Treat these as living design docs, not just
history: when the spec in `md_helpers_proposed/` and the code diverge, that's worth flagging.

## Environment

There is no `.venv` inside this worktree. The project's virtualenv lives in the main checkout at
`/home/madalina/Dropbox/work/bin/python/Superradiant_Thomson/.venv` (Python 3.14, numpy, scipy,
matplotlib, pytest). Use that interpreter to run code and tests from this worktree, e.g.:

```bash
/home/madalina/Dropbox/work/bin/python/Superradiant_Thomson/.venv/bin/python -m pytest
```

## Commands

```bash
# Run the full test suite (tests are unittest.TestCase classes, run via pytest)
.venv/bin/python -m pytest

# Run a single test file / class / test
.venv/bin/python -m pytest tests/test_screen.py
.venv/bin/python -m pytest tests/test_screen.py::TestScreenEvaluator
.venv/bin/python -m pytest tests/test_screen.py::TestScreenEvaluator::test_form1_vs_form2_agreement -v

# Run the full simulation (writes figures/arrays to a new run directory)
python main.py            # headless (Agg backend), writes under ~/output/SRT_<timestamp>_<id>/
python main.py --show     # also opens interactive plot windows
```

`main.py` is also importable: `sample_pulse(inputs=...)` runs just the temporal-envelope step,
`main(show=..., output_root=..., inputs=...)` runs the whole pipeline and returns the run directory.
`INPUTS` at module scope is the example configuration dict — edit it or pass a custom `inputs` dict
with the same keys to `main()`/`sample_pulse()`.

## Architecture

Pipeline stages, matching `md_helpers_proposed/00-general_advice.md`'s mandated layering
(`configuration -> physics kernels -> orchestration/parallelism -> output/plots`):

1. **`parameters.py`** — dimensional unit system and extensible parameter resolver. All inputs are
   given as `{'value': ..., 'unit': ...}` (or bare numbers, implicitly atomic units) and resolved
   into Hartree atomic units via a `ScaleRegistry` of `Scale`s, some of which depend on other
   resolved parameters (e.g. `lambda` and `T` depend on `laser.omega`; `w_0` unit depends on
   `laser.w_0`). Each module contributes its own parameter `Parameter` schema dict (e.g.
   `lg_schema()`, `electron_schema()`, `screen_schema()`) which are merged with `|` before calling
   `ParameterResolver(...).resolve(inputs)`. Unknown parameter names and dimensionally incompatible
   units raise `ResolutionError` — inputs are never silently repaired.
2. **`laser.py` + `lg_mode.py`** — temporal envelope (`PulseTiming`, `TemporalFactor`: truncated
   Gaussian wings around a flat top) and paraxial Laguerre-Gauss spatial mode (`LGMode`), combined
   in `evaluate_laser_fields` to give the six E/B field components, and in
   `faraday_tensor_from_fields` to build the contravariant Faraday tensor `F^{mu nu}`.
3. **`electron.py`** — `generate_electron_initial_conditions` samples an electron bunch (uniform
   disk x cylinder height in space, Gaussian in momentum); `solve_single_electron_trajectory`
   integrates the relativistic Lorentz-force ODE (`scipy.integrate.solve_ivp`, DOP853) for
   `r^mu(tau), u^mu(tau)`, recomputing `w^mu` (4-acceleration) from the equation-of-motion
   right-hand side rather than differentiating `u`. Proper-time sampling is Doppler-adjusted
   (`compute_doppler_adjusted_tau_eval`) so a boosted bunch still samples the full laser phase.
4. **`screen.py`** — `ScreenGeometry` (rectangular or annular observation screen) and
   `ScreenResult` (accumulated `F_l`/`F_s`/`F_b` — long-range, short-range/velocity, and boundary
   terms of the frequency-domain Faraday tensor). Two independent Fourier-transform formulas
   (`method='simplified'`/Form 2 and `'direct'`/Form 1) are implemented from scratch rather than
   sharing an integrand, by design (see `md_helpers_proposed/00-general_advice.md`), so their
   agreement is a real cross-check, not a shared-bug tautology — do not "simplify" by merging them.
   They agree well at realistic screen distances (relative difference shrinks with `screen.z_screen`
   and converges with `electron.NT`; see the near-field caveat below) — do not re-litigate this
   without a fresh, carefully-checked test case.
   `compute_screen_emitted_field_from_laser_and_bunch` is the top-level orchestrator: it generates
   the bunch, solves each electron's trajectory once, computes and coherently sums that electron's
   screen contribution, and does this per-electron work in parallel `ProcessPoolExecutor` workers
   (one electron per worker chunk), keeping only a handful of sample trajectories
   (`max_stored_trajectories`) for later plotting. `ScreenResult` also derives the spectral
   electromagnetic observables of `md_helpers_proposed/11-numerical_calculation_of_observables.md`
   from `F_total` (spin/orbital/total angular-momentum density and flux along Oz, energy density,
   Poynting flux, and `integrate_over_screen` for their per-harmonic surface integrals) — see
   `md_helpers_implemented/11-numerical_calculation_of_observables.md`. These replaced an older,
   now-discarded kinetic angular-momentum-flux formula and an ad hoc `|F|^2` "intensity"; don't
   resurrect either from `md_helpers_implemented/10-angular_momentum_flux_density-old.md`.
   `tests/test_output.py::test_screen_observable_flux_density_ratio_equals_c` is a useful physical
   regression guard for this code: in the far field, integrated flux / integrated density along Oz
   must equal +c or -c (sign per which side of the source the screen is on) for energy, total
   angular momentum, and spin alone — a real bug in any of these formulas tends to blow that ratio
   up by orders of magnitude or flip its sign, not perturb it slightly.
5. **`four_vector.py`** — shared Minkowski-space primitives (metric signature `(+,-,-,-)`, index
   convention `(0,1,2,3)=(ct,x,y,z)`) used by both the electron ODE and the screen field formulas.
6. **`plotting/`** — one module per stage (`laser.py`, `electron.py`, `screen.py`, shared `style.py`);
   every plotter returns `(figure, axes)` and never calls `plt.show()`/`savefig()` itself — callers
   (`main.py`, tests) control display and saving. `screen.py` has two per-harmonic figure
   generators, both writing one subfolder per frequency under `run_dir` when called from `main.py`:
   `generate_all_screen_breakdown_plots` decomposes the raw Faraday tensor per-harmonic into
   `F_l`/`F_s`/`F_b`/total, 2x2 panel (Real/Imag/Modulus/Phase), into
   `screen_breakdown_N_<N>_omega_<val>_omega0/`; `generate_all_screen_observable_plots` plots the 6
   derived physical observables (energy density/flux, total angular-momentum density/flux, spin
   density/flux — all from `F_total` only) into `screen_observables_N_<N>_omega_<val>_omega0/`.
   There is deliberately no aggregate all-harmonics-in-one-figure plot for any screen quantity
   anymore — every screen plot is per-harmonic, in its own folder.
7. **`output.py`** — `create_run_directory` makes a fresh timestamped+uuid directory under
   `~/output` (never inside the source tree) using exclusive `mkdir` so runs never collide or get
   silently overwritten; `write_json` writes atomically (temp file + `replace`) so readers never see
   partial output. Per-harmonic screen observable surface integrals (and their flux/density
   ratios) go only into `run.json`'s `metadata['screen_observables']` — there is deliberately no
   separate `run_log.txt` file (it existed briefly and was removed; don't re-add it).

### Threading env vars in `main.py`

`main.py` sets `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/etc. to `1` at import time,
*before* numpy is imported. Electron trajectories are parallelized across OS processes (one electron
per worker), so without this each worker's BLAS backend would itself spawn a full-core thread pool,
oversubscribing the machine. Preserve this ordering (env vars set before `import numpy`) in any new
entry point that also does process-level parallelism over electrons.

### `direct` vs `simplified` agreement — checked, and near-field caveat

`screen.method='direct'` (Form 1) and `'simplified'` (Form 2, the default everywhere in this
codebase) were checked against each other numerically (2026-09-18). **They agree well at any
realistic screen distance** and the residual shrinks with `electron.NT`:

| `screen.z_screen` | rel. diff in `F_total` |
|---|---|
| 0.01 lambda | 23% |
| 1 lambda | 4.6% |
| 10 lambda | 0.83% |
| 100 lambda | 0.092% |
| 1000 lambda | 0.0094% |
| 25000 lambda (this repo's `INPUTS` scale) | 0.0013%, shrinking further with `electron.NT` |

The relative difference grows quickly, though, as the screen approaches the source at
**sub-wavelength** distances (`screen.z_screen` given in bare atomic units rather than the
`'lambda'` unit is an easy way to land there by accident, since 1 lambda is ~15000 a.u. for
`laser.omega=0.057` — e.g. `z_screen=1000.0` with no unit is only ~0.066 lambda out, deep
near-field). Whether that near-field divergence itself is a bug or expected behavior of the two
forms' derivations has not been investigated — it doesn't matter for any realistic configuration
this codebase runs (`z_screen` is always many wavelengths out in `main.py`'s `INPUTS` and in the
saved `~/output` runs). Don't re-test agreement between the two forms with a `z_screen` that isn't
expressed `{'value': ..., 'unit': 'lambda'}` (or an equivalently large distance) — a sub-wavelength
screen produced a spurious "~95%-6250x mismatch, confirmed structural bug" conclusion here that
didn't hold up once checked at real distances, and cost real time to walk back.

### Conventions to preserve

- Four-vector index order `(0,1,2,3) = (ct,x,y,z)`; Faraday tensor's six independent upper-triangular
  components are always ordered `(F01,F02,F03,F12,F13,F23)` (see `COMPONENT_NAMES`/`COMPONENT_INDICES`
  in `screen.py`).
- All internal computation is in Hartree atomic units (`c = 1/alpha`); SI conversion only happens at
  the input/output boundary via `parameters.py`'s unit registry.
- Complex frequency-domain tensors are always summed across electrons *before* taking `|.|^2` or
  another quadratic observable (coherent accumulation) — never sum intensities.
- Trajectories are integrated once per electron and reused for every screen pixel/frequency; don't
  reintegrate per-pixel.
