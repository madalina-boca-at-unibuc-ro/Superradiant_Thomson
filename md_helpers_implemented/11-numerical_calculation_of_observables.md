# Implemented: Numerical Calculation of Spectral Electromagnetic Observables

Module: `superradiant_thomson/screen.py` (`ScreenResult` methods). Plotting:
`superradiant_thomson/plotting/screen.py`. Output: `run.json`, written via
`superradiant_thomson/output.py`'s `write_json`, wired into `main.py`. Tests: `tests/test_screen.py`,
`tests/test_output.py`.

Implements `md_helpers_proposed/11-numerical_calculation_of_observables.md`, replacing the
outdated `md_helpers_implemented/10-angular_momentum_flux_density-old.md` (kinetic $\mathbf r
\times \mathbf g$ angular-momentum flux) and the old ad hoc `ScreenResult.intensity`
($\sum_{\mu<\nu}|F_{\text{total}}^{\mu\nu}|^2$) with the canonical spin/orbital decomposition and
proper energy density/flux, all derived from the same total Fourier-transformed Faraday tensor
`ScreenResult.F_total`.

---

## 1. Field reconstruction

`ScreenResult.fields(c)` reconstructs the complex Fourier field components from the six stored
upper-triangular components (`F01,F02,F03,F12,F13,F23`) using $F^{ji}=-F^{ij}$ together with the
proposed doc's mapping $\tilde E_x=cF^{10},\dots,\tilde B_x=F^{32},\dots$:

$$\tilde E_x=-cF^{01},\ \tilde E_y=-cF^{02},\ \tilde E_z=-cF^{03},\quad
\tilde B_x=-F^{23},\ \tilde B_y=F^{13},\ \tilde B_z=-F^{12}$$

This is the same tensor-to-field convention used by `lg_mode.faraday_tensor_from_fields`.

## 2. Atomic-unit constant

`EPSILON_0_AU = 1/(4*pi)` (module constant in `screen.py`): Hartree atomic units fix the Coulomb
constant $1/(4\pi\varepsilon_0)=1$, so every SI-form formula from the proposed doc is evaluated
with $\varepsilon_0=1/(4\pi)$ and $c$ in atomic units.

## 3. Implemented quantities (all `ScreenResult` methods, shape `(N_omega, Ny, Nx)` unless noted)

- `spin_angular_momentum_density_z(c)` — $dS_z/d\omega = (4\varepsilon_0/\omega)\,\mathrm{Im}[\tilde E_x^*\tilde E_y]$.
- `orbital_angular_momentum_density_z(c)` — $dL_z/d\omega = (2\varepsilon_0/\omega)\sum_i\mathrm{Im}[\tilde E_i^*\,\hat L_z\tilde E_i]$,
  where $\hat L_z = x\partial_y - y\partial_x$ on the rectangular grid (`numpy.gradient`) or
  $\hat L_z=\partial_\phi$ on the annular grid (periodic central differences when the screen spans
  a full $2\pi$, one-sided `numpy.gradient` otherwise).
- `total_angular_momentum_density_z(c)` — $dJ_z/d\omega = dL_z/d\omega + dS_z/d\omega$.
- `spin_angular_momentum_flux_zz(c)` — $d\Sigma_{zz}/d\omega = (2\varepsilon_0c^2/\omega)\,\mathrm{Im}[\tilde B_x^*\tilde E_x+\tilde B_y^*\tilde E_y-\tilde B_z^*\tilde E_z]$.
- `orbital_angular_momentum_flux_zz(c)` — $d\Lambda_{zz}/d\omega = (2\varepsilon_0c^2/\omega)\,\mathrm{Im}[\tilde B_y^*\hat L_z\tilde E_x-\tilde B_x^*\hat L_z\tilde E_y+\tilde B_z^*\tilde E_z]$.
- `total_angular_momentum_flux_zz(c)` — $d\Sigma_{zz}/d\omega + d\Lambda_{zz}/d\omega$.
- `energy_density(c)` — $du/d\omega = \varepsilon_0(|\tilde{\mathbf E}|^2+c^2|\tilde{\mathbf B}|^2)$.
- `energy_flux_z(c)` — $dP_z/d\omega = 2\varepsilon_0c^2\,\mathrm{Re}[\tilde E_x\tilde B_y^*-\tilde E_y\tilde B_x^*]$.
- `integrate_over_screen(quantity)` — composite trapezoidal surface integral, returns shape
  `(N_omega,)`. Rectangular screens integrate $dx\,dy$ (`numpy.trapezoid`, nested over $x$ then
  $y$); annular screens integrate $\rho\,d\rho\,d\phi$, including the explicit radial Jacobian
  weight `geometry.r_centers`.

## 4. Plotting: one folder per harmonic

`plotting.screen.generate_all_screen_observable_plots` plots 8 heatmaps per non-linear-Thomson
harmonic — `energy_density`, `energy_flux_z`, `total_angular_momentum_density_z`,
`total_angular_momentum_flux_zz`, `spin_angular_momentum_density_z`,
`spin_angular_momentum_flux_zz`, `orbital_angular_momentum_density_z`,
`orbital_angular_momentum_flux_zz` — into `run_dir / 'screen_observables_N_<N>_omega_<val>_omega0/'`,
mirroring `generate_all_screen_breakdown_plots`'s per-harmonic folder layout for the raw Faraday
tensor components. There is no aggregate, all-harmonics-in-one-figure plot for any screen
observable; every plot is per-harmonic. This replaced the original single top-level
`screen_emitted_intensity.png`/`screen_angular_momentum_flux.png` (and their generating functions
`plot_screen_emitted_intensity`/`plot_screen_angular_momentum_flux_density`, both removed) once all
observables needed plotting rather than just two.

## 5. `run.json`

`main.py` computes, per harmonic, the surface integrals via `ScreenResult.integrate_over_screen`:
the six named in the proposed doc (`int_dSz_domega`, `int_dLz_domega`, `int_dJz_domega`,
`int_dSigma_zz_domega`, `int_du_domega`, `int_dPz_domega`) plus `int_dLambda_zz_domega` (orbital
flux — added so every density has a matching flux integral; the proposed doc's own list omits it,
but `int_dSigma_zz_domega + int_dLambda_zz_domega` is the total angular-momentum flux matching
`int_dJz_domega = int_dSz_domega + int_dLz_domega`), plus the three flux/density ratios described
below, `N` (harmonic index), and `omega_au`. These are written into
`metadata['screen_observables']['rows']` in `run.json` only — there is no separate `run_log.txt`
(the earlier tab-separated-file design, `output.write_run_log`, was removed; don't re-add it).

## 6. Physical checks in `run.json`

`main.py` precomputes these ratios into each `run.json` row (`screen_observables` and `incident_laser_observables`):
- `energy_flux_density_ratio` (`int_dPz_domega / int_du_domega`), `spin_flux_density_ratio` (`int_dSigma_zz_domega / int_dSz_domega`), and `angular_momentum_flux_density_ratio` (`(int_dSigma_zz + int_dLambda_zz) / int_dJz_domega`). In the far field, each equals $\pm c$ (sign matching $\operatorname{sgn}(z_{\text{screen}})$).
- `spin_flux_energy_flux_ratio` (`int_dSigma_zz_domega / int_dPz_domega`) compared against theoretical spin-to-energy ratio $\epsilon / \omega$.
- `orbital_flux_energy_flux_ratio` (`int_dLambda_zz_domega / int_dPz_domega`) compared against theoretical orbital-to-energy ratio $\epsilon \cdot m / \omega$.

`tests/test_output.py::OutputTests::test_screen_observable_flux_density_ratio_equals_c` runs the
full `main.py` pipeline and checks, per harmonic, that each of these three ratios

is `+c` or `-c` (sign matching which side of the source `screen.z_screen` is on). This holds because far from the source every
field component shares the same phase velocity $c$ in vacuum, so any locally-conserved bilinear
quantity radiation carries satisfies flux $= c\,n_z\,$density. It is a *near-field* effect, not a
resolution artifact: the ratio is within ~1e-4 relative of $\pm c$ already at
`screen.z_screen` ~ $10^4$-$10^5\,\lambda$ (this repo's usual scale) and degrades quickly below
~10 $\lambda$ — see the "near-field caveat" note in `CLAUDE.md`.

## 7. On-axis behavior

Because $\hat L_z$ multiplies its finite-difference derivative by $x$ and $y$ before summing, both
`orbital_angular_momentum_density_z` and the $\hat L_z$ terms inside
`orbital_angular_momentum_flux_zz` vanish exactly at the screen-center pixel $x=y=0$ (tested with
an odd-pixel-count rectangular geometry so a grid point lands exactly on the origin). The
$\tilde B_z^*\tilde E_z$ term in $d\Lambda_{zz}/d\omega$ has no such prefactor and is **not**
expected to vanish there; neither is $d\Sigma_{zz}/d\omega$, since the spin quantities carry no
$x,y$ weighting at all.
