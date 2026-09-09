# Implemented: 2D Observation Screen, FT Faraday Tensor Evaluators, and Parallel Reduction

Module: `superradiant_thomson/screen.py`. Tests: `tests/test_screen.py`.

Provides the `ScreenGeometry` dataclass, parameter schema `screen_schema()`, `ScreenResult` dataclass, single-electron FT Faraday tensor evaluator `_compute_single_electron_screen_field()`, legacy pre-computed trajectory evaluator `compute_screen_emitted_field()`, and fused streaming parallel solver `compute_screen_emitted_field_from_laser_and_bunch()`.

---

## 1. 2D Observation Screen Geometry (`ScreenGeometry`)

The rectangular observation screen is centered on the positive $Oz$ laser axis at $z = Z_{\text{screen}}$ with width $W$ along $Ox$ and height $H_s$ along $Oy$:

$$x \in [-W/2, W/2], \quad y \in [-H_s/2, H_s/2]$$

### Pixel Centers and Coordinate Grids:
For pixel counts $N_x, N_y$ and grid indices $i=0,\dots,N_x-1$, $j=0,\dots,N_y-1$:

$$\Delta x = \frac{W}{N_x}, \quad \Delta y = \frac{H_s}{N_y}$$

$$x_i = -\frac{W}{2} + \left(i + \frac{1}{2}\right)\Delta x, \quad y_j = -\frac{H_s}{2} + \left(j + \frac{1}{2}\right)\Delta y$$

Evaluation coordinates $\mathbf{x}_{ji} = (x_i, y_j, Z_{\text{screen}})$ are stored in 2D array order `[y, x]`.

### Non-Linear Thomson Frequency Grid:
Harmonic frequencies $\omega_N$ for integer harmonic orders $N \in [N_{\min}, N_{\max}]$ in Hartree atomic units:

$$\omega_N = N \omega_0 \frac{n_L \cdot q}{n_s \cdot q}$$

where $q = p + (m c)\frac{a_0^2}{4} n_L$ is the dressed electron 4-momentum, $n_L = (1, 0, 0, 1)$, and $n_s = (1, 0, 0, \operatorname{sgn}(Z_{\text{screen}}))$. Wavenumber $k_N = \omega_N / c$.

---

## 2. Contravariant Faraday Tensor Storage

Each screen pixel $(j, i)$ stores the 6 upper-triangular components of the complex Fourier-transformed Faraday tensor $F^{\mu\nu}(\omega, \mathbf{x}_{ji})$:

$$(F^{01}, F^{02}, F^{03}, F^{12}, F^{13}, F^{23}) \longleftrightarrow (\text{F01}, \text{F02}, \text{F03}, \text{F12}, \text{F13}, \text{F23})$$

### Component Array Order:
- `F_l`: shape `(N_omega, Ny, Nx, 6)` — long-distance radiation component ($1/|R_0|$ scaling).
- `F_s`: shape `(N_omega, Ny, Nx, 6)` — short-distance velocity component ($1/|R_0|^2$ scaling).
- `F_b`: shape `(N_omega, Ny, Nx, 6)` — finite integration interval endpoint boundary term.
- `F_total = F_l + F_s + F_b`: total Faraday tensor.
- `intensity`: 2D/3D scalar field norm square sum $\sum_{\mu < \nu} |F_{\text{total}}^{\mu\nu}|^2$.

---

## 3. Fourier-Transformed Emitted Field Evaluators

Evaluates the Fourier transform of the emitted radiation from electron trajectory $r(\tau), u(\tau), w(\tau)$:

### Phase & Displacement:
For observation point $\mathbf{x}_0$ and source proper time $\tau$:

$$\mathbf{R}_0(\tau) = \mathbf{x}_0 - \mathbf{r}_0(\tau), \quad R_0 = |\mathbf{R}_0|, \quad \mathbf{n}_{R_0} = \mathbf{R}_0 / R_0$$

$$u \cdot n_{R_0} = u^0 - \mathbf{u} \cdot \mathbf{n}_{R_0}, \quad \Phi(\tau) = r^0(\tau) + R_0(\tau)$$

### Form 2 ("Simplified" - `method='simplified'`, Default):
Derived via integration-by-parts on Jackson's form:

$$F_l^{\alpha\beta}(\omega, \mathbf{x}_0) = \frac{q}{2\pi c^2} \int e^{i k \Phi(\tau)} \left( \frac{-i k}{R_0} \right) \frac{n_{R_0}^\alpha u^\beta - n_{R_0}^\beta u^\alpha}{u \cdot n_{R_0}} d\tau$$

$$F_s^{\alpha\beta}(\omega, \mathbf{x}_0) = \frac{q}{2\pi c^2} \int e^{i k \Phi(\tau)} \left( \frac{-\mathbf{n}_{R_0} \cdot \mathbf{u}}{R_0^2 (u \cdot n_{R_0})} \right) (n_{R_0}^\alpha u^\beta - n_{R_0}^\beta u^\alpha) d\tau$$

$$F_b^{\alpha\beta}(\omega, \mathbf{x}_0) = \frac{q}{2\pi c^2} \left[ \frac{e^{i k \Phi(\tau)} (n_{R_0}^\alpha u^\beta - n_{R_0}^\beta u^\alpha)}{R_0 (u \cdot n_{R_0})} \right]_{\tau=0}^{\tau=\tau_{\max}}$$

### Form 1 ("Direct" - `method='direct'`):
Direct transform of Liénard–Wiechert field without integration by parts.

---

## 4. Fused Streaming Parallel Solver (`compute_screen_emitted_field_from_laser_and_bunch`)

Fuses initial condition generation, relativistic ODE trajectory integration, and screen Faraday tensor evaluation directly inside parallel worker processes (`ProcessPoolExecutor`):

1. **Partitioning**: Electron indices $0 \dots N_e-1$ are split into chunks across `max_workers`.
2. **Worker Streaming Execution**: Each worker process generates initial conditions for its chunk using deterministic `SeedSequence(seed).spawn(N)` child seeds, solves ODE trajectories on the fly, evaluates single-electron FT Faraday tensors on the 2D screen, and accumulates partial tensors.
3. **RAM Optimization**: Trajectory arrays are integrated and evaluated for radiation point-by-point or electron-by-electron, reducing memory footprint from $O(N_e \times N_\tau)$ down to $O(N_{\text{chunk}} \times N_\tau)$.
4. **10-Electron Trajectory Sample**: Worker 0 retains full trajectories ($r, u, w$) for electron indices $0 \dots 9$, returning `sample_electron` for detailed 4-panel trajectory component plotting and mass-shell diagnostics.
5. **Initial Conditions Gathering**: Workers return initial 4-position and 4-momentum vectors $(r_{0,i}, u_{0,i})$ for all $N_e$ electrons ($\sim 64\text{ bytes/electron}$), enabling exact 3D ensemble scatter plotting (`plot_electron_initial_distribution`).
6. **Progress Indicator**: Only the first worker chunk (`k == 0` in the index-chunk partition) is given a `progress=True` flag; that worker prints `Electron {k+1}/{n_chunk} (progress worker chunk)` to stdout before solving each electron's trajectory in its chunk. Since chunks are equal-sized (`np.array_split`), this single stream is a proxy for overall progress without interleaving output from all workers.

---

## 5. Parameter Schema (`screen_schema()`)

- `screen.z_screen`: Screen z-position (default $25000 \lambda$).
- `screen.width`, `screen.height`: Screen dimensions $W, H_s$ (default $400 \lambda$).
- `screen.Nx`, `screen.Ny`: Pixel grid resolution (default $32 \times 32$).
- `screen.N_min`: Minimum harmonic order $N_{\min}$ (default $1$).
- `screen.N_max`: Maximum harmonic order $N_{\max}$ (default $3$).

---

## 6. Verification

Verified by unit tests in `tests/test_screen.py`:
- `test_geometry_grid_properties`: Screen pixel coordinate grids and spacing.
- `test_form1_vs_form2_agreement`: Form 1 ("direct") vs Form 2 ("simplified") tensor equality.
- `test_charge_reversal_antisymmetry`: Reversing charge $q \to -q$ inverts Faraday tensor $F \to -F$.
- `test_serial_vs_parallel_equivalence`: Serial (`max_workers=1`) and parallel (`max_workers=2`) reductions match to machine precision.
- `test_streaming_parallel_solver_and_core_count_invariance`: Fused parallel streaming solver produces 100% identical initial conditions and Faraday tensors regardless of worker count.
