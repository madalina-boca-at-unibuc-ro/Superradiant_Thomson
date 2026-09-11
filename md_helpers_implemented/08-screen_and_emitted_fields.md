# Implemented: 2D Observation Screen, FT Faraday Tensor Evaluators, and Parallel Reduction

Module: `superradiant_thomson/screen.py`. Tests: `tests/test_screen.py`.

Provides the `ScreenGeometry` dataclass, parameter schema `screen_schema()`, `ScreenResult` dataclass, single-electron FT Faraday tensor evaluator `_compute_single_electron_screen_field()`, legacy pre-computed trajectory evaluator `compute_screen_emitted_field()`, and fused streaming parallel solver `compute_screen_emitted_field_from_laser_and_bunch()`.

---

## 1. 2D Observation Screen Geometry (`ScreenGeometry`)

The 2D observation screen is centered on the positive $Oz$ laser axis at longitudinal position $z = Z_{\text{screen}}$. The screen geometry is configured via `'screen.shape'`: `'rectangular'` (default) or `'annular'`.

### A. Rectangular Screen (`screen.shape = 'rectangular'`)
Defined by width $W$ along $Ox$, height $H_s$ along $Oy$, and pixel resolution $N_x \times N_y$:
$$x \in [-W/2, W/2], \quad y \in [-H_s/2, H_s/2]$$

$$\Delta x = \frac{W}{N_x}, \quad \Delta y = \frac{H_s}{N_y}$$

$$x_i = -\frac{W}{2} + \left(i + \frac{1}{2}\right)\Delta x, \quad y_j = -\frac{H_s}{2} + \left(j + \frac{1}{2}\right)\Delta y$$

### B. Annular / Sector Screen (`screen.shape = 'annular'`)
Defined by inner radius $R_{\min}$, outer radius $R_{\max}$, radial rings $N_R$, azimuthal bounds $\Phi_{\min}, \Phi_{\max}$ (input in units of $\pi$, where $1\text{ pi} = \pi \text{ rad}$), and sector count $N_\Phi$:

#### Equal Surface Area Radial Discretization:
To ensure uniform surface area coverage per cell ($\Delta A = \frac{1}{2}\Delta(R^2)\Delta\phi = \text{const}$), radial cell centers $r_i$ and cell edges $r_{\text{edge}, k}$ are sampled uniformly in $r^2$:

$$r_i = \sqrt{R_{\min}^2 + \left(i + \frac{1}{2}\right) \frac{R_{\max}^2 - R_{\min}^2}{N_R}}, \quad i = 0, \dots, N_R - 1$$

$$r_{\text{edge}, k} = \sqrt{R_{\min}^2 + k \frac{R_{\max}^2 - R_{\min}^2}{N_R}}, \quad k = 0, \dots, N_R$$

#### Azimuthal Sector Discretization:
$$\phi_j = \Phi_{\min} + \left(j + \frac{1}{2}\right) \Delta\phi, \quad \Delta\phi = \frac{\Phi_{\max} - \Phi_{\min}}{N_\Phi}, \quad j = 0, \dots, N_\Phi - 1$$

#### 2D Cartesian Evaluation Coordinates:
For each cell $(j, i)$ on the screen:
$$x_{j, i} = r_i \cos\phi_j, \quad y_{j, i} = r_i \sin\phi_j$$

Evaluation coordinates are stored in 2D meshgrids `grid_x` and `grid_y` of shape `(Ny, Nx)` (where $N_y = N_\Phi$ and $N_x = N_R$ for annular screens).

### Non-Linear Thomson Frequency Grid:
Harmonic frequencies $\omega_N$ for integer harmonic orders $N \in [N_{\min}, N_{\max}]$ in Hartree atomic units:

$$\omega_N = N \omega_0 \frac{n_L \cdot q}{n_s \cdot q}$$

where $q = p + (m c)^2 \frac{a_0^2}{4 (n_L \cdot p)} n_L$ is the dressed electron 4-momentum ($n_L \cdot p = p^0 - p^z$), $n_L = (1, 0, 0, 1)$, and $n_s = (1, 0, 0, \operatorname{sgn}(Z_{\text{screen}}))$. Wavenumber $k_N = \omega_N / c$.

---

## 2. Contravariant Faraday Tensor Storage (`ScreenResult`)

Each screen cell $(j, i)$ stores the 6 upper-triangular components of the complex Fourier-transformed Faraday tensor $F^{\mu\nu}(\omega_N, \mathbf{x}_{ji})$:

$$(F^{01}, F^{02}, F^{03}, F^{12}, F^{13}, F^{23}) \longleftrightarrow (\text{F01}, \text{F02}, \text{F03}, \text{F12}, \text{F13}, \text{F23})$$

### Component Array Order:
- `F_l`: shape `(N_omega, Ny, Nx, 6)` — long-distance radiation component ($1/|R_0|$ scaling).
- `F_s`: shape `(N_omega, Ny, Nx, 6)` — short-distance velocity component ($1/|R_0|^2$ scaling).
- `F_b`: shape `(N_omega, Ny, Nx, 6)` — finite integration interval endpoint boundary term.
- `F_total = F_l + F_s + F_b`: total Faraday tensor.
- `intensity`: 2D/3D scalar field norm square sum $\sum_{\mu < \nu} |F_{\text{total}}^{\mu\nu}|^2$.

---

## 3. Emitted Field Calculation Methods (`screen.method`)

Configurable via parameter `'screen.method'`:

### Form 2 ("Simplified" - `method='simplified'`, Default):
Derived via integration-by-parts on Jackson's form:

$$F_l^{\alpha\beta}(\omega, \mathbf{x}_0) = \frac{q}{2\pi c^2} \int e^{i k \Phi(\tau)} \left( \frac{-i k}{R_0} \right) \frac{n_{R_0}^\alpha u^\beta - n_{R_0}^\beta u^\alpha}{u \cdot n_{R_0}} d\tau$$

$$F_s^{\alpha\beta}(\omega, \mathbf{x}_0) = \frac{q}{2\pi c^2} \int e^{i k \Phi(\tau)} \left( \frac{-\mathbf{n}_{R_0} \cdot \mathbf{u}}{R_0^2 (u \cdot n_{R_0})} \right) (n_{R_0}^\alpha u^\beta - n_{R_0}^\beta u^\alpha) d\tau$$

$$F_b^{\alpha\beta}(\omega, \mathbf{x}_0) = \frac{q}{2\pi c^2} \left[ \frac{e^{i k \Phi(\tau)} (n_{R_0}^\alpha u^\beta - n_{R_0}^\beta u^\alpha)}{R_0 (u \cdot n_{R_0})} \right]_{\tau=0}^{\tau=\tau_{\max}}$$

### Form 1 ("Direct" - `method='direct'`):
Direct transform of Liénard–Wiechert field without integration by parts. When `method='direct'`, invalid methods raise an explicit `ValueError`.

---

## 4. Real Wall-Clock Performance Timing & Metrics

High-precision wall-clock timing is measured across parallel worker execution using `time.perf_counter()`:

$$\text{Total Wall-Clock Time } T \quad [\text{seconds}]$$

$$\text{Time per Electron} = \frac{T}{N_e} \quad [\text{ms/electron}]$$

$$\text{Time per Electron per Screen Point} = \frac{T}{N_e \times N_{\text{points}}} \quad [\mu\text{s/electron/point}]$$

where $N_{\text{points}} = N_x \times N_y$ (or $N_R \times N_\Phi$).

### Exposed Dataclass Properties & Outputs:
- `ScreenResult.wall_time_seconds`: Total solver time $T$.
- `ScreenResult.time_per_electron_seconds`: Average time per electron $T / N_e$.
- `ScreenResult.time_per_electron_per_point_seconds`: Average time per electron per pixel $T / (N_e N_{\text{points}})$.
- **Console Log**: Prints total time ($\text{s}$ and $\text{min}$), time per electron ($\text{ms/electron}$), and time per electron per screen point ($\mu\text{s/electron/point}$).
- **Metadata Export**: Recorded in `run.json` under `metadata['screen_emitted_field']`.

---

## 5. Parameter Schema (`screen_schema()`)

- `screen.shape`: `'rectangular'` or `'annular'` (default `'rectangular'`).
- `screen.z_screen`: Screen z-position (default $25000 \lambda$).
- `screen.width`, `screen.height`: Rectangular screen dimensions $W, H_s$ (default $400 \lambda$).
- `screen.Nx`, `screen.Ny`: Rectangular pixel grid resolution (default $32 \times 32$).
- `screen.R_min`, `screen.R_max`: Annular inner/outer radii (default $0$ to $200 \lambda$).
- `screen.N_R`: Annular radial ring count (default $32$).
- `screen.Phi_min`, `screen.Phi_max`: Sector angle bounds in units of $\pi$ (default $0$ to $2\pi$).
- `screen.N_Phi`: Sector count (default $32$).
- `screen.N_min`, `screen.N_max`: Harmonic order range (default $1$ to $3$).
- `screen.method`: `'simplified'` or `'direct'` (default `'simplified'`).

---

## 6. Curvilinear Heatmap Plotting

Visualized in `superradiant_thomson/plotting/screen.py` via `_draw_2d_heatmap()`:
- **Annular Screens**: Rendered on Cartesian $(x,y)$ axes using `ax.pcolormesh(X_corners, Y_corners, data_2d, shading='flat')` with equal aspect ratio. `X_corners` and `Y_corners` arrays of shape `(N_Phi + 1, N_R + 1)` define true curvilinear polar cell boundaries without interpolation artifacts.
- **Rectangular Screens**: Rendered using `ax.imshow()` with exact extent boundaries.

---

## 7. Verification

Verified by unit tests in `tests/test_screen.py`:
- `test_annular_geometry_grid_and_corners`: Verifies equal $r^2$ surface area binning, center coordinates, and corner boundary grids.
- `test_screen_parameters_resolution_in_w0_units`: Resolving screen parameters in units of $w_0$ and $w_0$ aliases.
- `test_form1_vs_form2_agreement`: Form 1 ("direct") vs Form 2 ("simplified") tensor equality.
- `test_charge_reversal_antisymmetry`: Reversing charge $q \to -q$ inverts Faraday tensor $F \to -F$.
- `test_streaming_parallel_solver_and_core_count_invariance`: Fused parallel streaming solver produces 100% identical initial conditions and Faraday tensors regardless of worker count.

