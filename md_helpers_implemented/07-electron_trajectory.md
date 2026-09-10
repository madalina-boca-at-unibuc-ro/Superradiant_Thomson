# Implemented: Electron Relativistic Trajectory Solver & Random Distribution Engine

Module: `superradiant_thomson/electron.py`. Tests: `tests/test_electron.py`.

Provides the `Electron` trajectory dataclass, initial distribution generator `generate_electron_initial_conditions()`, input parameter schema `electron_schema()`, single-electron ODE solver `solve_single_electron_trajectory()`, and relativistic ensemble ODE solver `solve_electron_ensemble()`.

---

## 1. Physical Equations of Motion

The relativistic equations of motion for an electron of mass $m=1$ and charge $q=-1$ in a Laguerre-Gaussian laser field are integrated in proper time $\tau$:

$$\frac{d r^\mu}{d\tau} = u^\mu, \quad \frac{d u^\mu}{d\tau} = w^\mu = \frac{q}{m^2} F^{\mu\nu}(r) u_\nu$$

where:
- $r^\mu(\tau) = (c t(\tau), x(\tau), y(\tau), z(\tau))$ is the 4-position in Hartree atomic units.
- $u^\mu(\tau) = (\gamma c, \gamma v_x, \gamma v_y, \gamma v_z) = (\gamma c, p_x, p_y, p_z)$ is the 4-velocity.
- $w^\mu(\tau) = \frac{d u^\mu}{d\tau}$ is the 4-acceleration.
- $F^{\mu\nu}$ is the contravariant Faraday tensor of the Laguerre-Gaussian laser pulse.

---

## 2. Proper Time Sampling Grid & Relativistic Doppler Adjustment

The trajectory is solved over proper time $\tau \in [0, D]$, where $D = 2 L_{\text{wing}} + P$ is the total laser pulse duration.

### A. Doppler Factor Calculation (`compute_doppler_factor`)
When an electron beam is in motion with initial mean 3-momentum $\mathbf{p} = (p_x, p_y, p_z) \neq 0$, the laser field phase experienced along the electron trajectory evolves at a rate dependent on initial momentum:

$$\phi(\tau) = \omega_0 \left( t(\tau) - \frac{z(\tau)}{c} \right) \implies \frac{d\phi}{d\tau} = \frac{\omega_0}{m c} \left( p^0 - p_z \right)$$

where $p^0 = \sqrt{(mc)^2 + p_x^2 + p_y^2 + p_z^2}$. The relativistic Doppler factor $\mathcal{D}$ is:

$$\mathcal{D} = \frac{p^0 - p_z}{m c}$$

### B. Numerical Step Size Adjustment (`compute_doppler_adjusted_tau_eval`)
For an electron at rest ($\mathbf{p} = 0$), the step size $dt_r$ is determined by parameter `electron.NT` (points per laser period $T = 2\pi/\omega_0$):

$$dt_r = \frac{T}{\text{NT}} = \frac{2\pi}{\omega_0 \, \text{NT}}$$

For moving electrons ($\mathbf{p} \neq 0$), to ensure $\text{NT}$ sampling points per Doppler-shifted laser period in the electron frame, the step size $dt_m$ is scaled by the Doppler factor:

$$dt_m = \frac{dt_r}{\mathcal{D}} = \frac{dt_r}{\frac{p^0 - p_z}{m c}}$$

- **Head-on Collision ($p_z < 0$)**: $\mathcal{D} > 1.0 \implies dt_m < dt_r$. The step size shrinks to resolve fast blue-shifted phase oscillations.
- **Co-propagating ($p_z > 0$)**: $\mathcal{D} < 1.0 \implies dt_m > dt_r$. The step size expands according to the red-shifted phase evolution.
- **At Rest ($\mathbf{p} = 0$)**: $\mathcal{D} = 1.0 \implies dt_m = dt_r$.

The total number of proper-time sampling points $N_\tau$ is:

$$n_{\text{periods, doppler}} = \frac{D}{T} \times \mathcal{D}, \quad N_\tau = \max\left(2, \lfloor n_{\text{periods, doppler}} \times \text{NT} \rfloor + 1\right)$$

---

## 3. Core-Count-Invariant Initial Distribution Engine (`generate_electron_initial_conditions`)

Generates initial 4-positions $r_0$ and 4-velocities $u_0$ for an ensemble of $N$ electrons (or a specific array of electron indices) using reproducible random generator seed `electron.seed`:

### Deterministic Per-Index Seeding:
- Uses `np.random.SeedSequence(seed).spawn(N)` to derive independent child seeds for each electron index $i \in [0, N-1]$.
- Electron index $i$ has identical initial conditions $(r_{0,i}, u_{0,i})$ regardless of whether execution is serial or split across $P$ parallel CPU workers.

### Spatial Cylinder Distribution (aligned along $Oz$):
- Centered at $(x_0, y_0, z_0)$ (`electron.x_0`, `electron.y_0`, `electron.z_0`).
- Transverse radius $R_{\text{beam}}$ (`electron.R_beam`): sampled uniformly in disk $r = R_{\text{beam}}\sqrt{U_1}$, $\theta = 2\pi U_2$.
- Longitudinal height $h_{\text{beam}}$ (`electron.h_beam`): sampled uniformly $z \in [z_0 - h_{\text{beam}}/2, z_0 + h_{\text{beam}}/2]$.
- Zero-width limit ($R_{\text{beam}}=0, h_{\text{beam}}=0$) generates exact fixed position $(x_0, y_0, z_0)$.

### Gaussian 3-Momentum Distribution:
- Mean 3-momentum $\mathbf{p}_{\text{beam}} = (\bar{p}_x, \bar{p}_y, \bar{p}_z)$ (`electron.px_beam`, `py_beam`, `pz_beam`).
- Standard deviations $\boldsymbol{\sigma}_p = (\sigma_{px}, \sigma_{py}, \sigma_{pz})$ (`electron.sigma_px_beam`, `sigma_py_beam`, `sigma_pz_beam`).
- Zero-width limit ($\sigma_p = 0$) generates exact fixed 3-momentum.
- Exact Relativistic Conversion: $u_0^\mu = (\gamma c, p_x, p_y, p_z)$ with $\gamma = \sqrt{1 + |\mathbf{p}|^2 / c^2}$, guaranteeing $u_0 \cdot u_0 = c^2$.

---

## 4. Data Structure (`Electron`) & Streaming Memory Optimization

```python
from superradiant_thomson.electron import Electron

electron = Electron(tau=tau_eval, r=r_eval, u=u_eval, w=w_eval, q=-1.0, m=1.0)
```

### Fields:
- `tau`: 1D array of proper-time values, shape `(N_tau,)`.
- `r`: 4-position array $(ct, x, y, z)$, shape `(N_tau, 4)` for single electron or `(N_electrons, N_tau, 4)` for ensemble.
- `u`: 4-velocity array $(\gamma c, p_x, p_y, p_z)$, shape `(N_tau, 4)` or `(N_electrons, N_tau, 4)`.
- `w`: 4-acceleration array $(du/dtau)$, shape `(N_tau, 4)` or `(N_electrons, N_tau, 4)`.
- `q`: charge in atomic units (default `-1.0`).
- `m`: rest mass in atomic units (default `1.0`).

### Streaming Parallel Worker Pipeline:
- Trajectory integration is fused directly into the parallel screen radiation workers.
- Trajectories are integrated and evaluated for screen emitted fields on the fly, reducing memory footprint from $O(N_e \times N_\tau)$ down to $O(N_{\text{chunk}} \times N_\tau)$.
- **10-Electron Trajectory Sample**: Worker 0 retains full trajectories ($r, u, w$) for electron indices $0 \dots 9$, returning `sample_electron` for detailed 4-panel trajectory component plotting and mass-shell diagnostics.
- **Full Initial Conditions Gathering**: All workers return initial 4-position and 4-momentum vectors $(r_{0,i}, u_{0,i})$ for all $N_e$ electrons ($\sim 64\text{ bytes/electron}$), enabling exact 3D ensemble scatter plotting (`plot_electron_initial_distribution`).

### Diagnostic Methods:
- `mass_shell_residual(c)`: computes $\frac{u \cdot u - c^2}{c^2}$ over all electrons and trajectory points.
- `acceleration_orthogonality_residual(c)`: computes $\frac{u \cdot w}{c |w|}$ over all electrons and trajectory points.

---

## 5. Parameter Schema (`electron_schema()`)

- `electron.N`: Positive integer count of electrons (default `1`).
- `electron.seed`: Nonnegative integer random seed or `None` (default `42`).
- `electron.NT`: Positive integer points per period (default `100`).
- `electron.x_0`, `electron.y_0`, `electron.z_0`: Beam center coordinates (default `0.0 w_0`).
- `electron.R_beam`, `electron.h_beam`: Cylinder radius and height (default `0.0 w_0`).
- `electron.px_beam`, `electron.py_beam`, `electron.pz_beam`: Mean 3-momentum (default `0.0 c`).
- `electron.sigma_px_beam`, `electron.sigma_py_beam`, `electron.sigma_pz_beam`: 3-momentum std. dev. (default `0.0 c`).

---

## 6. Main Script Integration and Output Artifacts

Integrated in `main.py`:
- Solves trajectories and screen radiation via streaming parallel pipeline `compute_screen_emitted_field_from_laser_and_bunch(...)`.
- Saves trajectory array output (sample trajectories $r, u, w$ and full initial conditions $r_0, u_0$) to `electron_trajectory.npz`.
- Appends trajectory metadata (`N`, `seed`, `NT`, residual metrics) to `run.json`.

---

## 7. Verification

Verified by unit tests in `tests/test_electron.py` and `tests/test_screen.py`:
- Dataclass validation for 2D and 3D array shapes.
- Spatial cylinder distribution bounds ($r \le R_{\text{beam}}$, $|z - z_0| \le h_{\text{beam}}/2$).
- Gaussian momentum distribution means, standard deviations, and exact zero-width limit.
- Core-count-invariant PRNG seed reproducibility ($P=1$ vs $P=2$).
- Free-particle trajectory ($a_0 = 0$).
- Laser field trajectory conservation ($a_0 = 1.0$):
  - Mass shell conservation $u \cdot u = c^2$ to tolerance $< 10^{-7}$.
  - 4-acceleration orthogonality $u \cdot w = 0$ to tolerance $< 10^{-6}$.
- Multi-electron ensemble trajectory solver ($N > 1$).

