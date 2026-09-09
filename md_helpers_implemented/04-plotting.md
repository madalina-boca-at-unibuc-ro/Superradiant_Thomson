# Implemented: Plotting Package

`superradiant_thomson/plotting/laser.py` provides plotting functions for laser temporal dynamics, focal-plane spatial mode intensity, and 6-component electromagnetic field time series.
`superradiant_thomson/plotting/electron.py` provides 3D scatter plotting functions for initial electron positions and momenta distributions, and 4-panel component plots for relativistic 4-position $r^\mu(\tau)$, 4-velocity $u^\mu(\tau)$, and 4-acceleration $w^\mu(\tau)$ trajectories.

Shared axis presentation helpers live in `superradiant_thomson/plotting/style.py`. Physics classes have no Matplotlib dependency. Plotters return `(fig, ax)` or `(fig, axs)` and never show or save automatically. Callers (such as `main.py`) control figure display and file saving.

---

## 1. Laser Temporal Factor (`plot_temporal_factor`)

Plots the real, imaginary, or envelope of the complex laser temporal factor $f(t-z/c)$ at a fixed $z$.

### Interface & Signature

```python
from superradiant_thomson.plotting import plot_temporal_factor

fig, ax = plot_temporal_factor(pulse, time, z=0.0, time_unit='fs', component='real', ax=None)
```

### Key Properties
- **Inputs**: `time` (1D increasing array in atomic time units), `z` (float in atomic units).
- **Time units**: `'au'`, `'T'` (laser period), or registered fixed time units (e.g. `'fs'`, `'s'`).
- **Components**: `'real'`, `'imag'`, `'both'`, or `'envelope'`. Carrier plots include signed envelope dashed lines.
- **Output**: Returns `(fig, ax)`.

---

## 2. Focal-Plane LG Spatial Intensity Heatmap (`plot_lg_intensity`)

Plots the spatial scalar intensity diagnostic $|E_0 u_{pm}(\mathbf{r}) f(t, z)|^2$ in the focal plane ($z=0$) at the start of the flat-top plateau ($t = \text{wing\_factor} \times \sigma_l$, where $|f|=1$).

### Interface & Signature

```python
from superradiant_thomson.plotting import plot_lg_intensity

fig, ax = plot_lg_intensity(x, y, intensity, w_0=mode.w_0, time_in_periods=t_flat/period, ax=None)
```

### Key Properties
- **Inputs**: 1D spatial coordinate grids `x` and `y` in atomic units (default 401 samples per axis over $[-2 w_0, 2 w_0]$); 2D array `intensity` ordered `[y, x]` in atomic electric field squared ($a.u.$).
- **Axes & Aspect**: Horizontal $x/w_0$, vertical $y/w_0$, equal aspect ratio, linear colorbar (`inferno`, vmin=0).
- **Physical Quantity**: Scalar diagnostic $|E_0 u_{pm} f|^2$ in atomic field squared ($E_H^2$). Unscaled by spatial peak; the fundamental Gaussian $u_{00}$ peaks at $2 E_0^2$.
- **Output**: Returns `(fig, ax)`.

---

## 3. 6-Panel Time-Domain Electromagnetic Fields (`plot_laser_fields`)

Calculates and plots all 6 real paraxial electromagnetic field components ($E_x/c, E_y/c, E_z/c, B_x, B_y, B_z$) at a specified spatial evaluation point $\mathbf{r} = (x, y, z)$ over time. Electric field components are divided by $c$ on the plot to share similar magnitude scales with the magnetic field components $B$.

### Interface & Signature

```python
from superradiant_thomson.lg_mode import evaluate_laser_fields
from superradiant_thomson.plotting import plot_laser_fields

fields = evaluate_laser_fields(mode, amplitude, pulse, r_au, time)
fig, axs = plot_laser_fields(fields, time, r_w0=(0.0, 0.0, 0.0), pulse=pulse, time_unit='T')
```

### Key Properties
- **Layout**: $2 \times 3$ grid of subplots (top row: $E_x/c, E_y/c, E_z/c$; bottom row: $B_x, B_y, B_z$).
- **Suptitle**: Displays evaluation point $\mathbf{r} = (x, y, z)$ in units of $w_0$.
- **Output**: Returns `(fig, axs)`.

---

## 4. Initial Electron Ensemble 3D Scatter Plots (`plot_electron_initial_distribution`)

Plots 3D spatial scatter plots of initial electron positions $(x, y, z)$ in units of $w_0$ (or $a.u.$) and 3D momentum scatter plots of initial momenta $(p_x, p_y, p_z)$ in units of $m_e c$ (or $a.u.$).

### Interface & Signature

```python
from superradiant_thomson.plotting import (
    plot_electron_initial_positions,
    plot_electron_initial_momenta,
    plot_electron_initial_distribution,
)

# 2-panel 3D distribution plot
fig, (ax_pos, ax_mom) = plot_electron_initial_distribution(r0, u0, w_0=mode.w_0, c=units.c)

# Standalone 3D position plot
fig_pos, ax_pos = plot_electron_initial_positions(r0, w_0=mode.w_0)

# Standalone 3D momentum plot
fig_mom, ax_mom = plot_electron_initial_momenta(u0, c=units.c)
```

### Key Properties
- **Inputs**: Initial 4-position $r_0$ array of shape `(N, 4)` and initial 4-velocity $u_0$ array of shape `(N, 4)`.
- **Position Units**: Coordinates $(x, y, z)$ scaled to $w_0$ when specified.
- **Momentum Units**: 3-momentum $(p_x, p_y, p_z)$ scaled to $m_e c$ when $c$ is specified.
- **Marker Sizing**: Dynamically scales down marker size ($s=2$ for $N \ge 1000$) to prevent clutter in large ensemble scatter plots.
- **Output**: Returns `(fig, (ax_pos, ax_mom))`, `(fig, ax_pos)`, or `(fig, ax_mom)`.

---

## 5. Electron 4-Vector Trajectory Component Plots (`plot_electron_ensemble_trajectories`)

Plots the 4 components of 4-position $r^\mu(\tau) = (ct, x, y, z)$, 4-velocity $u^\mu(\tau) = (\gamma c, p_x, p_y, p_z)$, and 4-acceleration $w^\mu(\tau)$ as functions of proper time $\tau$ for up to $K = \min(\text{max\_electrons}, N)$ electrons in $2 \times 2$ panel figures.

### Interface & Signature

```python
from superradiant_thomson.plotting import (
    plot_electron_position_trajectories,
    plot_electron_velocity_trajectories,
    plot_electron_acceleration_trajectories,
    plot_electron_ensemble_trajectories,
)

# 3 separate 4-panel trajectory figures
fig_pos, fig_vel, fig_acc = plot_electron_ensemble_trajectories(
    electron, max_electrons=10, tau_unit='T', w_0=mode.w_0, c=units.c, pulse=pulse
)
```

### Key Properties
- **Layout**: $2 \times 2$ grid of subplots for each 4-vector (Panel 0: component 0, Panel 1: component 1, Panel 2: component 2, Panel 3: component 3).
- **Proper Time Scale**: Displayed in units of laser periods $T = 2\pi/\omega$ when `tau_unit='T'`.
- **Multi-Electron Overlay**: Plots up to $K = 10$ distinct colored trajectory curves with automatic figure legends.
- **Output**: Returns tuple of figures `(fig_pos, fig_vel, fig_acc)`.

---

## 6. Entry Point & Saved Run Outputs

Running the entry point:
```sh
.venv/bin/python main.py
```
generates a dedicated timestamped run output directory containing:
- `temporal_factor.png` & `temporal_factor.npz`: Temporal profile diagnostic.
- `lg_intensity.png` & `lg_intensity.npz`: Spatial focal-plane heatmap.
- `laser_fields.png` & `laser_fields.npz`: 6-panel time-domain EM fields plot.
- `electron_initial_distribution.png`: 2-panel 3D scatter plots of initial electron positions and momenta.
- `electron_position_trajectories.png`: 4-panel trajectory component plot for $r^\mu(\tau)$.
- `electron_velocity_trajectories.png`: 4-panel trajectory component plot for $u^\mu(\tau)$.
- `electron_acceleration_trajectories.png`: 4-panel trajectory component plot for $w^\mu(\tau)$.
- `electron_trajectory.npz`: Full multi-electron relativistic trajectories.
- `run.json`: Execution metadata including parameter resolution, array dimensions, units, and evaluation coordinates.

The CLI flag `--no-show` suppresses Matplotlib GUI plot display.

---

## 7. Verification

All plotting and field evaluation routines are verified via unit tests in `tests/`:
- `test_laser.py`: Temporal factor plotting and scale conversions.
- `test_lg_mode.py`: Spatial mode calculations and derivatives.
- `test_laser_fields.py`: 6-panel field calculations.
- `test_electron_plots.py`: 3D scatter plotting of initial electron positions/momenta and 4-panel component plots for 4-position, 4-velocity, and 4-acceleration trajectories across 1D, 2D, and 3D trajectory arrays.
