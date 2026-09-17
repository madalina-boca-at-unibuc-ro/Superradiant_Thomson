# Implemented: Observation Screen Geometry

This document provides a comprehensive specification and reference for the 2D observation screen geometry implemented in the `superradiant_thomson` codebase (`superradiant_thomson/screen.py`, `superradiant_thomson/plotting/screen.py`, and `tests/test_screen.py`).

---

## 1. Global Coordinate System and Conventions

The simulation models relativistic electrons interacting with an intense Laguerre–Gaussian (LG) laser pulse and calculates the emitted radiation fields on a transverse observation screen.

### Coordinate Axes and Laser Propagation
- **Propagation Axis ($Oz$):** The laser pulse propagates along the positive longitudinal axis $\hat{\mathbf{z}} = (0, 0, 1)$.
- **Transverse Plane ($Oxy$):** The coordinates $x$ and $y$ span the transverse plane perpendicular to the laser propagation direction.
- **Screen Plane:** The observation screen is a planar surface orthogonal to the laser propagation axis, positioned at a fixed longitudinal coordinate:
  $$z = Z_{\text{screen}}$$
- **Screen Center:** The screen is centered on the optical axis $(x=0, y=0, z=Z_{\text{screen}})$.
- **Scattering Regimes:**
  - **Forward scattering:** $Z_{\text{screen}} > 0$ (radiation detected downstream of the interaction).
  - **Backward scattering:** $Z_{\text{screen}} < 0$ (radiation detected upstream of the interaction, reflecting back towards the laser source).

All physical quantities internally use **Hartree atomic units** ($\hbar = m_e = e = 4\pi\epsilon_0 = 1$, $c = 1/\alpha \approx 137.036$), as defined in `superradiant_thomson/parameters.py`.

---

## 2. Screen Shapes and Configurations

The project supports two distinct screen shapes configured by the parameter `'screen.shape'`:
1. **Rectangular Screen (`'rectangular'`)** — standard Cartesian grid suitable for broad planar field distributions and rectangular detectors.
2. **Annular / Sector Screen (`'annular'`)** — polar grid with equal surface area radial binning and azimuthal angular sectoring, ideal for analyzing cylindrical vortex modes, orbital angular momentum (OAM), and azimuthally symmetric Thomson scattering.

### Parameter Schema Summary

Screen parameters are resolved through `screen_schema()` in `superradiant_thomson/screen.py`:

| Parameter | Type / Dimension | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `screen.shape` | `str` | `'rectangular'` | Screen geometry: `'rectangular'` or `'annular'` |
| `screen.z_screen` | `LENGTH` | $25000.0\,\lambda$ | Longitudinal position $Z_{\text{screen}}$ of the screen |
| `screen.width` | `LENGTH` | $400.0\,\lambda$ | Width $W$ along $Ox$ (rectangular) |
| `screen.height` | `LENGTH` | $400.0\,\lambda$ | Height $H_s$ along $Oy$ (rectangular) |
| `screen.Nx` | `DIMENSIONLESS` | $32$ | Number of pixel cells $N_x$ along $Ox$ (rectangular) |
| `screen.Ny` | `DIMENSIONLESS` | $32$ | Number of pixel cells $N_y$ along $Oy$ (rectangular) |
| `screen.R_min` | `LENGTH` | $0.0\,\lambda$ | Inner radius $R_{\min} \ge 0$ (annular) |
| `screen.R_max` | `LENGTH` | $200.0\,\lambda$ | Outer radius $R_{\max} > R_{\min}$ (annular) |
| `screen.N_R` | `DIMENSIONLESS` | $32$ | Number of radial rings $N_R$ (annular) |
| `screen.Phi_min` | `DIMENSIONLESS` | $0.0\,\pi$ | Minimum azimuthal angle $\Phi_{\min}$ (annular) |
| `screen.Phi_max` | `DIMENSIONLESS` | $2.0\,\pi$ | Maximum azimuthal angle $\Phi_{\max}$ (annular) |
| `screen.N_Phi` | `DIMENSIONLESS` | $32$ | Number of azimuthal sectors $N_\Phi$ (annular) |
| `screen.N_min` | `DIMENSIONLESS` | $1$ | Minimum harmonic order $N_{\min}$ for Thomson frequencies |
| `screen.N_max` | `DIMENSIONLESS` | $3$ | Maximum harmonic order $N_{\max}$ for Thomson frequencies |
| `screen.method` | `str` | `'simplified'` | Field evaluator: `'simplified'` (Form 2) or `'direct'` (Form 1) |

Length inputs (`screen.z_screen`, `screen.width`, `screen.height`, `screen.R_min`, `screen.R_max`) support physical unit scaling, including laser wavelengths (`'lambda'`), beam waist (`'w_0'` or `'w0'`), Bohr radii (`'bohr'` / `'a_B'`), metric scales (`'um'`, `'nm'`, `'mm'`, `'m'`), or raw atomic units (`'au'`). Angle parameters (`screen.Phi_min`, `screen.Phi_max`) support `'pi'` ($1\text{ pi} = \pi\text{ rad}$).

---

## 3. Rectangular Screen Geometry

The rectangular screen spans a planar rectangle perpendicular to $Oz$ at $z = Z_{\text{screen}}$.

```
                     +---------------------------+  y = +H_s / 2
                     |                           |
                     |             ^ y           |
                     |             |             |
                     |             +---> x       |
                     |            (0,0)          |
                     |                           |
                     +---------------------------+  y = -H_s / 2
                   x = -W / 2                  x = +W / 2
```

### 3.1 Spatial Dimensions and Domain
- Width along $Ox$: $W > 0$
- Height along $Oy$: $H_s > 0$ (the subscript $s$ distinguishes screen height from the electron bunch height $h_{\text{beam}}$)
- Transverse coordinate bounds:
  $$x \in \left[-\frac{W}{2}, \frac{W}{2}\right], \qquad y \in \left[-\frac{H_s}{2}, \frac{H_s}{2}\right]$$

### 3.2 Discretization and Cell Centers
The rectangle is subdivided into $N_x$ columns and $N_y$ rows of rectangular pixels:
$$\Delta x = \frac{W}{N_x}, \qquad \Delta y = \frac{H_s}{N_y}$$

Field values are evaluated and stored at **pixel centers**. For zero-based indices $i = 0, \dots, N_x - 1$ and $j = 0, \dots, N_y - 1$:
$$x_i = -\frac{W}{2} + \left(i + \frac{1}{2}\right)\Delta x$$
$$y_j = -\frac{H_s}{2} + \left(j + \frac{1}{2}\right)\Delta y$$

The 3D Cartesian coordinates of screen cell $(j, i)$ are:
$$\mathbf{x}_{ji} = (x_i, y_j, Z_{\text{screen}})$$

### 3.3 Array Layout and Properties
- In `ScreenGeometry`, 1D center coordinate vectors are available as:
  - `geom.x`: array of length $N_x$
  - `geom.y`: array of length $N_y$
- 2D evaluation grids `geom.grid_x` and `geom.grid_y` have shape `(Ny, Nx)`:
  ```python
  xx, _ = np.meshgrid(self.x, self.y)  # shape (Ny, Nx)
  _, yy = np.meshgrid(self.x, self.y)  # shape (Ny, Nx)
  ```
- **Indexing Convention:** Array order is `screen[y, x, ...]`, i.e. index $j$ corresponds to $y$ (rows) and index $i$ corresponds to $x$ (columns). Increasing indices correspond to strictly increasing Cartesian coordinates.
- **Edge cases:** If $N_x = 1$, $x_0 = 0$; if $N_y = 1$, $y_0 = 0$.

---

## 4. Annular / Sector Screen Geometry

The annular screen spans a circular ring or sector perpendicular to $Oz$ at $z = Z_{\text{screen}}$.

```
                              ^ y
                         .---+---.
                      .-'    |    '-.      R_max
                    .'   .---+---.   '.
                   /   .'    |    '.   \   R_min
                  |   /      |      \   |
             <----+---->-----+-------+----> x
                  |   \      |      /   |
                   \   '.    |    .'   /
                    '.   '---+---'   .'
                      '-.    |    .-'
                         '---+---'
```

### 4.1 Polar Domain
- Radial interval: $r \in [R_{\min}, R_{\max}]$, with $0 \le R_{\min} < R_{\max}$
- Azimuthal interval: $\phi \in [\Phi_{\min}, \Phi_{\max}]$, with $\Phi_{\min} < \Phi_{\max}$
- Full circular ring: $\Phi_{\min} = 0$, $\Phi_{\max} = 2\pi$
- Full circular disk: $R_{\min} = 0$, $\Phi_{\min} = 0$, $\Phi_{\max} = 2\pi$

### 4.2 Uniform Surface Area Radial Discretization ($r^2$ Binning)

In polar coordinates, the differential area element is:
$$dA = r\, dr\, d\phi = \frac{1}{2}\, d(r^2)\, d\phi$$

If radial bins were sampled linearly in $r$, outer bins would encompass significantly larger surface areas than inner bins ($\Delta A \propto r$). To ensure **equal physical surface area coverage per cell**:
$$\Delta A = \frac{1}{2} \Delta(r^2) \Delta\phi = \text{constant}$$

The squared radius $u = r^2$ is partitioned into $N_R$ uniform intervals across $[R_{\min}^2, R_{\max}^2]$:
$$\Delta(r^2) = \frac{R_{\max}^2 - R_{\min}^2}{N_R}$$

#### Radial Cell Boundaries (Edges)
For $k = 0, \dots, N_R$:
$$r_{\text{edge}, k} = \sqrt{R_{\min}^2 + k \frac{R_{\max}^2 - R_{\min}^2}{N_R}}$$
- $r_{\text{edge}, 0} = R_{\min}$
- $r_{\text{edge}, N_R} = R_{\max}$

#### Radial Cell Centers
For $i = 0, \dots, N_R - 1$:
$$r_i = \sqrt{R_{\min}^2 + \left(i + \frac{1}{2}\right) \frac{R_{\max}^2 - R_{\min}^2}{N_R}}$$

### 4.3 Azimuthal Sector Discretization

The azimuthal interval $[\Phi_{\min}, \Phi_{\max}]$ is divided into $N_\Phi$ uniform angular sectors:
$$\Delta\phi = \frac{\Phi_{\max} - \Phi_{\min}}{N_\Phi}$$

#### Azimuthal Sector Boundaries (Edges)
For $l = 0, \dots, N_\Phi$:
$$\phi_{\text{edge}, l} = \Phi_{\min} + l \Delta\phi$$

#### Azimuthal Sector Centers
For $j = 0, \dots, N_\Phi - 1$:
$$\phi_j = \Phi_{\min} + \left(j + \frac{1}{2}\right) \Delta\phi$$

### 4.4 2D Cartesian Evaluation Coordinates
For each annular cell $(j, i)$ with sector index $j$ and radial ring index $i$:
$$x_{ji} = r_i \cos\phi_j, \qquad y_{ji} = r_i \sin\phi_j$$
$$\mathbf{x}_{ji} = (x_{ji}, y_{ji}, Z_{\text{screen}})$$

In `ScreenGeometry`:
- Shape attributes map directly: `Nx = N_R`, `Ny = N_Phi`.
- `geom.grid_x` and `geom.grid_y` have shape `(N_Phi, N_R) = (Ny, Nx)`.
- 1D radial and azimuthal center vectors are exposed via `geom.r_centers` (length $N_R$) and `geom.phi_centers` (length $N_\Phi$).

### 4.5 Curvilinear Corner Meshes for Visualization
To render true curvilinear polar quadrilaterals without interpolation artifacts, `ScreenGeometry` provides 2D corner vertex coordinate grids:
$$X_{\text{corners}, l, k} = r_{\text{edge}, k} \cos(\phi_{\text{edge}, l})$$
$$Y_{\text{corners}, l, k} = r_{\text{edge}, k} \sin(\phi_{\text{edge}, l})$$
- Accessible via properties `geom.grid_x_corners` and `geom.grid_y_corners`.
- Both corner meshes have shape:
  $$(N_\Phi + 1, N_R + 1) = (N_y + 1, N_x + 1)$$
- Used directly by `matplotlib.pyplot.pcolormesh(X_c, Y_c, data, shading='flat')`.

---

## 5. Non-Linear Thomson Frequency Grid

Because the observation screen is located at longitudinal coordinate $Z_{\text{screen}}$, the detected radiation frequencies undergo relativistic Doppler shifts and ponderomotive mass shifts (non-linear Thomson dressing).

When constructing `ScreenGeometry.from_parameters()`, the frequency grid $\omega_N$ is calculated for integer harmonic orders $N \in [N_{\min}, N_{\max}]$ according to:

$$\omega_N = N \omega_0 \frac{n_L \cdot q}{n_s \cdot q}$$

where:
- $\omega_0$ is the fundamental laser angular frequency.
- $N$ is the integer harmonic order ($N = N_{\min}, \dots, N_{\max}$).
- $k_N = \omega_N / c$ is the wavenumber in vacuum.

### 5.1 Direction 4-Vectors
- **Laser propagation 4-vector (null vector in $+z$):**
  $$n_L = (1, 0, 0, 1)$$
- **Screen observation direction 4-vector:**
  $$n_s = (1, 0, 0, \operatorname{sgn}(Z_{\text{screen}}))$$
  - Forward screen ($Z_{\text{screen}} > 0$): $n_s = (1, 0, 0, 1) = n_L$
  - Backward screen ($Z_{\text{screen}} < 0$): $n_s = (1, 0, 0, -1)$

### 5.2 Electron Quasi-Momentum (Volkov Ponderomotive Dressing)
The dressed drift momentum $q^\mu$ of the electron inside the laser field is:

$$q = p + (m c)^2 \frac{\langle a^2 \rangle}{2 (p \cdot n_L)} n_L$$

where:
- $p = (p^0, p_x, p_y, p_z)$ is the bare electron 4-momentum ($p^0 = \sqrt{(m c)^2 + \mathbf{p}^2}$).
- $p \cdot n_L = p^0 - p_z$ is the Minkowski inner product.
- $\langle a^2 \rangle$ is the laser cycle-averaged normalized vector potential squared:
  $$\langle a^2 \rangle = \frac{a_0^2}{2}$$
  This cycle-average holds universally for linear, circular, and elliptical polarizations because $\langle \cos^2 \rangle = 1/2$ and the orthogonal field components add in quadrature without cross terms.

### 5.3 Concrete Frequency Limiting Cases
For an electron initially at rest ($\mathbf{p} = \mathbf{0}$, $p^0 = m c$):
1. **Forward Screen ($Z_{\text{screen}} > 0$):**
   $$n_s = n_L \implies \frac{n_L \cdot q}{n_s \cdot q} = 1 \implies \omega_N = N \omega_0$$
2. **Backward Screen ($Z_{\text{screen}} < 0$):**
   $$n_s = (1, 0, 0, -1) \implies \omega_N = \frac{N \omega_0}{1 + a_0^2/2}$$

---

## 6. Retarded Observation Geometry in Field Calculations

When evaluating the Liénard–Wiechert Fourier-transformed Faraday tensor on the screen, the relative position between the emitting electron and each screen pixel is strictly non-local and time-dependent.

For an electron $a$ with worldline $r_a^\mu(\tau) = (c t_a(\tau), \mathbf{r}_a(\tau))$ and 4-velocity $u_a^\mu(\tau)$:

```
 Electron Trajectory r_a(tau)                  Screen Pixel x_ji
       *---------------------------------------------> [ (x_i, y_j, Z_screen) ]
                              R_0(tau)
```

### 6.1 Displacement 3-Vector $\mathbf{R}_0$
For screen cell $(j, i)$ at Cartesian position $\mathbf{x}_{ji} = (x_{ji}, y_{ji}, Z_{\text{screen}})$:
$$\mathbf{R}_{0, a, ji}(\tau) = \mathbf{x}_{ji} - \mathbf{r}_a(\tau) = \begin{pmatrix} x_{ji} - r_{a, x}(\tau) \\ y_{ji} - r_{a, y}(\tau) \\ Z_{\text{screen}} - r_{a, z}(\tau) \end{pmatrix}$$

### 6.2 Distance and Direction Null Vector
The exact distance from the emitter to the observation point is:
$$R_{0, a, ji}(\tau) = |\mathbf{R}_{0, a, ji}(\tau)| = \sqrt{(x_{ji} - r_{a, x})^2 + (y_{ji} - r_{a, y})^2 + (Z_{\text{screen}} - r_{a, z})^2}$$

The unit line-of-sight vector and associated light-like 4-vector are:
$$\mathbf{n}_{R_0}(\tau) = \frac{\mathbf{R}_{0, a, ji}(\tau)}{R_{0, a, ji}(\tau)}$$
$$n_{R_0}^\mu(\tau) = \left(1, \mathbf{n}_{R_0}(\tau)\right)$$

### 6.3 Retarded Phase and Doppler Factor
The phase in the Fourier transform integral is:
$$\Phi_a(\tau) = r_a^0(\tau) + R_{0, a, ji}(\tau) = c t_a(\tau) + |\mathbf{x}_{ji} - \mathbf{r}_a(\tau)|$$

The retarded relativistic Doppler factor in the denominator is:
$$u \cdot n_{R_0} = u^0 - \mathbf{u} \cdot \mathbf{n}_{R_0} = u^0 - (u_x n_x + u_y n_y + u_z n_z)$$

> [!NOTE]
> The evaluator calculates the exact displacement vector $\mathbf{R}_{0, a, ji}(\tau)$ for **every individual electron and every screen pixel**. It does not substitute a common center-of-screen direction, preserving exact spherical wavefront curvature, near-field short-distance corrections ($F_s \propto 1/R_0^2$), and boundary terms ($F_b$).

---

## 7. Faraday Tensor and Observable Data Shapes

At each screen pixel $(j, i)$ and harmonic frequency $\omega_N$, the code evaluates the 6 independent upper-triangular components of the complex contravariant Faraday tensor:

$$(F^{01}, F^{02}, F^{03}, F^{12}, F^{13}, F^{23})$$

### Tensor Array Dimensions
In `ScreenResult`:
- Long-distance radiation field `F_l`: shape `(N_omega, Ny, Nx, 6)`
- Short-distance velocity field `F_s`: shape `(N_omega, Ny, Nx, 6)`
- Finite boundary endpoint field `F_b`: shape `(N_omega, Ny, Nx, 6)`
- Total field `F_total = F_l + F_s + F_b`: shape `(N_omega, Ny, Nx, 6)`

### Scalar Observables on the Screen
1. **Emitted Spectral Intensity:**
   $$I(\omega_N, \mathbf{x}_{ji}) = \sum_{\mu < \nu} |F_{\text{total}}^{\mu\nu}(\omega_N, \mathbf{x}_{ji})|^2$$
   Stored with shape `(N_omega, Ny, Nx)`.
2. **Spectral Angular Momentum Flux Density Along $Oz$:**
   $$\frac{d\mathcal{F}_{J_z}}{d\omega}(\omega_N, \mathbf{x}_{ji}) = \frac{c}{4\pi^2} \operatorname{Re}\left[ x_{ji} \left( F^{03} \bar{F}^{23} - F^{01} \bar{F}^{12} \right) - y_{ji} \left( F^{02} \bar{F}^{12} + F^{03} \bar{F}^{13} \right) \right]$$
   Stored with shape `(N_omega, Ny, Nx)`.
   *Geometric property:* On the optical axis where $x_{ji} = 0$ and $y_{ji} = 0$, $d\mathcal{F}_{J_z}/d\omega \equiv 0$ identically due to the spatial moment prefactors.

---

## 8. Screen Visualization and Rendering

The visualization package (`superradiant_thomson/plotting/screen.py`) renders screen fields with accurate geometric scaling.

```
+-------------------------------------------------------------+
|                      x / w_0 (Top Axis)                     |
|         -2.0       -1.0        0.0        1.0        2.0    |
|   2.0 +-------------------------------------------------+ 2.0
|       |                                                 |   |
| y /   |           [ Annular pcolormesh /                |   | y /
| lambda|             Rectangular imshow ]                |   | w_0
| (Left)|                                                 |   |(Right)
|  -2.0 +-------------------------------------------------+ -2.0
|         -2.0       -1.0        0.0        1.0        2.0    |
|                     x / lambda (Bottom Axis)                |
+-------------------------------------------------------------+
```

### 8.1 Rendering Techniques
- **Rectangular Screen:** Rendered via `ax.imshow()` with:
  ```python
  extent = [
      x[0] - 0.5 * dx, x[-1] + 0.5 * dx,
      y[0] - 0.5 * dy, y[-1] + 0.5 * dy
  ]
  ax.imshow(data_2d, origin='lower', extent=extent, aspect='equal')
  ```
- **Annular Screen:** Rendered via `ax.pcolormesh()` using true curvilinear cell corner meshes:
  ```python
  ax.pcolormesh(geom.grid_x_corners, geom.grid_y_corners, data_2d, shading='flat')
  ax.set_xlim(-1.05 * R_max, 1.05 * R_max)
  ax.set_ylim(-1.05 * R_max, 1.05 * R_max)
  ax.set_aspect('equal')
  ```
  This preserves the exact polar curvature of each $(r, \phi)$ cell without interpolation distortion.

### 8.2 Dual-Unit Coordinate Axes
Screen plots automatically display dual coordinate axes via `add_dual_unit_axes()` in `superradiant_thomson/plotting/style.py`:
- **Primary (Bottom & Left) Axes:** Coordinates normalized to the laser wavelength $\lambda$ ($x/\lambda$, $y/\lambda$).
- **Secondary (Top & Right) Axes:** Coordinates normalized to the laser focal beam waist $w_0$ ($x/w_0$, $y/w_0$).
- Aspect ratio is always locked to `equal` to maintain round circular modes and symmetric diffraction rings.

---

## 9. Python API Reference (`ScreenGeometry`)

### Class Signature
```python
@dataclass(frozen=True)
class ScreenGeometry:
    z_screen: float
    omega: np.ndarray
    harmonics: np.ndarray | None = None
    shape_type: str = 'rectangular'  # 'rectangular' or 'annular'
    
    # Rectangular screen fields
    width: float | None = None
    height: float | None = None
    Nx_rect: int | None = None
    Ny_rect: int | None = None
    
    # Annular screen fields
    R_min: float | None = None
    R_max: float | None = None
    N_R: int | None = None
    Phi_min: float | None = None
    Phi_max: float | None = None
    N_Phi: int | None = None
```

### Factory Constructor
```python
screen_geom = ScreenGeometry.from_parameters(parameters, units=units)
```
Extracts all parameters from the resolved dictionary, computes the non-linear Thomson frequency grid $\omega_N$, and constructs the validated `ScreenGeometry` instance.

### Key Properties

| Property | Return Type | Description |
| :--- | :--- | :--- |
| `Nx` | `int` | Number of columns ($N_x$ for rectangular, $N_R$ for annular) |
| `Ny` | `int` | Number of rows ($N_y$ for rectangular, $N_\Phi$ for annular) |
| `dx`, `dy` | `float` | Pixel dimensions (rectangular only; raises `AttributeError` on annular) |
| `x`, `y` | `np.ndarray` (1D) | Pixel center coordinate arrays (rectangular only) |
| `r_centers` | `np.ndarray` (1D) | Radii of ring centers with equal $r^2$ area binning (annular) |
| `phi_centers` | `np.ndarray` (1D) | Azimuthal angles of sector centers in radians (annular) |
| `r_edges` | `np.ndarray` (1D) | Radii of ring boundaries, length $N_R + 1$ (annular) |
| `phi_edges` | `np.ndarray` (1D) | Angles of sector boundaries, length $N_\Phi + 1$ (annular) |
| `grid_x`, `grid_y` | `np.ndarray` (2D) | Cartesian center coordinates, shape `(Ny, Nx)` |
| `grid_x_corners`, `grid_y_corners` | `np.ndarray` (2D) | Cartesian corner boundaries for `pcolormesh`, shape `(Ny + 1, Nx + 1)` |

---

## 10. Summary Comparison: Rectangular vs. Annular

| Feature | Rectangular Screen | Annular / Sector Screen |
| :--- | :--- | :--- |
| **Shape Identifier** | `'rectangular'` | `'annular'` |
| **Coordinates** | Cartesian $(x, y)$ | Polar $(r, \phi) \to (x, y)$ |
| **Radial/Width Param** | `screen.width` ($W$) | `screen.R_min`, `screen.R_max` ($R_{\min}, R_{\max}$) |
| **Height/Azimuth Param** | `screen.height` ($H_s$) | `screen.Phi_min`, `screen.Phi_max` ($\Phi_{\min}, \Phi_{\max}$) |
| **Grid Resolutions** | `screen.Nx`, `screen.Ny` | `screen.N_R`, `screen.N_Phi` |
| **Area Discretization** | Uniform Cartesian ($\Delta x \times \Delta y$) | Uniform $r^2$ area ($\Delta(r^2) \times \Delta\phi$) |
| **Cell Area $\Delta A$** | $\Delta x \Delta y = \frac{W H_s}{N_x N_y}$ | $\frac{1}{2}\Delta(r^2)\Delta\phi = \frac{(R_{\max}^2 - R_{\min}^2)(\Phi_{\max} - \Phi_{\min})}{2 N_R N_\Phi}$ |
| **Plot Method** | `ax.imshow()` | `ax.pcolormesh()` with curvilinear corner mesh |
| **Primary Use Case** | Broad planar distributions, standard lab detectors | Cylindrical beams, vortex beams (OAM), ring patterns |
