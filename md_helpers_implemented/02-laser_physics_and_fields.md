# Implemented: Laser Physics, Spatial Mode, Fields, and Faraday Tensor

The laser infrastructure (`superradiant_thomson/laser.py` and `superradiant_thomson/lg_mode.py`) implements the complete electromagnetic description of the pulsed Laguerre–Gauss laser beam: temporal factor, dimensionless amplitude scaling, paraxial spatial mode, full vector electric and magnetic fields, and contravariant Faraday tensor representation.

---

## 1. Laser Temporal Factor (`superradiant_thomson/laser.py`)

Using retarded laser time $s = t - z/c$, carrier period $T = 2\pi/\omega$, wing length $L = \text{wing\_factor} \times \sigma_l$, flat-top duration $P = N \times T$, and total duration $D = 2L + P$. The complex temporal factor is $f(s) = A(s) e^{-i\omega s}$:

- $A(s) = 0$ outside $[0, D]$.
- $A(s) = \exp\left(-\left[\frac{s - L}{\sigma_l}\right]^2\right)$ for $0 \le s < L$.
- $A(s) = 1$ for $L \le s \le L + P$.
- $A(s) = \exp\left(-\left[\frac{s - L - P}{\sigma_l}\right]^2\right)$ for $L + P < s \le D$.

### Parameter Schema & Usage
`temporal_schema()` defines:
- `laser.omega`: positive angular frequency in atomic units
- `laser.flat_top_periods`: nonnegative integer number of plateau periods $N$
- `laser.sigma_l`: positive wing time width
- `laser.wing_factor`: positive cutoff factor

```python
from superradiant_thomson.laser import PulseTiming, TemporalFactor

timing = PulseTiming.from_parameters(parameters)
pulse = TemporalFactor(timing, units.c)
factor = pulse(t=time_array, z=0.0)
```

---

## 2. Dimensionless Laser Strength & Carrier Amplitude (`LaserAmplitude`)

The electron-normalized laser strength parameter $a_0 = \frac{|e| A_0}{m_e c}$ fixes the vector potential and electric field amplitudes in Hartree atomic units ($m_e = |e| = 1$):

- $A_0 = a_0 c$
- $E_0 = \omega A_0 = a_0 \omega c$

`amplitude_schema()` accepts `laser.a_0` (default `1.0`).

```python
from superradiant_thomson.laser import LaserAmplitude

amplitude = LaserAmplitude.from_parameters(parameters, units)
# Properties: amplitude.a_0, amplitude.A_0, amplitude.E_0
```

---

## 3. Paraxial Laguerre–Gauss Spatial Mode (`LGMode`)

Propagating along $+z$ with beam waist $w_0$ at $z=0$, the scalar mode $u_{pm}(\mathbf{r})$ and its transverse derivatives $u_x = \partial u/\partial x$, $u_y = \partial u/\partial y$ are evaluated in `superradiant_thomson/lg_mode.py`.

### Parameter Schema
`lg_schema()` defines:
- `laser.p`: nonnegative radial mode index $p \ge 0$
- `laser.m`: nonnegative azimuthal mode index $m \ge 0$
- `laser.epsilon`: azimuthal winding sign $\epsilon = \pm 1$
- `laser.w_0`: beam waist $w_0$ (default $75\lambda$)
- `laser.zeta_x`, `laser.zeta_y`: complex polarization amplitudes (automatically normalized so $|\zeta_x|^2 + |\zeta_y|^2 = 1$)

### Associated Laguerre Representation
$$u_{pm}(\mathbf{r}) = \sqrt{\frac{2 p!}{(p+m)!}} \frac{w_0}{w(z)} \left(\frac{\sqrt{2} q}{w(z)}\right)^m e^{-i(2p+m+1)\psi_G(z)} e^{-\frac{k}{2}\frac{\rho^2}{z_R + i z}} L_p^m\left(\frac{2\rho^2}{w(z)^2}\right)$$
where $q = x + i\epsilon y$, $\rho^2 = x^2 + y^2$, $z_R = \frac{k w_0^2}{2}$, $w(z) = w_0 \sqrt{1 + z^2/z_R^2}$, and $\psi_G(z) = \text{atan2}(z, z_R)$.

```python
from superradiant_thomson.lg_mode import LGMode

mode = LGMode.from_parameters(parameters, units)
u, du_dx, du_dy = mode.evaluate(x, y, z)
```

---

## 4. Full Paraxial Vector Electric and Magnetic Fields (`evaluate_laser_fields`)

Calculates the 6 full paraxial field components at position $\mathbf{r}=(x,y,z)$ over time array $t$ (equations I.2.1.5--I.2.1.10):

- $E_x(\mathbf{r}, t) = E_0 \operatorname{Re}\{\zeta_x u_{pm}(\mathbf{r}) f(\phi)\}$
- $E_y(\mathbf{r}, t) = E_0 \operatorname{Re}\{\zeta_y u_{pm}(\mathbf{r}) f(\phi)\}$
- $E_z(\mathbf{r}, t) = E_0 \operatorname{Re}\left\{ \frac{i}{k} (\zeta_x u_x + \zeta_y u_y) f(\phi) \right\}$
- $B_x(\mathbf{r}, t) = \frac{E_0}{c} \operatorname{Re}\{-\zeta_y u_{pm}(\mathbf{r}) f(\phi)\}$
- $B_y(\mathbf{r}, t) = \frac{E_0}{c} \operatorname{Re}\{\zeta_x u_{pm}(\mathbf{r}) f(\phi)\}$
- $B_z(\mathbf{r}, t) = \frac{E_0}{c} \operatorname{Re}\left\{ \frac{i}{k} (-\zeta_y u_x + \zeta_x u_y) f(\phi) \right\}$

where $\phi = \omega t - k z$. Returns a dictionary `{'E_x': Ex, 'E_y': Ey, 'E_z': Ez, 'B_x': Bx, 'B_y': By, 'B_z': Bz}` in atomic field units.

---

## 5. Contravariant Faraday Tensor & 4-Vector Contraction (`F^{\mu\nu}`)

Using metric signature $(+,-,-,-)$ with 4-position $x^\mu = (ct, \mathbf{r})$ and 4-momentum $p^\mu = (\gamma mc, \mathbf{p})$, the contravariant Faraday tensor is:

$$
F^{\mu\nu} = \begin{pmatrix}
0 & -E_x/c & -E_y/c & -E_z/c \\
E_x/c & 0 & -B_z & B_y \\
E_y/c & B_z & 0 & -B_x \\
E_z/c & -B_y & B_x & 0
\end{pmatrix}
$$

`contract_faraday(F, p)` computes the Lorentz force 4-vector $F^{\mu\nu} p_\nu$:

$$
F^{\mu\nu} p_\nu = \begin{pmatrix}
\frac{1}{c} \mathbf{E} \cdot \mathbf{p} \\
\gamma m \left[ \mathbf{E} + \mathbf{v} \times \mathbf{B} \right]_x \\
\gamma m \left[ \mathbf{E} + \mathbf{v} \times \mathbf{B} \right]_y \\
\gamma m \left[ \mathbf{E} + \mathbf{v} \times \mathbf{B} \right]_z
\end{pmatrix}
$$

```python
from superradiant_thomson.lg_mode import (
    faraday_tensor_from_fields, evaluate_laser_faraday_tensor, contract_faraday
)

F = evaluate_laser_faraday_tensor(mode, amplitude, pulse, (x, y, z), time)
force_4d = contract_faraday(F, p_contravariant)
```

---

## 6. Verification

All physics formulas, normalization conventions, scale resolutions, derivatives, field components, and 4D tensor contraction identities are verified by unit tests in `tests/`:
- `test_laser.py`: Temporal factor, duration, wing cutoffs, and amplitude scaling.
- `test_lg_mode.py`: Spatial LG profile, derivatives vs finite differences, polarization normalization, spot size, and Gouy phase.
- `test_laser_fields.py`: Field component relations ($B_y = E_x/c$), $w_0$ unit resolution, Faraday tensor anti-symmetry, and $F^{\mu\nu} p_\nu$ Lorentz force contraction identity.
