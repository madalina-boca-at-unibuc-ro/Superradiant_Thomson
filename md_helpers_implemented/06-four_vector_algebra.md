# Implemented: 4-Vector and 4-Tensor Algebra Package

Module: `superradiant_thomson/four_vector.py`. Tests: `tests/test_four_vector.py`.

Provides relativistic kinematics and tensor algebra in metric signature $(+,-,-,-)$ with $g_{\mu\nu} = \operatorname{diag}(1, -1, -1, -1)$.

---

## 1. Metric Conventions and Index Operations

- Contravariant 4-position: $x^\mu = (ct, \mathbf{r}) = (x^0, x^1, x^2, x^3)$
- Contravariant 4-momentum: $p^\mu = (\gamma mc, \mathbf{p}) = (p^0, p^1, p^2, p^3)$
- Index lowering: $a_\mu = g_{\mu\nu} a^\nu = (a^0, -\mathbf{a})$
- Index raising: $a^\mu = g^{\mu\nu} a_\nu = (a_0, -\mathbf{a}_{\text{spatial}})$

```python
from superradiant_thomson.four_vector import lower_index, raise_index

a_lower = lower_index(a)
a_raised = raise_index(a_lower)
```

---

## 2. Inner Products & Norms (`minkowski_dot`, `minkowski_norm_sq`)

- Inner product: $a \cdot b = a^\mu b_\mu = a^0 b^0 - \mathbf{a} \cdot \mathbf{b}$
- Squared norm: $a \cdot a = (a^0)^2 - |\mathbf{a}|^2$

```python
from superradiant_thomson.four_vector import minkowski_dot, minkowski_norm_sq

dot = minkowski_dot(a, b)
norm_sq = minkowski_norm_sq(a)
```

---

## 3. Kinematic 4-Vector Constructors

- `four_position(ct, r)`: constructs $x^\mu = (ct, x, y, z)$.
- `four_momentum(p_spatial, m, c)`: constructs mass-shell 4-momentum $p^\mu = (p^0, \mathbf{p})$ where $p^0 = \sqrt{m^2 c^2 + |\mathbf{p}|^2}$ ($p \cdot p = m^2 c^2$).
- `four_velocity(p, m)`: constructs $u^\mu = p^\mu / m = (\gamma c, \gamma \mathbf{v})$ ($u \cdot u = c^2$).
- `four_acceleration(F, u, q, m)`: computes $w^\mu = \frac{du^\mu}{d\tau} = \frac{q}{m^2} F^{\mu\nu} u_\nu$ ($u \cdot w = 0$).

---

## 4. 4D Anti-Symmetric Outer Product (`wedge_4d`)

Computes anti-symmetric tensor $(a \wedge b)^{\mu\nu} = a^\mu b^\nu - a^\nu b^\mu$ with shape `(..., 4, 4)`:

```python
from superradiant_thomson.four_vector import wedge_4d

w = wedge_4d(a, b)  # Shape (..., 4, 4), w[i,j] == -w[j,i]
```

---

## 5. Verification

All functions support arbitrary batch shapes (e.g. `(N_particles, 4)` or `(N_pixels, N_tau, 4)`). Verified by unit tests in `tests/test_four_vector.py`:
- Minkowski inner products and metric signature.
- Index raising and lowering inverse identity.
- Mass-shell identity $p \cdot p = m^2 c^2$.
- 4-velocity norm $u \cdot u = c^2$.
- 4-acceleration orthogonality $u \cdot w = 0$.
- Batch array broadcasting.
