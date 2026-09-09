# Implemented: Longitudinal Faraday Field ($F^{03}$) Cancellation and Total Field Breakdown

Module: `superradiant_thomson/plotting/screen.py`. Tests: `tests/test_screen.py`.

Provides documentation and implementation for total Faraday tensor field heatmaps ($F_{\text{total}} = F_l + F_s + F_b$) and mathematical proof of on-axis $F^{03}$ cancellation in Form 2 ("simplified") integration-by-parts formulation.

---

## 1. On-Axis $F^{03}$ Tensor Identity & Cancellation Proof

In Form 2 (`FT-simplified.tex`), both the long-distance component $F_l^{\mu\nu}$ and boundary term $F_b^{\mu\nu}$ contain the tensor factor:

$$T^{\alpha\beta}(\tau) = \frac{n_{R_0}^\alpha u^\beta(\tau) - n_{R_0}^\beta u^\alpha(\tau)}{u(\tau) \cdot n_{R_0}(\tau)}$$

For the $03$ longitudinal component ($\alpha=0, \beta=3$) on the central $z$-axis ($x = 0, y = 0$):

$$T^{03}(\tau) = \frac{u^z(\tau) - n_{R_0}^z u^0(\tau)}{u^0(\tau) - n_{R_0}^z u^z(\tau)}$$

- **Forward Screen ($Z_{\text{screen}} > 0 \implies n_{R_0}^z = +1$)**:
  $$T^{03}(\tau) = \frac{u^z - u^0}{u^0 - u^z} \equiv -1 \quad \text{(EXACTLY } -1 \text{ for ALL } \tau \text{ and ALL velocities!)}$$

- **Backward Screen ($Z_{\text{screen}} < 0 \implies n_{R_0}^z = -1$)**:
  $$T^{03}(\tau) = \frac{u^z + u^0}{u^0 + u^z} \equiv +1 \quad \text{(EXACTLY } +1 \text{ for ALL } \tau \text{ and ALL velocities!)}$$

---

## 2. Integration-by-Parts Artifact vs Physical Total Field

Because $T^{03} = \mp 1$ is constant on axis:

1. **$F_l^{03}$ in Form 2**:
   $$F_l^{03}(\omega) = \frac{q}{2\pi c^2} (\mp 1) \int_{\tau_0}^{\tau_{\max}} \left( \frac{-i k}{R_0(\tau)} \right) e^{i k \Phi(\tau)} \, d\tau$$

2. **$F_b^{03}$ in Form 2**:
   $$F_b^{03}(\omega) = \frac{q}{2\pi c^2} (\mp 1) \left[ \frac{e^{i k \Phi(\tau)}}{R_0(\tau)} \right]_{\tau_0}^{\tau_{\max}}$$

Using $\frac{d}{d\tau} e^{i k \Phi(\tau)} = i k (u \cdot n_{R_0}) e^{i k \Phi(\tau)}$, integration by parts gives:

$$\int_{\tau_0}^{\tau_{\max}} \left( \frac{-i k}{R_0} \right) e^{i k \Phi} \, d\tau \approx -\left[ \frac{e^{i k \Phi}}{R_0} \right]_{\tau_0}^{\tau_{\max}} \implies F_l^{03} + F_b^{03} \approx 0$$

- **$F_l^{03}$ and $F_b^{03}$ are individually large** in Form 2 as an artifact of integration by parts.
- **Form 1 ("direct" Liénard-Wiechert form)** has no integration by parts; its long-distance component $F_{l,\text{direct}}^{03}$ is driven by acceleration $w^z \mp w^0$, which is identically $0$ on axis.
- **The physical observable** is the total Faraday tensor:
  $$\tilde{F}_{\text{total}}^{\mu\nu}(\omega, \mathbf{x}) = \tilde{F}_l^{\mu\nu}(\omega, \mathbf{x}) + \tilde{F}_s^{\mu\nu}(\omega, \mathbf{x}) + \tilde{F}_b^{\mu\nu}(\omega, \mathbf{x})$$

---

## 3. Total Field Breakdown Heatmaps

`plot_screen_faraday_component_breakdown` and `generate_all_screen_breakdown_plots` support four contribution options:

- `'total'` ($F_{\text{total}} = F_l + F_s + F_b$): the complete physical Faraday tensor field.
- `'long'` ($F_l$): long-distance radiation component.
- `'short'` ($F_s$): short-distance velocity/Coulomb component.
- `'boundary'` ($F_b$): finite integration interval endpoint boundary term.

Each harmonic frequency subfolder (e.g. `screen_breakdown_N_1_omega_0.9950_omega0/`) generates 24 4-panel breakdown figures (6 tensor components $\times$ 4 contributions):

```
screen_F01_total.png, screen_F01_long.png, screen_F01_short.png, screen_F01_boundary.png
screen_F02_total.png, screen_F02_long.png, screen_F02_short.png, screen_F02_boundary.png
screen_F03_total.png, screen_F03_long.png, screen_F03_short.png, screen_F03_boundary.png
screen_F12_total.png, screen_F12_long.png, screen_F12_short.png, screen_F12_boundary.png
screen_F13_total.png, screen_F13_long.png, screen_F13_short.png, screen_F13_boundary.png
screen_F23_total.png, screen_F23_long.png, screen_F23_short.png, screen_F23_boundary.png
```
