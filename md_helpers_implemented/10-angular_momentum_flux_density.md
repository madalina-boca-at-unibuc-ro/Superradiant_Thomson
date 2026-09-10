# Implemented: Angular Momentum Flux Density Along $Oz$

Module: `superradiant_thomson/screen.py`. Plotting: `superradiant_thomson/plotting/screen.py`. Tests: `tests/test_screen.py`.

Provides the `ScreenResult.angular_momentum_flux_density` property and `plot_screen_angular_momentum_flux_density()` plot generator.

---

## 1. Physical Definitions

### Time Domain
The electromagnetic linear momentum density is:

$$\mathbf{g}(\mathbf{r}, t) = \varepsilon_0 \big( \mathbf{E}(\mathbf{r}, t) \times \mathbf{B}(\mathbf{r}, t) \big)$$

The angular momentum density vector with respect to the coordinate origin is:

$$\mathbf{j}(\mathbf{r}, t) = \mathbf{r} \times \mathbf{g}(\mathbf{r}, t) = \varepsilon_0 \Big( \mathbf{r} \times \big( \mathbf{E}(\mathbf{r}, t) \times \mathbf{B}(\mathbf{r}, t) \big) \Big)$$

The instantaneous local **angular momentum flux density** along the $Oz$ axis (rate of $J_z$ passing through an infinitesimal surface element $dx\,dy$ at position $(x, y, Z_{\text{screen}})$) is:

$$\mathcal{F}_{J_z}(\mathbf{r}, t) = [\mathbf{r} \times \mathbf{g}]_z = x g_y(\mathbf{r}, t) - y g_x(\mathbf{r}, t) = \varepsilon_0 \Big[ x \big( E_z B_x - E_x B_z \big) - y \big( E_y B_z - E_z B_y \big) \Big]$$

---

## 2. Spectral / Fourier Domain

Defining the temporal Fourier transform according to:

$$\mathbf{E}(\mathbf{r}, t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \tilde{\mathbf{E}}(\mathbf{r}, \omega) e^{-i\omega t} d\omega, \qquad \mathbf{B}(\mathbf{r}, t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \tilde{\mathbf{B}}(\mathbf{r}, \omega) e^{-i\omega t} d\omega$$

By Parseval's theorem for cross-products, the time-integrated local angular momentum flux density is:

$$\int_{-\infty}^{\infty} \mathcal{F}_{J_z}(\mathbf{r}, t) \, dt = \int_0^\infty \frac{d\mathcal{F}_{J_z}(\mathbf{r}, \omega)}{d\omega} \, d\omega$$

where the **spectral angular momentum flux density per unit positive angular frequency $d\omega$** is:

$$\frac{d\mathcal{F}_{J_z}(\mathbf{r}, \omega)}{d\omega} = \frac{\varepsilon_0}{\pi} \operatorname{Re}\Big[ x \left( \tilde{E}_z(\mathbf{r}, \omega) \tilde{B}_x^*(\mathbf{r}, \omega) - \tilde{E}_x(\mathbf{r}, \omega) \tilde{B}_z^*(\mathbf{r}, \omega) \right) - y \left( \tilde{E}_y(\mathbf{r}, \omega) \tilde{B}_z^*(\mathbf{r}, \omega) - \tilde{E}_z(\mathbf{r}, \omega) \tilde{B}_y^*(\mathbf{r}, \omega) \right) \Big]$$

---

## 3. Faraday Tensor Component Mapping

The complex Fourier-transformed Faraday tensor components stored in `ScreenResult.F_total` map to physical electric and magnetic fields as follows:

$$\tilde{E}_x = c F^{01}, \quad \tilde{E}_y = c F^{02}, \quad \tilde{E}_z = c F^{03}$$

$$\tilde{B}_x = F^{23}, \quad \tilde{B}_y = -F^{13}, \quad \tilde{B}_z = F^{12}$$

Substituting these relationships into the spectral angular momentum flux density formula yields:

$$\tilde{E}_z \tilde{B}_x^* - \tilde{E}_x \tilde{B}_z^* = c \Big( F^{03} (F^{23})^* - F^{01} (F^{12})^* \Big)$$

$$\tilde{E}_y \tilde{B}_z^* - \tilde{E}_z \tilde{B}_y^* = c \Big( F^{02} (F^{12})^* + F^{03} (F^{13})^* \Big)$$

In Hartree atomic units, $\varepsilon_0 = \frac{1}{4\pi}$. Thus, the prefactor $\frac{\varepsilon_0 c}{\pi} = \frac{c}{4\pi^2}$:

$$\frac{d\mathcal{F}_{J_z}(\mathbf{r}, \omega)}{d\omega} = \frac{c}{4\pi^2} \operatorname{Re}\Bigg[ x \Big( F_{\text{total}}^{03} (F_{\text{total}}^{23})^* - F_{\text{total}}^{01} (F_{\text{total}}^{12})^* \Big) - y \Big( F_{\text{total}}^{02} (F_{\text{total}}^{12})^* + F_{\text{total}}^{03} (F_{\text{total}}^{13})^* \Big) \Bigg]$$

---

## 4. Total Field Evaluation Rule

As specified in `md_helpers_proposed/09-angular_momentum_flux_density.md`, the angular momentum flux is calculated using the **total fields** ($F_{\text{total}} = F_l + F_s + F_b$).

---

## 5. Properties & On-Axis Cancellation

1. **On-Axis Value**: At $x = 0, y = 0$ (the center of the observation screen), $d\mathcal{F}_{J_z} / d\omega \equiv 0$ identically.
2. **Dimension**: Expressed in atomic units of angular momentum per frequency per unit area ($E_{\text{a.u.}} \cdot B_{\text{a.u.}} \cdot \text{length}_{\text{a.u.}}$).
