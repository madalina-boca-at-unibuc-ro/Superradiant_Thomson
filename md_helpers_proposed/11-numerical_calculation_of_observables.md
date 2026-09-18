# Numerical calculation of the angular momentum density and flux

## Calculation of the temporal average of a bi-linear combination of vector fields

If real vector quantities ${\bf A}(t)={\bf A}^*(t)$ and ${\bf B}(t) = {\bf B}^*(t)$ are represented by their Fourier transforms:

$${\bf A}(t) = \int\limits_{-\infty}^{\infty}d\omega\, e^{-i\omega t}\tilde{\bf A}(\omega),\qquad {\bf B}(t) = \int\limits_{-\infty}^{\infty}d\omega\, e^{-i\omega t}\tilde{\bf B}(\omega)$$

their frequency components satisfy:
$$\tilde{\bf A}(\omega)={\bf A}^*(-\omega),\qquad \tilde{\bf B}(\omega) = {\bf B}^*(-\omega)$$

We express the time-integrated average of the bilinear form $C_{ij}(t) = A_i(t)B_j(t)$:
$$\langle C_{ij}\rangle = \int\limits_{-\infty}^{\infty}dt\, A_i(t) B_j(t) = \int\limits_{-\infty}^{\infty}dt\int\limits_{-\infty}^{\infty}d\omega\int\limits_{-\infty}^{\infty}d\omega' e^{-it(\omega-\omega')}\tilde A_i(\omega)\tilde B_j^*(\omega')$$

Evaluating the temporal integral gives $2\pi\delta(\omega-\omega')$, yielding:
$$\langle C_{ij}\rangle = \int\limits_{-\infty}^{\infty}d\omega\, \tilde A_i(\omega)\tilde B_j^*(\omega) = 2\int\limits_{0}^{\infty}d\omega\, \operatorname{Re}\left[\tilde A_i(\omega)\tilde B_j^*(\omega)\right]$$

The one-sided Fourier spectral density for $\omega \in (0,\infty)$ is therefore:
<a id="eq-bilinear-spectral-density"></a>
$$\frac{d\langle C_{ij}\rangle}{d\omega} = 2\operatorname{Re}\left[\tilde A_i(\omega)\tilde B_j^*(\omega)\right] = \tilde A_i(\omega)\tilde B_j^*(\omega) + \tilde A_i^*(\omega)\tilde B_j(\omega) \tag{bilinear-spectral-density}$$

---

## Applications for the angular momentum density

### Notations and Conventions
* All field quantities below are evaluated in the frequency domain for a positive frequency $\omega > 0$. The explicit argument $(\omega)$ and spatial coordinate ${\bf x}$ are suppressed for brevity.
* Fourier amplitudes are denoted with a tilde ($\tilde{\bf E}, \tilde{\bf B}, \tilde{\bf A}$).
* We consider the third ($z$) component of the angular momentum along the beam propagation / detector normal axis.

The complex Faraday tensor calculated on the detector grid is:
$$F^{\mu\nu}({\bf x},\omega)=\left(\begin{array}{cccc}0&-\tilde E_x/c& -\tilde E_y/c& -\tilde E_z/c\\\tilde E_x/c&0&-\tilde B_z&\tilde B_y\\\tilde E_y/c&\tilde B_z&0&-\tilde B_x\\\tilde E_z/c&-\tilde B_y&\tilde B_x&0\end{array}\right)$$

From which we extract the electric and magnetic field components:
<a id="eq-definition-of-E-and-B"></a>
$$\tilde E_x = c F^{10},\quad \tilde E_y = c F^{20},\quad \tilde E_z = c F^{30},\quad \tilde B_x = F^{32},\quad \tilde B_y = F^{13},\quad \tilde B_z = F^{21} \tag{definition-of-E-and-B}$$

In the temporal gauge / Coulomb-radiation gauge, the vector potential $\tilde{\bf A}$ and auxiliary vector $\tilde{\bf C}$ satisfy:
$$\tilde {\bf A} = \frac{1}{i\omega}\tilde {\bf E},\qquad \tilde {\bf C}=\frac{c}{i\omega}\tilde {\bf B}$$

In terms of the tensor components:
<a id="eq-definition-of-A"></a>
$$\tilde A_x = \frac{c}{i\omega}F^{10},\quad \tilde A_y = \frac{c}{i\omega}F^{20},\quad \tilde A_z = \frac{c}{i\omega}F^{30} \tag{definition-of-A}$$

<a id="eq-definition-of-C"></a>
$$\tilde C_x = \frac{c}{i\omega}F^{32},\quad \tilde C_y = \frac{c}{i\omega}F^{13},\quad \tilde C_z = \frac{c}{i\omega}F^{21} \tag{definition-of-C}$$

---

### 1. Spin Angular Momentum (SAM) Density ($Oz$ component)

The fundamental definition of the spin angular momentum density vector in the time domain is:
$${\boldsymbol{\cal S}} = \epsilon_0{\bf E}\times{\bf A}$$

Its $z$-component is ${\cal S}_z = \epsilon_0(E_x A_y - E_y A_x)$. Applying [Eq. (bilinear-spectral-density)](#eq-bilinear-spectral-density) gives:
$$\frac{d {\cal S}_z}{d\omega} = 2\epsilon_0\operatorname{Re}\left[\tilde E_x \tilde A_y^* - \tilde E_y \tilde A_x^*\right]$$

Substituting $\tilde A_i^* = \frac{i}{\omega}\tilde E_i^*$:
$$\tilde E_x \tilde A_y^* - \tilde E_y \tilde A_x^* = \frac{i}{\omega}\left(\tilde E_x \tilde E_y^* - \tilde E_y \tilde E_x^*\right) = -\frac{2}{\omega}\operatorname{Im}\left[\tilde E_x \tilde E_y^*\right] = \frac{2}{\omega}\operatorname{Im}\left[\tilde E_x^* \tilde E_y\right]$$

Thus, the SAM spectral density is:
<a id="eq-primary-definition-of-Sz"></a>
$$\frac{d {\cal S}_z}{d\omega} = \frac{4\epsilon_0}{\omega}\operatorname{Im}\left[\tilde E_x^* \tilde E_y\right] = -\frac{4\epsilon_0}{\omega}\operatorname{Im}\left[\tilde E_x \tilde E_y^*\right] \tag{primary-definition-of-Sz}$$

---

### 2. Orbital Angular Momentum (OAM) Density ($Oz$ component)

The canonical orbital angular momentum density in the time domain is:
$${\boldsymbol{\cal L}} = \epsilon_0 \sum_{i=x,y,z} E_i\, ({\bf r}\times{\boldsymbol\nabla})\, A_i$$

Its $z$-component involves the transverse differential operator $\hat{L}_z = ({\bf r}\times{\boldsymbol\nabla})_z = (x\partial_y - y\partial_x)$:
$${\cal L}_z = \epsilon_0 \sum_{i=x,y,z} E_i\, (x\partial_y - y\partial_x) A_i$$

Applying [Eq. (bilinear-spectral-density)](#eq-bilinear-spectral-density):
$$\frac{d {\cal L}_z}{d\omega} = 2\epsilon_0 \sum_{i=x,y,z} \operatorname{Re}\left[\tilde E_i\, \Big(\hat{L}_z \tilde A_i\Big)^*\right]$$

Using $\tilde A_i^* = \frac{i}{\omega}\tilde E_i^*$ and noting that coordinates and spatial derivative operators are real:
$$\Big(\hat{L}_z \tilde A_i\Big)^* = \frac{i}{\omega}\hat{L}_z \tilde E_i^*$$
$$\operatorname{Re}\left[\tilde E_i \left(\frac{i}{\omega}\hat{L}_z \tilde E_i^*\right)\right] = -\frac{1}{\omega}\operatorname{Im}\left[\tilde E_i \hat{L}_z \tilde E_i^*\right] = \frac{1}{\omega}\operatorname{Im}\left[\tilde E_i^*\, \hat{L}_z \tilde E_i\right]$$

#### A. Rectangular Screen (Cartesian Grid)
On a Cartesian grid $(x, y)$, the operator is $\hat{L}_z = x \partial_y - y \partial_x$:
<a id="eq-primary-definition-of-Lz-rectangular"></a>
$$\frac{d {\cal L}_z}{d\omega} = \frac{2\epsilon_0}{\omega}\sum_{i=x,y,z}\operatorname{Im}\left[\tilde E_i^* \left(x \frac{\partial \tilde E_i}{\partial y} - y \frac{\partial \tilde E_i}{\partial x}\right)\right] \tag{primary-definition-of-Lz-rectangular}$$

#### B. Circular Screen (Polar Grid)
On a polar grid $(\rho, \phi)$ where $x = \rho\cos\phi$, $y = \rho\sin\phi$, we use the identity $x\partial_y - y\partial_x = \partial_\phi$:
<a id="eq-primary-definition-of-Lz-circular"></a>
$$\frac{d {\cal L}_z}{d\omega} = \frac{2\epsilon_0}{\omega}\sum_{i=x,y,z}\operatorname{Im}\left[\tilde E_i^* \frac{\partial \tilde E_i}{\partial \phi}\right] \tag{primary-definition-of-Lz-circular}$$

---

### 3. Spin Angular Momentum Flux and Continuity Equation

The local conservation law for the spin angular momentum in vacuum takes the form of a continuity equation:
<a id="eq-continuity-spin"></a>
$$\partial_t {\cal S}_i + \partial_j \Sigma_{ij} = 0 \tag{continuity-spin}$$

where ${\cal S}_i = \epsilon_0(\mathbf{E} \times \mathbf{A})_i$ is the spin density, and $\Sigma_{ij}$ is the spin flux tensor:
<a id="eq-spin-flux-tensor"></a>
$$\Sigma_{ij} = \epsilon_0 c^2 \Big[ \delta_{ij}(\mathbf{B}\cdot\mathbf{A}) - B_i A_j - B_j A_i \Big] = \frac{1}{\mu_0}\Big[ \delta_{ij}(\mathbf{B}\cdot\mathbf{A}) - B_i A_j - B_j A_i \Big] \tag{spin-flux-tensor}$$

#### Derivation of the Continuity Equation in SI Units
1. In the radiation gauge ($\nabla \cdot \mathbf{A} = 0$, $\phi = 0$), the fields are $\mathbf{E} = -\partial_t \mathbf{A}$ and $\mathbf{B} = \boldsymbol{\nabla} \times \mathbf{A}$. The time derivative of $\boldsymbol{\cal S}$ is:
   $$\partial_t \boldsymbol{\cal S} = \epsilon_0 \left[ (\partial_t \mathbf{E}) \times \mathbf{A} + \mathbf{E} \times (\partial_t \mathbf{A}) \right]$$
   Since $\mathbf{E} \times (\partial_t \mathbf{A}) = -\mathbf{E} \times \mathbf{E} = \mathbf{0}$, and using the vacuum Maxwell-Ampère law $\partial_t \mathbf{E} = c^2 (\boldsymbol{\nabla}\times\mathbf{B})$:
   $$\partial_t S_i = \epsilon_0 c^2 \big[ (\boldsymbol{\nabla}\times\mathbf{B})\times\mathbf{A} \big]_i = \epsilon_0 c^2 \Big[ (\partial_l B_i) A_l - (\partial_i B_l) A_l \Big]$$

2. Taking the spatial divergence of the flux tensor $\partial_j \Sigma_{ij}$:
   $$\partial_j \Sigma_{ij} = \epsilon_0 c^2 \partial_j \Big[ \delta_{ij}(B_l A_l) - B_i A_j - B_j A_i \Big]$$
   Using the transversality conditions $\partial_j A_j = 0$ and $\partial_j B_j = 0$:
   $$\partial_j \Sigma_{ij} = \epsilon_0 c^2 \Big[ (\partial_i B_l) A_l - (\partial_l B_i) A_l + B_j(\partial_i A_j - \partial_j A_i) \Big]$$
   Because $B_j(\partial_i A_j - \partial_j A_i) = [\mathbf{B}\times(\boldsymbol{\nabla}\times\mathbf{A})]_i = [\mathbf{B}\times\mathbf{B}]_i = 0$, this simplifies to:
   $$\partial_j \Sigma_{ij} = \epsilon_0 c^2 \Big[ (\partial_i B_l) A_l - (\partial_l B_i) A_l \Big] = -\partial_t S_i$$
   which confirms $\partial_t S_i + \partial_j \Sigma_{ij} = 0$.

#### Spectral Density of the $z$-directed Flux $\Sigma_{zz}$
We are interested in the flux along $Oz$ of the spin angular momentum ${\cal S}_z$, represented by the component $\Sigma_{zz}$. Setting $i=z, j=z$:
$$\Sigma_{zz} = \epsilon_0 c^2 \Big[ (\mathbf{B}\cdot\mathbf{A}) - 2 B_z A_z \Big] = \epsilon_0 c^2 \Big[ B_x A_x + B_y A_y - B_z A_z \Big]$$

Applying [Eq. (bilinear-spectral-density)](#eq-bilinear-spectral-density) and substituting $\tilde{A}_k^* = \frac{i}{\omega}\tilde{E}_k^*$:
<a id="eq-primary-definition-of-Sigma-zz"></a>
$$\frac{d\Sigma_{zz}}{d\omega} = -\frac{2\epsilon_0 c^2}{\omega}\operatorname{Im}\left[\tilde B_x \tilde E_x^* + \tilde B_y \tilde E_y^* - \tilde B_z \tilde E_z^*\right] = \frac{2\epsilon_0 c^2}{\omega}\operatorname{Im}\left[\tilde B_x^* \tilde E_x + \tilde B_y^* \tilde E_y - \tilde B_z^* \tilde E_z\right] \tag{primary-definition-of-Sigma-zz}$$


### 4. Orbital Angular Momentum Flux and Continuity Equation

The local conservation law for the orbital angular momentum in vacuum is:
<a id="eq-continuity-oam"></a>
$$\partial_t L_i + \partial_j \Lambda_{ij} = 0 \tag{continuity-oam}$$

where $L_i = \epsilon_0 \sum_m E_m (\mathbf{r}\times\boldsymbol{\nabla})_i A_m$ is the canonical orbital angular momentum density, and the OAM flux tensor in SI units is:
<a id="eq-oam-flux-tensor"></a>
$$\Lambda_{ij} = \epsilon_0 c^2 \left\{ \varepsilon_{ikl} r_k \left[ \varepsilon_{jmn} B_n (\partial_l A_m) + \frac{1}{2}\delta_{lj}\left(\frac{E^2}{c^2} - B^2\right) \right] + B_j A_i \right\} \tag{oam-flux-tensor}$$

#### Reduction of the Axial Component $\Lambda_{zz}$
For $i = z = 3$ and $j = z = 3$, the trace term vanishes identically because $\varepsilon_{3k3} = 0$. Using the transverse operator $\hat{L}_z = x\partial_y - y\partial_x$, the time-domain axial flux reduces to:
$$\Lambda_{zz} = \epsilon_0 c^2 \Big[ B_y \hat{L}_z A_x - B_x \hat{L}_z A_y + B_z A_z \Big]$$

#### Spectral Density of the $z$-directed Flux $\Lambda_{zz}$
Applying [Eq. (bilinear-spectral-density)](#eq-bilinear-spectral-density) and substituting $\tilde{A}_k^* = \frac{i}{\omega}\tilde{E}_k^*$:

##### A. Rectangular Screen (Cartesian Grid)
Using $\hat{L}_z = x\partial_y - y\partial_x$:
<a id="eq-primary-definition-of-Lambda-zz-rectangular"></a>
$$\frac{d\Lambda_{zz}}{d\omega} = \frac{2\epsilon_0 c^2}{\omega}\operatorname{Im}\left[ \tilde{B}_y^* \left(x\frac{\partial \tilde E_x}{\partial y} - y\frac{\partial \tilde E_x}{\partial x}\right) - \tilde{B}_x^* \left(x\frac{\partial \tilde E_y}{\partial y} - y\frac{\partial \tilde E_y}{\partial x}\right) + \tilde{B}_z^* \tilde E_z \right] \tag{primary-definition-of-Lambda-zz-rectangular}$$

##### B. Circular Screen (Polar Grid)
Using $\hat{L}_z = \partial_\phi$:
<a id="eq-primary-definition-of-Lambda-zz-circular"></a>
$$\frac{d\Lambda_{zz}}{d\omega} = \frac{2\epsilon_0 c^2}{\omega}\operatorname{Im}\left[ \tilde{B}_y^* \frac{\partial \tilde E_x}{\partial \phi} - \tilde{B}_x^* \frac{\partial \tilde E_y}{\partial \phi} + \tilde{B}_z^* \tilde E_z \right] \tag{primary-definition-of-Lambda-zz-circular}$$
---

---

### 5. Electromagnetic Energy Density and Energy Flux (Poynting Vector)

The electromagnetic energy density in the temporal domain (in SI units) is:
$$u(t) = \frac{1}{2}\epsilon_0 \mathbf{E}^2(t) + \frac{1}{2\mu_0} \mathbf{B}^2(t) = \frac{1}{2}\epsilon_0 \left[ \mathbf{E}^2(t) + c^2 \mathbf{B}^2(t) \right]$$


*** Spectral Density of the Energy Density***
For any real field component $E_i(t)$, applying [Eq. (bilinear-spectral-density)](#eq-bilinear-spectral-density) with identical factors yields:
$$\frac{d\langle E_i^2\rangle}{d\omega} = 2\operatorname{Re}\left[\tilde{E}_i \tilde{E}_i^*\right] = 2 |\tilde{E}_i|^2$$

The prefactor $\frac{1}{2}\epsilon_0$ cancels the factor of 2, giving the spectral energy density:
<a id="eq-primary-definition-of-u"></a>
$$\frac{d u}{d\omega} = \epsilon_0 |\tilde{\mathbf{E}}|^2 + \frac{1}{\mu_0} |\tilde{\mathbf{B}}|^2 = \epsilon_0 \left( |\tilde{\mathbf{E}}|^2 + c^2 |\tilde{\mathbf{B}}|^2 \right) \tag{primary-definition-of-u}$$

where $|\tilde{\mathbf{E}}|^2 = |\tilde{E}_x|^2 + |\tilde{E}_y|^2 + |\tilde{E}_z|^2$ and $|\tilde{\mathbf{B}}|^2 = |\tilde{B}_x|^2 + |\tilde{B}_y|^2 + |\tilde{B}_z|^2$.

### 6. Spectral Density of the Axial Energy Flux (Poynting Vector $P_z$)

The energy flux density (Poynting vector) is:
$$\mathbf{P}(t) = \frac{1}{\mu_0}\big(\mathbf{E}(t) \times \mathbf{B}(t)\big) = \epsilon_0 c^2 \big(\mathbf{E}(t) \times \mathbf{B}(t)\big)$$
(note that we use the notation $P$ to distinguish from $S$ which is the spin)

In vacuum, energy conservation is governed by the Poynting continuity theorem:
<a id="eq-continuity-energy"></a>
$$\partial_t u + \boldsymbol{\nabla} \cdot \mathbf{P} = 0 \tag{continuity-energy}$$
The component along the screen normal ($Oz$) is:
$$P_z(t) = \epsilon_0 c^2 \big( E_x(t) B_y(t) - E_y(t) B_x(t) \big)$$

Applying [Eq. (bilinear-spectral-density)](#eq-bilinear-spectral-density):
<a id="eq-primary-definition-of-Pz"></a>
$$\frac{d P_z}{d\omega} = 2\epsilon_0 c^2 \operatorname{Re}\left[ \tilde{E}_x \tilde{B}_y^* - \tilde{E}_y \tilde{B}_x^* \right] = \frac{2}{\mu_0}\operatorname{Re}\left[ \tilde{E}_x \tilde{B}_y^* - \tilde{E}_y \tilde{B}_x^* \right] \tag{primary-definition-of-Pz}$$

*(Sanity check: For a paraxial wave propagating along $+z$ with $\tilde{B}_y = \tilde{E}_x/c$ and $\tilde{B}_x = -\tilde{E}_y/c$, this reduces to $\frac{d P_z}{d\omega} = c \frac{d u}{d\omega}$, matching free-space energy transport at the speed of light).*

## Indications for numerical implementation in Python

1. **Target Screens:** Implement the angular momentum calculations for both the rectangular screen (Cartesian grid) and the circular screen (polar grid)[cite: 4].
2. **Data Ingestion:** For each run and for both exported field states (the emitted radiation and the incident LG beam), load the exported Faraday tensor array $F^{\mu\nu}({\bf x},\omega)$ from disk[cite: 4].
3. **Field Reconstruction:** At every spatial grid node, reconstruct the complex electric field vector $\tilde{\bf E} = (\tilde E_x, \tilde E_y, \tilde E_z)$ and magnetic field vector $\tilde{\bf B} = (\tilde B_x, \tilde B_y, \tilde B_z)$ using [Eq. (definition-of-E-and-B)](#eq-definition-of-E-and-B)[cite: 4].
4. **Spectral Quantity Evaluations:**
   - **SAM Density ($\frac{d{\cal S}_z}{d\omega}$):** Calculate according to [Eq. (primary-definition-of-Sz)](#eq-primary-definition-of-Sz)[cite: 4].
   - **OAM Density ($\frac{d{\cal L}_z}{d\omega}$):**
     - *Rectangular Screen:* Compute numerical derivatives $\frac{\partial \tilde E_i}{\partial x}$ and $\frac{\partial \tilde E_i}{\partial y}$ with `numpy.gradient` and evaluate [Eq. (primary-definition-of-Lz-rectangular)](#eq-primary-definition-of-Lz-rectangular)[cite: 4].
     - *Circular Screen:* Compute azimuthal derivatives $\frac{\partial \tilde E_i}{\partial \phi}$ along the periodic $\phi$ grid and evaluate [Eq. (primary-definition-of-Lz-circular)](#eq-primary-definition-of-Lz-circular)[cite: 4].
   - **Total Angular Momentum (TAM) Density ($\frac{d{\cal J}_z}{d\omega}$):**
     $$\frac{d{\cal J}_z}{d\omega} = \frac{d{\cal L}_z}{d\omega} + \frac{d{\cal S}_z}{d\omega}$$
   - **SAM Flux along $z$ ($\frac{d\Sigma_{zz}}{d\omega}$):** Calculate according to [Eq. (primary-definition-of-Sigma-zz)](#eq-primary-definition-of-Sigma-zz)[cite: 4].
   - - **Energy Density ($\frac{du}{d\omega}$):**
     $$\frac{du}{d\omega} = \epsilon_0 \left( |\tilde{E}_x|^2 + |\tilde{E}_y|^2 + |\tilde{E}_z|^2 \right) + \epsilon_0 c^2 \left( |\tilde{B}_x|^2 + |\tilde{B}_y|^2 + |\tilde{B}_z|^2 \right)$$
   - **Energy Flux along $z$ ($\frac{dP_z}{d\omega}$):**
     $$\frac{dP_z}{d\omega} = 2\epsilon_0 c^2 \operatorname{Re}\left[ \tilde{E}_x \tilde{B}_y^* - \tilde{E}_y \tilde{B}_x^* \right]$$
1. **Surface Integration (Integrated Screen Totals):**
   - Integrate each spectral quantity ($\frac{d{\cal S}_z}{d\omega}$, $\frac{d{\cal L}_z}{d\omega}$, $\frac{d{\cal J}_z}{d\omega}$,  $\frac{d\Sigma_{zz}}{d\omega}$, $\frac{du}{d\omega$}$, $\frac{dP_z}{d\omega}$) across the screen surface using the composite 2D trapezoidal rule:
     - *Rectangular screen:* $\iint \dots\, dx\, dy$ via `scipy.integrate.trapezoid` or nested `numpy.trapz`[cite: 4].
     - *Circular screen:* $\iint \dots\, \rho\, d\rho\, d\phi$, explicitly including the radial Jacobian weight $\rho$[cite: 4].
   - Append the computed surface integrals to `run_log.txt` under labeled columns: `int_dSz_domega`, `int_dLz_domega`, `int_dJz_domega`, `int_dSigma_zz_domega`,`int_du_domega`and `int_dPz_domega`