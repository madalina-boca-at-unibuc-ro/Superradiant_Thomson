# Fourier transform of the Faraday tensor — two equivalent forms

Source: `shared_src/elm_el_sc/frequency-domain/FT-direct.tex` (§ "direct
formula") and `shared_src/elm_el_sc/frequency-domain/FT-simplified.tex`
(§ "simplified form"). Each ends in a highlighted result box; both boxes are
transcribed below.

Related implementation notes: units.md defines the atomic-unit convention,
equations_of_motion.md defines the trajectory solver, and
emitted_field_time.md defines the corresponding time-domain field.

**Purpose of this note**: these are two of a planned set of *equivalent*
analytic expressions for the Fourier transform of the radiation field
$F^{\alpha\beta}(\omega,{\bf x})$ emitted by a single accelerated charge in
Thomson scattering. The end goal is a numerical cross-check: implement each
form independently and verify they produce the same $F^{\alpha\beta}(\omega,{\bf x})$
for the same input trajectory.

## Common notation (shared by both forms)

- $\tau$: proper time of the emitting charge, the variable of integration.
- ${\bf r}_0(\tau)$: particle trajectory (3-vector); $r_0^0(\tau)=c t(\tau)$ is
  the corresponding lab-time coordinate along the worldline.
- $u^\mu(\tau) = dr_0^\mu/d\tau$: four-velocity.
- $w^\mu(\tau) = du^\mu/d\tau$: four-acceleration.
- ${\bf x}_0$: fixed observation point (3-vector); $x_0^0 = ct$ (observation lab time, the outer Fourier-conjugate variable before the change of variable to $\tau$).
- ${\bf R}_0(\tau) = {\bf x}_0 - {\bf r}_0(\tau)$, $|{\bf R}_0| = |{\bf x}_0-{\bf r}_0(\tau)|$.
- ${\bf n}_{R_0}(\tau) = {\bf R}_0(\tau)/|{\bf R}_0(\tau)|$ (unit 3-vector).
- $n_{R_0} = (1, {\bf n}_{R_0})$: null four-vector, $R_0 = |{\bf R}_0|\, n_{R_0}$.
- $k = \omega/c$.
- Dot products are Minkowski four-products, e.g. $u\cdot n_{R_0} = u^0 - {\bf u}\cdot{\bf n}_{R_0}$ (metric signature $(+,-,-,-)$, consistent with $u\cdot u = c^2$).
- The phase in every integral is $k\,(r_0^0(\tau) + |{\bf R}_0(\tau)|)$, i.e. $\omega$ times the retarded-time relation $ct = r_0^0(\tau)+|{\bf x}_0-{\bf r}_0(\tau)|$.
- **Jacobian of the change of variable $t\to\tau$.** The required Jacobian is $dt/d\tau=(u\cdot n_{R_0})/c$. It must be included explicitly; in atomic units the corrected overall factor is $q/c^2$.
- Both forms decompose the result as $F^{\alpha\beta} = F_l^{\alpha\beta} + F_s^{\alpha\beta}$ ("long"/short-range split), but **the split is defined differently in each derivation** — see the note at the end. Only the **total** $F^{\alpha\beta}$ is guaranteed to agree between the two forms; do not expect $F_l$ (direct) to equal $F_l$ (simplified) term by term.

All quantities follow units.md: $q$ is the signed charge and $c=1/\alpha$.

---

## Form 1 — "direct" (from `FT-direct.tex`)

Starting point: direct Fourier transform of the closed-form primary expression
for $F^{\alpha\beta}(x)$ (Liénard–Wiechert field written without an explicit
$\tau$-derivative).

$$
F^{\alpha\beta}(ck,{\bf x}) = F^{\alpha\beta}_{l}(ck,{\bf x}) + F^{\alpha\beta}_{s}(ck,{\bf x})
\tag{IV.1.1.14}
$$

$$
F_l^{\alpha\beta}(ck,{\bf x}) =\frac{1}{2\pi}\frac{q}{c^2}
\int_{-\infty}^{\infty} d\tau\; e^{ik\left(r_0^0(\tau)+|{\bf R}_0(\tau)|\right)}\;
\frac{1}{|{\bf R}_0(\tau)|}
\frac{(u\cdot n_{R_0})(n_{R_0}^{\alpha}w^{\beta}-n_{R_0}^{\beta}w^{\alpha})-(w\cdot n_{R_0})(n_{R_0}^{\alpha}u^{\beta}-n_{R_0}^{\beta}u^{\alpha})}
{(u\cdot n_{R_0})^2}
\tag{IV.1.1.15}
$$

$$
F_s^{\alpha\beta}(ck,{\bf x}) =\frac{q}{2\pi}
\int_{-\infty}^{\infty} d\tau\; e^{ik\left(r_0^0(\tau)+|{\bf R}_0(\tau)|\right)}\;
\frac{1}{|{\bf R}_0(\tau)|^2}
\frac{n_{R_0}^{\alpha}u^{\beta}-n_{R_0}^{\beta}u^{\alpha}}
{(u\cdot n_{R_0})^2}
\tag{IV.1.1.16}
$$

Notes for implementation:
- $F_l$ scales as $1/|{\bf R}_0|$ and carries the atomic-unit prefactor
  $q/c^2$; $F_s$ scales as $1/|{\bf R}_0|^2$ and carries $q$.
- Both integrands are antisymmetric in $\alpha\leftrightarrow\beta$ by
  construction (each is built from $n_{R_0}^{\alpha}(\cdot)^\beta - n_{R_0}^\beta(\cdot)^\alpha$-type combinations), so only the 6 independent components of the antisymmetric tensor need to be evaluated.
- Denominator power: $(u\cdot n_{R_0})^2$ in both terms (**not** $^3$ — see below).
- **Do not confuse $n_{R_0}(\tau)$ with $n_0$.** $n_{R_0}(\tau)=(1,{\bf n}_{R_0}(\tau))$ is the *exact*, $\tau$-dependent unit direction from the emitting charge to the observation point, defined above and used throughout this note. $n_0=(1,{\bf n}_0)$, ${\bf n}_0={\bf x}_0/|{\bf x}_0|$, is a *different*, constant vector used only in the separate long-distance/far-field expansion (`shared_src/elm_el_sc/time-domain/c2.tex`, `c3.tex`); it never appears in the exact Fourier-transform formulas here. An earlier version of the `FT-direct.tex` source used `n_0` as shorthand for `n_{R_0}` in this section — that was a notation bug in the LaTeX source and has since been fixed to write `n_{R_0}` explicitly; this note now matches the corrected source.
- **Jacobian bug (fixed).** The dropped $dt/d\tau$ changed the atomic-unit prefactor from $q/c$ to $q/c^2$ and reduced the denominator power from $(u\cdot n_{R_0})^3$ to $(u\cdot n_{R_0})^2$. The displayed formulas contain the correction.

---

## Form 2 — "simplified" (from `FT-simplified.tex`)

Starting point: Fourier transform of Jackson's compact form
$F^{\alpha\beta}(x) = \frac{q}{c}\frac{1}{u\cdot R_0}\frac{d}{d\tau}\!\left[\frac{R_0^\alpha u^\beta - R_0^\beta u^\alpha}{R_0\cdot u}\right]_{\tau=\tau_r}$,
followed by an integration by parts in $\tau$ that moves a derivative onto the
phase factor. Dropping the boundary term requires the convergence prescription described below.

$$
F^{\alpha\beta}(ck,{\bf x}) = F^{\alpha\beta}_{l}(ck,{\bf x}) + F^{\alpha\beta}_{s}(ck,{\bf x})
\tag{IV.1.2.13}
$$

$$
F^{\alpha\beta}_{l}(ck,{\bf x}) =
\frac{1}{2\pi}\frac{q}{c^2}\int_{-\infty}^{\infty} d\tau\;
e^{ik\left(r_0^0(\tau)+|{\bf R}_0(\tau)|\right)}
\left(-ik\,\frac{u\cdot n_{R_0}}{|{\bf R}_0|}\right)
\frac{n_{R_0}^{\alpha}u^{\beta}-n_{R_0}^{\beta}u^{\alpha}}{n_{R_0}\cdot u}
\tag{IV.1.2.14}
$$

$$
F^{\alpha\beta}_{s}(ck,{\bf x}) =
\frac{1}{2\pi}\frac{q}{c^2}\int_{-\infty}^{\infty} d\tau\;
e^{ik\left(r_0^0(\tau)+|{\bf R}_0(\tau)|\right)}
\left(-\frac{{\bf n}_{R_0}\cdot{\bf u}}{|{\bf R}_0|^2}\right)
\frac{n_{R_0}^{\alpha}u^{\beta}-n_{R_0}^{\beta}u^{\alpha}}{n_{R_0}\cdot u}
\tag{IV.1.2.15}
$$

Notes for implementation:
- Both terms share the same antisymmetric tensor factor. Their scalar weights are $-ik\,(u\cdot n_{R_0})/|{\bf R}_0|$ for $F_l$ and $-({\bf n}_{R_0}\cdot{\bf u})/|{\bf R}_0|^2$ for $F_s$. The latter numerator is a 3-vector dot product.
- This form is algebraically simpler to implement (one shared tensor factor, two scalar weights) and is a good candidate for a first numerical implementation.
- **Jacobian bug (fixed).** An earlier version of `FT-simplified.tex` combined the Jacobian $dt/d\tau=(u\cdot n_{R_0})/c$ with Jackson's own $1/(u\cdot R_0)$ prefactor into $\dfrac{u\cdot n_{R_0}}{u\cdot R_0}=\dfrac1{|{\bf R}_0|}$ — dropping the $1/c$, so the combined factor had units of $1/\text{length}$ instead of being dimensionless as required. The correct combination is $\dfrac{u\cdot n_{R_0}}{c\,(u\cdot R_0)}=\dfrac1{c\,|{\bf R}_0|}$. The source has been corrected: every prefactor from this point on carries $c^2$ instead of $c$ (the shape of the integration-by-parts result is otherwise unchanged). The formulas above already reflect the corrected source.

---

## Shared notation block (reproduced from both source highlightboxes)

$$
k = \frac{\omega}{c},\qquad
R_0 = (|{\bf R}_0|,{\bf R}_0) = |{\bf R}_0|(1,{\bf n}_{R_0}) = |{\bf R}_0|\,n_{R_0},\qquad
{\bf R}_0 = {\bf x}_0-{\bf r}_0(\tau),\qquad
{\bf n}_{R_0} = \frac{{\bf R}_0}{|{\bf R}_0|}.
\tag{IV.1.1.17; IV.1.2.16}
$$

In all integrals, every quantity built from $u$, $w$, ${\bf r}_0$, ${\bf R}_0$, $n_{R_0}$ is evaluated **as a function of the integration variable $\tau$** (not at a fixed retarded time) — the retarded-time constraint has already been absorbed into the change of variable $t\to\tau$ and appears only through the phase $e^{ik(r_0^0(\tau)+|{\bf R}_0(\tau)|)}$.

## Cross-check target

For a given trajectory ${\bf r}_0(\tau)$ (hence $u,w$ known analytically or
numerically) and observation point ${\bf x}_0$, both forms should be
numerically integrated over $\tau$ to give the same total
$F^{\alpha\beta}(ck,{\bf x})$ for every $\alpha,\beta,k$. Recommended check:
compare the **sum** $F_l+F_s$ from each form (not the individual $F_l$, $F_s$
pieces, which are not required to agree between the two derivations — see the
note on differing decompositions above).

## Numerical integration contract

Use the trajectory conventions in equations_of_motion.md, the time-domain
definitions in emitted_field_time.md, and atomic units from units.md.

These are oscillatory improper integrals. The infinite limits must not be
implemented as one arbitrary large finite interval.

1. Select an interval on which the trajectory and derivatives are accurate.
2. Interpolate $r^\mu,u^\mu,w^\mu$ consistently, preferably evaluating $w$
   from the equation-of-motion right-hand side.
3. Use oscillatory quadrature or subdivisions based on the phase
   $\Phi(\tau)=k[r_0^0(\tau)+|{\bf R}_0(\tau)|]$.
4. Enlarge both limits and require convergence of every complex component.
5. Accumulate the six independent antisymmetric components.
6. Compare the two total tensors with absolute and relative tolerances.

Integration by parts requires the boundary term to vanish under the selected
decay, switching, or distributional prescription. It does not generally
vanish at finite numerical endpoints. Retain it explicitly or choose/window
the endpoints until it is below tolerance.

For asymptotically inertial motion, specify adiabatic switching, analytic
subtraction of inertial tails, or an explicit window, and apply the same
physical prescription to both forms.
