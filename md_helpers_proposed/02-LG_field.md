# Laguerre--Gauss electromagnetic field propagating along \(Oz\)

Source: shared_src/elm_field/LG_modes/LG_modes.tex, first section, "The
Laguerre Gauss modes". This note is an implementation specification for the
special case in which the beam propagates in the positive \(z\) direction:

$$
{\bf n}={\bf e}_z,\qquad {\bf r}=(x,y,z),\qquad \phi=\omega t-kz.
$$

The principal coding target is the six Cartesian field equations numbered
I.2.1.5--I.2.1.10 in the complete THOMSON document. Their prerequisites are
given first.

## Input parameters and conventions

- \(p=0,1,2,\ldots\) is the radial mode index.
- \(m=0,1,2,\ldots\) is the non-negative azimuthal-mode magnitude used by the
  source formula. Its sign is selected separately by \(\epsilon=\pm1\).
- \(w_0>0\) is the beam waist, \(\omega>0\) the carrier angular frequency, and
  \(c=1/\alpha\) in atomic units.
- \(k=\omega/c\), \(T=2\pi/\omega\), and \(\lambda=cT=2\pi/k\).
- \(z_R=kw_0^2/2\) is the Rayleigh length.
- \(w(z)=w_0\sqrt{1+z^2/z_R^2}\) is the spot size.
- \(\psi_G(z)=\arctan(z/z_R)\) is the Gouy phase. In code, use
  atan2(z, z_R) to make the intended branch explicit.
- \(\zeta_x,\zeta_y\in\mathbb C\) are polarization amplitudes satisfying
  \(\lvert\zeta_x\rvert^2+\lvert\zeta_y\rvert^2=1\).
- \(E_0\) is the real electric-field amplitude. The source states
  \(E_0=\omega\xi mc/|e|\), while calling the dimensionless intensity
  parameter \(a_0\). Until that notation is reconciled, code should accept
  \(E_0\) directly in atomic units rather than infer it from \(a_0\) or
  \(\xi\).
- \(\tau_0\) and \(\tau>0\) are dimensionless pulse-envelope parameters in
  phase units: pulse center and pulse width. They are unrelated to proper time.
- The displayed mode, derivatives, and amplitudes are complex. Take the real
  part only when constructing the physical fields.

All inputs and outputs follow md_helpers/units.md; no unit conversion belongs
inside the field evaluator.

## Scalar Laguerre--Gauss mode

Define

$$
\rho^2=x^2+y^2,\qquad q=x+i\epsilon y,\qquad
s=\frac{2\rho^2}{w(z)^2},
$$

$$
N_{pm}=\frac{\sqrt{2}}{m!}\sqrt{\frac{(p+m)!}{p!}}.
$$

The scalar mode required by all six field components is

$$
u_{pm}({\bf r})=
N_{pm}\frac{w_0}{w(z)}
\left(\frac{\sqrt{2}}{w(z)}\right)^m
e^{-i(2p+m+1)\psi_G(z)}
e^{-\frac{k}{2}\frac{\rho^2}{z_R+iz}}
{}_1F_1(-p,m+1,s)q^m.
$$

For integer \(p\geq0\), the hypergeometric function terminates and can be
evaluated through

$$
{}_1F_1(-p,m+1,s)=\frac{p!\,m!}{(p+m)!}L_p^m(s).
$$

The associated-Laguerre form is often preferable numerically. Use log-gamma
functions for factorial ratios if large mode indices must be supported.

## Transverse derivatives

The longitudinal field components require
\(u_x=\partial u_{pm}/\partial x\) and
\(u_y=\partial u_{pm}/\partial y\). Introduce

$$
C=N_{pm}\frac{w_0}{w(z)}
\left(\frac{\sqrt{2}}{w(z)}\right)^m
e^{-i(2p+m+1)\psi_G(z)}
e^{-\frac{k}{2}\frac{\rho^2}{z_R+iz}},
$$

$$
M_0={}_1F_1(-p,m+1,s),\qquad
M_1={}_1F_1(-p+1,m+2,s).
$$

Direct differentiation gives

$$
u_x=C\left[
M_0\left(mq^{m-1}-q^m\frac{kx}{z_R+iz}\right)
-\frac{p}{m+1}\frac{4x}{w(z)^2}M_1q^m
\right],
$$

$$
u_y=C\left[
M_0\left(i\epsilon m q^{m-1}-q^m\frac{ky}{z_R+iz}\right)
-\frac{p}{m+1}\frac{4y}{w(z)^2}M_1q^m
\right].
$$

The \(q^m\) factors in the \(M_1\) terms follow from differentiating
\(M_0(s)q^m\). They are absent from the corresponding derivative lines in the
current LaTeX source and appear to be typographical omissions. Verify these
analytic derivatives against automatic or finite-difference derivatives of
\(u_{pm}\).

For \(m=0\), set the first terms to zero explicitly; do not evaluate
\(m q^{m-1}\) literally at \(q=0\). For \(m=1\), use \(m q^{m-1}=1\).

## Temporal factor

Use one shared complex pulse factor:

$$
f(\phi)=\exp\left[-i\phi-\frac{(\phi-\tau_0)^2}{\tau^2}\right],
\qquad \phi=\omega t-kz.
$$

The magnetic-field source line writes \((\phi-\tau)^2/\tau^2\), whereas the
electric-field line writes \((\phi-\tau_0)^2/\tau^2\). Since both fields
describe the same pulse, this note uses the electric-field definition for all
six components.

## Electric field: equations I.2.1.5--I.2.1.7

With \(u=u_{pm}({\bf r})\),

$$
E_x({\bf r},t)=E_0\Re\left\{\zeta_x u f(\phi)\right\}.
\tag{I.2.1.5}
$$

$$
E_y({\bf r},t)=E_0\Re\left\{\zeta_y u f(\phi)\right\}.
\tag{I.2.1.6}
$$

$$
E_z({\bf r},t)=E_0\Re\left\{
-\frac{1}{ik}\left(\zeta_xu_x+\zeta_yu_y\right)f(\phi)
\right\}.
\tag{I.2.1.7}
$$

## Magnetic field: equations I.2.1.8--I.2.1.10

$$
B_x({\bf r},t)=\frac{E_0}{c}\Re\left\{-\zeta_yu f(\phi)\right\}.
\tag{I.2.1.8}
$$

$$
B_y({\bf r},t)=\frac{E_0}{c}\Re\left\{\zeta_xu f(\phi)\right\}.
\tag{I.2.1.9}
$$

$$
B_z({\bf r},t)=\frac{E_0}{c}\Re\left\{
-\frac{1}{ik}\left(-\zeta_yu_x+\zeta_xu_y\right)f(\phi)
\right\}.
\tag{I.2.1.10}
$$

## Recommended code interface and evaluation

A field evaluator should accept at least

~~~text
(x, y, z, t, p, m, epsilon, omega, w0,
 zeta_x, zeta_y, E0, pulse_center, pulse_width, c)
~~~

and return real three-vectors E and B. A useful lower-level routine should
return the complex triplets before the real part and multiplication by \(E_0\).

Evaluate in this order:

1. Validate \(p,m\in\mathbb N_0\), \(\epsilon\in\{-1,+1\}\), and positive
   \(\omega,w_0,\tau,c\); check polarization normalization within tolerance.
2. Compute \(k,z_R,w(z),\psi_G(z),\rho^2,q,s,N_{pm}\).
3. Compute \(C,M_0,M_1\), then \(u,u_x,u_y\).
4. Compute \(\phi\) and \(f(\phi)\) once.
5. Assemble equations I.2.1.5--I.2.1.10 and take their real parts.

## Minimum numerical checks

- Compare \(u_x,u_y\) with finite differences or automatic differentiation at
  off-axis points for several \((p,m)\), including \(p=0\), \(m=0\), \(m=1\).
- Check the wave normalization \(B=E/c\) for the leading transverse
  components.
- At \(p=m=0,z=0\), confirm that the transverse scalar profile is proportional
  to \(e^{-\rho^2/w_0^2}\).
- For \((\zeta_x,\zeta_y)=(1,0)\), check the leading components
  \(B_y=E_x/c\), \(B_x=0\). For \((0,1)\), check
  \(B_x=-E_y/c\), \(B_y=0\).
- Verify decay with the pulse envelope when
  \(\lvert\phi-\tau_0\rvert\gg\tau\).
- Test \(x=y=0\) separately because naive powers such as \(q^{m-1}\) can
  produce undefined floating-point values despite a finite analytic limit.

These are the paraxial Laguerre--Gauss formulas as stated in the source. Do
not silently generalize them to arbitrary propagation direction; that case
requires an explicit transverse basis.
