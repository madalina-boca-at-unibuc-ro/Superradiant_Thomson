# Relativistic equations of motion in an electromagnetic field

Source: shared_src/classical_motion/Lagrangean_theory/Lagrangean_theory.tex,
section "The relativistic Lagrangian theory for a particle in an
electromagnetic field". The primary coding target is equations II.1.2.7 and
II.1.2.8.

This note specifies the proper-time equations for a classical point particle
moving in a prescribed electromagnetic field. For the present numerical
project, the prescribed field can be the Laguerre--Gauss field documented in
md_helpers/LG_field.md.

## Conventions and state variables

- The metric signature is \((+,-,-,-)\).
- Greek indices run over \(0,1,2,3\), and repeated indices are summed.
- The four-position is
  \(x^\mu=(ct,x,y,z)=(ct,{\bf r})\).
- Proper time is \(\tau\), with \(dt/d\tau=\gamma\).
- The four-velocity and kinetic four-momentum are

$$
u^\mu=\frac{dx^\mu}{d\tau}
      =(\gamma c,\gamma{\bf v}),\qquad
p^\mu=mu^\mu=(\gamma mc,\gamma m{\bf v}).
$$

- Lowering an index changes the signs of spatial components:

$$
p_\mu=(p^0,-p^1,-p^2,-p^3).
$$

- \(m>0\) is the particle rest mass and \(q\) is its signed charge. For an
  electron in the convention of md_helpers/units.md, use \(m=1\), \(q=-1\).
- \(F^{\mu\nu}(x)\) is the externally prescribed electromagnetic tensor. Its
  fields must be evaluated at the instantaneous event
  \(({\bf r}(\tau),t(\tau))\), not at the start of the integration step.

The numerical state has eight real components:

$$
Y(\tau)=(x^0,x^1,x^2,x^3,p^0,p^1,p^2,p^3).
$$

Although eight quantities are evolved, the physical initial data obey the
mass-shell constraint, so the momentum has only three independent components.

## Equation II.1.2.7: four-position

$$
\frac{dx^\mu}{d\tau}=\frac{p^\mu}{m}.
\tag{II.1.2.7}
$$

Componentwise,

$$
\frac{d(ct)}{d\tau}=\frac{p^0}{m}=\gamma c,\qquad
\frac{d{\bf r}}{d\tau}=\frac{{\bf p}}{m}=\gamma{\bf v}.
$$

Therefore the laboratory time used by the field evaluator is

$$
t=\frac{x^0}{c}.
$$

## Equation II.1.2.8: four-momentum

$$
\frac{dp^\mu}{d\tau}
=\frac{q}{m}F^{\mu\nu}(x)p_\nu,
\qquad
F_{\mu\nu}(x)=\partial_\mu A_\nu(x)-\partial_\nu A_\mu(x).
\tag{II.1.2.8}
$$

The momentum on the right-hand side is \(p_\nu\), with a lower index. This is
important in code: contracting \(F^{\mu\nu}\) directly with an array holding
\(p^\nu\) without first applying the metric gives incorrect spatial signs.

For implementation with three-vectors, the same dynamics can be written
without constructing \(F^{\mu\nu}\):

$$
\frac{d{\bf p}}{d\tau}
=q\gamma\left({\bf E}+{\bf v}\times{\bf B}\right),
$$

$$
\frac{dp^0}{d\tau}
=\frac{q\gamma}{c}\,{\bf E}\cdot{\bf v}
=\frac{q}{mc}\,{\bf E}\cdot{\bf p},
$$

where

$$
\gamma=\frac{p^0}{mc}
=\sqrt{1+\frac{|{\bf p}|^2}{m^2c^2}},
\qquad
{\bf v}=c\frac{{\bf p}}{p^0}
=\frac{{\bf p}}{\gamma m}.
$$

These component equations fix the tensor sign convention operationally and
are the recommended form when the field routine returns Cartesian E and B.

## Complete right-hand side for a proper-time integrator

Given \(Y(\tau)\):

1. Read \(x^0,{\bf r},p^0,{\bf p}\) from the state.
2. Compute \(t=x^0/c\), \(\gamma=p^0/(mc)\), and
   \({\bf v}=c{\bf p}/p^0\).
3. Evaluate the real physical fields
   \({\bf E}({\bf r},t)\) and \({\bf B}({\bf r},t)\).
4. Return

$$
\frac{dY}{d\tau}=
\left(
\frac{p^0}{m},
\frac{{\bf p}}{m},
\frac{q}{mc}{\bf E}\cdot{\bf p},
q\gamma[{\bf E}+{\bf v}\times{\bf B}]
\right).
$$

The ordering inside the tuple is
\((dx^0/d\tau,d{\bf r}/d\tau,dp^0/d\tau,d{\bf p}/d\tau)\).

For the Laguerre--Gauss field, pass the instantaneous values
\((x,y,z,t)\) to the evaluator defined by md_helpers/LG_field.md. The motion
solver must not take another real part or rescale the returned fields.

## Initial conditions

Given a laboratory-frame initial position \({\bf r}_0\), time \(t_0\), and
velocity \({\bf v}_0\) satisfying \(|{\bf v}_0|<c\), compute

$$
\gamma_0=\frac{1}{\sqrt{1-|{\bf v}_0|^2/c^2}},
$$

$$
x_0^\mu=(ct_0,{\bf r}_0),\qquad
p_0^\mu=(\gamma_0mc,\gamma_0m{\bf v}_0).
$$

If the initial spatial momentum is supplied instead, use

$$
p_0^0=\sqrt{m^2c^2+|{\bf p}_0|^2},\qquad
\gamma_0=\frac{p_0^0}{mc},\qquad
{\bf v}_0=c\frac{{\bf p}_0}{p_0^0}.
$$

The initial proper time may be set to zero independently of \(t_0\).

## Recommended numerical representation

Use atomic units exclusively, as specified in md_helpers/units.md. In
particular, \(c=1/\alpha\) remains explicit and must not be set to one.

For highly relativistic trajectories, evolving the spatial momentum
\({\bf p}\) and reconstructing
\(p^0=\sqrt{m^2c^2+|{\bf p}|^2}\) suppresses numerical drift off the mass
shell. However, a literal implementation of II.1.2.7--II.1.2.8 evolves all
four components and uses the mass-shell error as a diagnostic.

Use an adaptive ODE method with tolerances scaled per component: \(x^0\),
spatial position, \(p^0\), and spatial momentum generally have very different
magnitudes even in atomic units.

## Required checks

### Mass shell

The Lorentz force preserves

$$
p^\mu p_\mu=(p^0)^2-|{\bf p}|^2=m^2c^2.
$$

Monitor the dimensionless residual

$$
\delta_{\rm shell}
=\frac{(p^0)^2-|{\bf p}|^2-m^2c^2}{m^2c^2}.
$$

### Four-velocity normalization

Equivalently,

$$
u^\mu u_\mu=c^2.
$$

### Time monotonicity and speed

For a positive-energy particle,

$$
\frac{dt}{d\tau}=\gamma\geq1,\qquad |{\bf v}|<c.
$$

### Field-free motion

With \({\bf E}={\bf B}=0\), momentum must remain constant and

$$
x^\mu(\tau)=x_0^\mu+\frac{p_0^\mu}{m}(\tau-\tau_0).
$$

### Simple-field regression tests

- In a uniform magnetic field with \({\bf E}=0\), \(p^0\) and
  \(|{\bf p}|\) remain constant.
- In any field, the laboratory-frame energy must satisfy
  \(d(\gamma mc^2)/dt=q{\bf E}\cdot{\bf v}\).
- Changing \(q\) to \(-q\) reverses the Lorentz-force direction for identical
  instantaneous state and fields.

## Optional laboratory-time formulation

If integrating with laboratory time \(t\) instead of proper time, use the six
equations

$$
\frac{d{\bf r}}{dt}={\bf v},
$$

$$
\frac{d{\bf v}}{dt}
=\frac{q}{m\gamma}
\left[
{\bf E}+{\bf v}\times{\bf B}
-\frac{({\bf E}\cdot{\bf v}){\bf v}}{c^2}
\right],
\qquad
\gamma=\frac{1}{\sqrt{1-|{\bf v}|^2/c^2}}.
$$

This system appears later in the same source, but it is not equations
II.1.2.7--II.1.2.8. Do not mix a proper-time left-hand side with a
laboratory-time right-hand side.
