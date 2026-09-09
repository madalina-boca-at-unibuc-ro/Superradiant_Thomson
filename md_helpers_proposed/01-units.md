# Units used by the numerical implementation

All numerical code described by the files in md_helpers/ must use atomic units
exclusively. Do not mix atomic-unit state variables with quantities expressed
in another unit system inside the numerical core.

## Atomic-unit convention

Use Hartree atomic units:

$$
m_e=1,\qquad |e|=1,\qquad \hbar=1,\qquad
\frac{1}{4\pi\epsilon_0}=1,\qquad a_{\rm B}=1,\qquad E_h=1.
$$

Here \(a_{\rm B}\) denotes the Bohr radius. It is distinct from the laser
intensity parameter denoted by \(a_0\) in some source files; code should use
unambiguous names such as bohr_radius and laser_strength.

Relativistic factors remain explicit:

$$
c=\frac{1}{\alpha}\approx137.036.
$$

Do not set \(c=1\). Obtain it from one shared constants module used by the
field, trajectory, and radiation routines.

## Numerical quantities

Use atomic units of length, time, angular frequency, energy, velocity,
momentum, electric field, magnetic field, mass, and signed charge. An electron
therefore has

$$
m=1,\qquad q=-1.
$$

Keeping \(m\) and \(q\) as parameters is useful for tests or other particle
species, but their values must also be expressed in atomic units.

## Electromagnetic formulas

Since \(1/(4\pi\epsilon_0)=1\), omit that factor from implemented radiation
formulas. For example,

$$
\frac{q}{4\pi\epsilon_0c^2}\longrightarrow\frac{q}{c^2}.
$$

The relations

$$
k=\frac{\omega}{c},\qquad
\gamma=\frac{1}{\sqrt{1-|{\bf v}|^2/c^2}},\qquad
p^\mu=mu^\mu
$$

remain unchanged. The Lorentz force retains the form

$$
\frac{d{\bf p}}{dt}=q\left({\bf E}+{\bf v}\times{\bf B}\right),
$$

and a propagating wave uses \(|{\bf B}|=|{\bf E}|/c\).

## Conversion boundary

If comparison with external data requires another unit system, convert only
at input/output boundaries. Internally stored parameters, ODE states,
intermediate arrays, tolerances, and results must remain in atomic units.
Centralize any boundary conversion factors and cover them with tests; never
embed them in field or equation-of-motion functions.
