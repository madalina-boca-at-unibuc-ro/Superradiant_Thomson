# Rectangular observation screen

All quantities use the atomic units specified in units.md.

## Geometry

The laser propagates along positive \(Oz\). The screen is orthogonal to that
direction and lies at

$$z=Z_{\rm screen}.$$

It is centered on the \(Oz\) axis. Its width \(W>0\) is along \(Ox\), and its
height \(H_s>0\) is along \(Oy\):

$$
-W/2\leq x\leq W/2,\qquad
-H_s/2\leq y\leq H_s/2.
$$

The name \(H_s\) distinguishes screen height from electron-cylinder height.

## Resolution and pixel centers

Let \(N_x,N_y>0\) be integer pixel counts along the width and height. Define

$$
\Delta x=W/N_x,\qquad \Delta y=H_s/N_y.
$$

Store fields at pixel centers. For zero-based indices
\(i=0,\ldots,N_x-1\), \(j=0,\ldots,N_y-1\),

$$
x_i=-\frac W2+\left(i+\frac12\right)\Delta x,\qquad
y_j=-\frac{H_s}{2}+\left(j+\frac12\right)\Delta y,
$$

$$
{\bf x}_{ji}=(x_i,y_j,Z_{\rm screen}).
$$

Use array order screen[y, x, ...]. Increasing indices correspond to
increasing Cartesian coordinates. If \(N_x=1\) or \(N_y=1\), the corresponding
single coordinate is zero.

## Stored Faraday-tensor components

Since

$$
F^{\nu\mu}=-F^{\mu\nu},\qquad F^{\mu\mu}=0,
$$

each pixel stores the six upper-triangular components in this fixed order:

$$
(01,02,03,12,13,23)
=\left(F^{01},F^{02},F^{03},F^{12},F^{13},F^{23}\right).
$$

Declare this order as a named constant in code and include it in output
metadata. The full tensor is

$$
F^{\mu\nu}=
\begin{pmatrix}
0&F^{01}&F^{02}&F^{03}\\
-F^{01}&0&F^{12}&F^{13}\\
-F^{02}&-F^{12}&0&F^{23}\\
-F^{03}&-F^{13}&-F^{23}&0
\end{pmatrix}.
$$

## Total field of the electron bunch

At each screen point, sum electron contributions componentwise:

$$
F_{\rm total}^{\mu\nu}(\omega,{\bf x}_{ji})
=\sum_{a=1}^{N_e}F_a^{\mu\nu}(\omega,{\bf x}_{ji}).
$$

Frequency-domain components are complex. Sum complex amplitudes before
calculating any quadratic observable. In electron_bunch.md each physical
electron has weight one. A later macroparticle extension would use
\(\sum_a w_aF_a^{\mu\nu}\).

For electron \(a\) and pixel \((j,i)\), always use the exact displacement

$$
{\bf R}_{a,ji}(\tau)={\bf x}_{ji}-{\bf r}_a(\tau).
$$

Do not replace it with a common screen-center direction unless a separately
specified far-field approximation is enabled.

## Storage

For one frequency:

~~~text
F_total[N_y, N_x, 6]                 complex
~~~

For a frequency grid:

~~~text
F_total[N_omega, N_y, N_x, 6]        complex
omega[N_omega]
~~~

For a time-domain screen:

~~~text
F_total[N_time, N_y, N_x, 6]         real
time[N_time]
~~~

Do not mix time- and frequency-domain results in one array. Store domain,
coordinates, component order, screen position, dimensions, resolution, and
unit policy as metadata.

## Evaluation and parallelism

Conceptually:

~~~text
initialize F_total to zero
for each electron:
    obtain its propagated trajectory
    for each screen pixel:
        evaluate six emitted-field components
        add them to F_total
~~~

Electron and pixel loops may be parallelized. Reduce partial complex tensor
sums deterministically. Pairwise or compensated summation is recommended
when destructive interference makes the total much smaller than individual
contributions.

## Required checks

- Every point has \(z=Z_{\rm screen}\).
- Pixel centers lie inside the rectangle and are symmetric about zero.
- Coordinate arrays are increasing when their resolution exceeds one.
- Six stored values reconstruct an antisymmetric \(4\times4\) tensor.
- One electron reproduces the single-electron evaluator.
- Two identical trajectories produce twice the complex field.
- Reversing all charge signs reverses every component.
- Spatial maps converge as \(N_x,N_y\) increase at fixed \(W,H_s\).
- Serial and parallel reductions agree within floating-point tolerance.

Intensity, energy flux, spectra, and polarization require a separate
observable definition.
