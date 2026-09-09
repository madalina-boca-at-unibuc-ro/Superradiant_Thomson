# Initial electron bunch

This helper defines random initial conditions for \(N_e\) physical electrons.
All quantities use the atomic units in units.md. Each particle has
\(m_j=1\), \(q_j=-1\), and weight \(w_j=1\).

## Inputs

~~~text
N_e
center = (x_c, y_c, z_c)
radius = R
height = H
momentum_mean = (px_mean, py_mean, pz_mean)
momentum_std = (px_std, py_std, pz_std)
initial_time = t0
random_seed
~~~

The cylinder axis is parallel to \(Oz\), and all electrons share laboratory
time \(t_0\). Require \(N_e>0\), \(R,H\geq0\), and all three momentum standard
deviations to be nonnegative.

## Uniform positions

For each electron draw independent
\(U_r,U_\theta,U_z\sim\operatorname{Uniform}[0,1)\), and set

$$
\rho=R\sqrt{U_r},\qquad
\theta=2\pi U_\theta,\qquad
\Delta z=H(U_z-\tfrac12),
$$

$$
x_j=x_c+\rho\cos\theta,\qquad
y_j=y_c+\rho\sin\theta,\qquad
z_j=z_c+\Delta z.
$$

The square root is essential for uniform transverse area density.

Degenerate geometries are valid:

- \(R>0,H>0\): uniform solid cylinder.
- \(R>0,H=0\): uniform disk at \(z=z_c\).
- \(R=0,H>0\): uniform line segment along \(Oz\), with
  \(x=x_c,y=y_c\).
- \(R=0,H=0\): all electrons start at \({\bf r}_c\).

Do not divide by \(R\) or \(H\). Random values irrelevant to an exactly
degenerate dimension need not be drawn, which preserves intuitive seeded
behavior.

## Gaussian momenta

For each electron draw independent
\(Z_{x,j},Z_{y,j},Z_{z,j}\sim\mathcal N(0,1)\), then use

$$
p_{x,j}=p_{x,\mathrm{mean}}+\sigma_{p_x}Z_{x,j},\qquad
p_{y,j}=p_{y,\mathrm{mean}}+\sigma_{p_y}Z_{y,j},
$$

$$
p_{z,j}=p_{z,\mathrm{mean}}+\sigma_{p_z}Z_{z,j}.
$$

Thus

$$
{\bf p}_j\sim\mathcal N({\boldsymbol\mu}_p,\Sigma_p),\qquad
\Sigma_p=\operatorname{diag}
(\sigma_{p_x}^2,\sigma_{p_y}^2,\sigma_{p_z}^2).
$$

The components are independent. A zero standard deviation returns the mean
exactly. This is a distribution in momentum, not velocity.

## Equation-of-motion state

For each sampled momentum, construct

$$
p_j^0=\sqrt{c^2+|{\bf p}_j|^2},\qquad
p_j^\mu=(p_j^0,p_{x,j},p_{y,j},p_{z,j}),
$$

$$
\gamma_j=\frac{p_j^0}{c},\qquad
{\bf v}_j=c\frac{{\bf p}_j}{p_j^0},
$$

and

$$
x_j^\mu(\tau_0)=(ct_0,x_j,y_j,z_j).
$$

These initialize equations_of_motion.md. A common proper-time origin
\(\tau_0=0\) may be used even when \(t_0\neq0\).

## Reproducibility and arrays

The generator algorithm, master seed, and all distribution parameters are
part of the simulation definition and must be stored with results. Derive
separate position and momentum random streams from the master seed so changes
to one sampler do not silently change the other sample.

Recommended arrays:

~~~text
position[N_e, 3]
momentum[N_e, 3]
four_position[N_e, 4]
four_momentum[N_e, 4]
charge[N_e]
mass[N_e]
weight[N_e]
~~~

Use Cartesian component order \((x,y,z)\) everywhere.

## Checks

- The same seed and inputs reproduce identical arrays.
- Positions satisfy
  \((x_j-x_c)^2+(y_j-y_c)^2\leq R^2\) and
  \(|z_j-z_c|\leq H/2\).
- For large samples, \(\mathbb E[\rho^2]=R^2/2\),
  \(\mathbb E[z]=z_c\), and \(\operatorname{Var}(z)=H^2/12\).
- Sample momentum means and deviations approach the requested values within
  statistical uncertainty.
- Every momentum satisfies
  \((p_j^0)^2-|{\bf p}_j|^2=c^2\), with \(|{\bf v}_j|<c\).
- Test all four degenerate cylinder cases explicitly.

## Later coherent accumulation

After propagation, add complex amplitudes before calculating observables:

$$
F_{\mathrm{tot}}^{\alpha\beta}(\omega,{\bf x})
=\sum_{j=1}^{N_e}F_j^{\alpha\beta}(\omega,{\bf x}).
$$

Screen geometry and detector observables require a separate helper.
