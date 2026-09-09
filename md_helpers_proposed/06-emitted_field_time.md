# Time-domain field emitted by one moving charge

Source: shared_src/elm_el_sc/time-domain/c1.tex. This helper specifies
III.1.2.1--III.1.2.4 and uses only the atomic units defined in units.md:
\(1/(4\pi\epsilon_0)=1\), signed charge \(q\), and \(c=1/\alpha\).

## Conventions and trajectory data

Use metric \((+,-,-,-)\), so

$$a\cdot b=a^0b^0-{\bf a}\cdot{\bf b}.$$

The observation event and source trajectory are

$$x=(ct,{\bf x}),\qquad r(\tau)=(r^0(\tau),{\bf r}(\tau)),$$

with proper time \(\tau\), four-velocity and four-acceleration

$$
u=\frac{dr}{d\tau}=(\gamma c,\gamma{\bf v}),\qquad
w=\frac{du}{d\tau},\qquad u\cdot u=c^2,\quad u\cdot w=0.
$$

The source permits a translated origin
\({\mathfrak R}_0=(0,{\boldsymbol{\mathfrak R}}_0)\), with
\(x_0=x-{\mathfrak R}_0\) and \(r_0=r-{\mathfrak R}_0\). Since
\(x_0-r_0=x-r\), code may omit this translation if it uses absolute
coordinates consistently.

## III.1.2.1: retarded proper time

$$
ct=r^0(\tau_r)+|{\bf x}-{\bf r}(\tau_r)|
=r_0^0(\tau_r)+|{\bf x}_0-{\bf r}_0(\tau_r)|.
\tag{III.1.2.1}
$$

Solve

$$g(\tau)=r^0(\tau)+|{\bf x}-{\bf r}(\tau)|-ct=0.$$

At a trial \(\tau\), set

$$
{\bf R}={\bf x}-{\bf r}(\tau),\quad R=|{\bf R}|,\quad
{\bf n}={\bf R}/R,\quad n=(1,{\bf n}).
$$

The analytic derivative is

$$g'(\tau)=u^0-{\bf n}\cdot{\bf u}=u\cdot n>0.$$

Bracket the root in the available trajectory history and use a
bracket-preserving method such as Brent's method. A safeguarded Newton step
may use \(g'\). Reject an event if the root is outside the stored history or
if \(r^0(\tau_r)>ct\). Never substitute observation time directly for source
proper time.

## Retarded separation

After finding \(\tau_r\), evaluate

$$
R_0=x-r(\tau_r)
=(ct-r^0(\tau_r),{\bf x}-{\bf r}(\tau_r)).
$$

Then

$$
R_0^0=|{\bf R}_0|,\qquad R_0\cdot R_0=0,\qquad
R_0=R\,n_{R_0},
$$

$$
R=|{\bf R}_0|,\quad
n_{R_0}=(1,{\bf n}_{R_0}),\quad
{\bf n}_{R_0}={\bf R}_0/R.
$$

Define

$$
D=u\cdot R_0=R(u^0-{\bf u}\cdot{\bf n}_{R_0}).
$$

All \(r,u,w,R_0,D\) below are evaluated at \(\tau_r\). Require \(R>0\) and
\(D>0\); \(R=0\) is the singular point-particle self-field.

## III.1.2.2: Liénard--Wiechert potential

$$
A^\mu(x)=\frac{q}{c}
\frac{u^\mu(\tau_r)}
{u(\tau_r)\cdot[x_0-r_0(\tau_r)]}
=\frac{q}{c}\frac{u^\mu}{D}.
\tag{III.1.2.2}
$$

This is optional if only the tensor is needed, but provides an intermediate
check; it obeys \(\partial_\mu A^\mu=0\).

## III.1.2.3: direct field tensor

$$
F^{\alpha\beta}(x)=
\left.\frac{q}{c}
\frac{
D(R_0^\alpha w^\beta-R_0^\beta w^\alpha)
-[(w\cdot R_0)-c^2](R_0^\alpha u^\beta-R_0^\beta u^\alpha)}
{D^3}\right|_{\tau=\tau_r}.
\tag{III.1.2.3}
$$

This algebraic expression is the recommended implementation. With
\((a\wedge b)^{\alpha\beta}=a^\alpha b^\beta-a^\beta b^\alpha\),

$$
F=\frac{q}{cD^3}
\left[D(R_0\wedge w)-((w\cdot R_0)-c^2)(R_0\wedge u)\right].
$$

Compute six independent components, set the diagonal to zero, and fill
\(F^{\beta\alpha}=-F^{\alpha\beta}\).

## III.1.2.4: Jackson form

$$
F^{\alpha\beta}(x)=
\left.\frac{q}{cD}\frac{d}{d\tau}
\left[\frac{R_0^\alpha u^\beta-R_0^\beta u^\alpha}{R_0\cdot u}\right]
\right|_{\tau=\tau_r}.
\tag{III.1.2.4}
$$

During this derivative the observation event is fixed:

$$\frac{dR_0}{d\tau}=-u,\qquad\frac{du}{d\tau}=w.$$

Analytic expansion reproduces III.1.2.3. Use this as an independent
regression form; avoid production finite differences because interpolation
and cancellation errors are amplified by \(1/D\).

## Connection to the motion solver

For trajectory state \((r^\mu,p^\mu)\) from equations_of_motion.md,

$$
u^\mu=p^\mu/m,\qquad
w^\mu=\frac{1}{m}\frac{dp^\mu}{d\tau}.
$$

Dense trajectory output must provide \(r,u,w\) at arbitrary \(\tau_r\).
Evaluate \(w\) from the ODE right-hand side rather than differentiating noisy
samples. The driving field and emitted field are distinct; do not feed the
point-particle self-field back into the motion unless radiation reaction is
explicitly added.

## Required checks

- Check the retarded residual \(g(\tau_r)\) and \(r^0(\tau_r)\le ct\).
- Check \(R_0^2\approx0\), \(R_0^0>0\), \(D>0\), and \(R>0\).
- Check \(u^2\approx c^2\) and \(u\cdot w\approx0\).
- Check \(F+F^{\mathsf T}\approx0\) and zero diagonal.
- Compare III.1.2.3 with an analytic or automatic-differentiation
  implementation of III.1.2.4.
- Reversing \(q\) must reverse \(A\) and \(F\).
- For \(w=0\), the acceleration term vanishes but the velocity/near-field
  contribution remains.
- At large distance, verify \(1/R\) acceleration-field and \(1/R^2\)
  velocity-field scaling.

These are exact time-domain formulas. Do not substitute the later far-field
retarded-time or inverse-distance expansions.
