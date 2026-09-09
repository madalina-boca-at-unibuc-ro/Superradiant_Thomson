# General advice for the numerical implementation

The Markdown helpers in this folder form the scientific specification for a
new numerical-code workspace. Copy them unchanged into that workspace and
keep implementation-specific documentation alongside the code.

Additional helpers may be introduced when a concrete need appears, such as
configuration-file syntax, output formats, observables, or a selected
far-field approximation.

## Recommended implementation order

Build and verify the code incrementally:

1. Atomic-unit constants and shared mathematical conventions.
2. Laguerre--Gauss field evaluator.
3. Single-electron equations of motion.
4. Dense trajectory interpolation providing \(r^\mu,u^\mu,w^\mu\).
5. Single-electron time-domain emitted field.
6. Independent direct and simplified Fourier-transform implementations.
7. Random electron-bunch generator.
8. Rectangular screen geometry and six-component tensor storage.
9. Coherent accumulation over the electron bunch.
10. Physical observables, data export, and plotting.

Do not begin optimization or large simulations before the corresponding
single-electron and single-pixel calculations pass their tests.

## Separate the software layers

Keep these responsibilities distinct:

~~~text
configuration -> physics kernels -> orchestration/parallelism -> output/plots
~~~

- Configuration code reads files, applies defaults, validates parameters, and
  constructs typed inputs.
- Physics kernels receive validated values and implement formulas without
  reading files or global settings.
- Orchestration code schedules electrons, pixels, and frequencies and combines
  partial results.
- Output code stores metadata and arrays and constructs plots or observables.

This separation keeps the scientific routines independently testable.

## Start with deterministic limiting cases

Before random bunches, use:

- one electron;
- one screen pixel;
- one frequency;
- an electron initially at rest;
- zero momentum spread;
- the fundamental \(p=m=0\) LG mode;
- field-free straight-line motion;
- fixed seeds for every stochastic test.

Increase one source of complexity at a time. Maintain at least one small,
fixed end-to-end reference case with stored expected results.

## Shared conventions

Define these once and use them everywhere:

$$
(0,1,2,3)=(ct,x,y,z),
$$

$$
\text{independent tensor order}=(01,02,03,12,13,23),
$$

$$
g_{\mu\nu}=\operatorname{diag}(1,-1,-1,-1).
$$

All internal quantities use the atomic-unit policy in units.md, including
\(c=1/\alpha\). Centralize constants rather than passing independently rounded
values to different modules.

Use explicit names that distinguish:

- proper time from pulse-duration parameters;
- electron-cylinder height from screen height;
- Bohr radius from the laser parameter often called \(a_0\);
- covariant and contravariant four-vector components;
- driving laser field from emitted radiation field;
- time-domain real fields from frequency-domain complex amplitudes.

## Independent cross-check implementations

Implement the direct and simplified Fourier forms independently. They may
share:

- trajectory data;
- atomic constants;
- Minkowski dot products;
- antisymmetric tensor storage;
- quadrature infrastructure.

They should not share one algebraic integrand implementation. Otherwise their
agreement cannot expose a shared transcription or sign error.

Compare the total tensors from both forms, not their separately defined
long- and short-range pieces.

## Trajectory reuse

Integrate each electron trajectory once. Save it or retain dense interpolants
and reuse it for all screen pixels and frequencies.

The trajectory representation must evaluate

$$
r^\mu(\tau),\qquad u^\mu(\tau),\qquad w^\mu(\tau)
$$

at arbitrary requested proper times. Prefer calculating \(w^\mu\) from the
equation-of-motion right-hand side instead of numerically differentiating
sampled velocities.

Do not reintegrate a trajectory independently for every pixel or frequency.

## Coherent accumulation

Preserve the complex frequency-domain tensor as the primary result:

$$
F_{\rm total}^{\mu\nu}(\omega,{\bf x})
=\sum_{a=1}^{N_e}F_a^{\mu\nu}(\omega,{\bf x}).
$$

Always sum electron amplitudes before computing intensity or another
quadratic observable. Store the six complex tensor components even if the
first desired output is only an intensity image.

Use deterministic pairwise or compensated summation when many large
contributions nearly cancel.

## Screen geometry

The current screen specification uses the exact displacement

$$
{\bf R}_{a,ji}(\tau)
={\bf x}_{ji}-{\bf r}_a(\tau).
$$

Do not replace it implicitly with a common direction or distance. If a
far-field approximation is later required, document it as a distinct model
and test its convergence against exact geometry.

## Testing strategy

Add automated tests after every implementation stage. Useful categories are:

- analytic limiting cases;
- physical invariants;
- dimensional and atomic-unit checks;
- finite-difference or automatic-differentiation comparisons;
- agreement between independent formulas;
- convergence under smaller ODE steps and tighter tolerances;
- convergence under larger Fourier integration intervals;
- convergence with screen resolution;
- statistical tests of bunch distributions;
- serial versus parallel agreement;
- reproducibility from a stored random seed.

Tests should use absolute and relative tolerances chosen for the scale of each
quantity. Avoid a single tolerance for positions, momenta, phases, and fields.

## Configuration

Configuration parsing must remain outside physics functions. A later
configuration.md helper should define:

- parameter names and meanings;
- atomic-unit requirements;
- required and optional fields;
- defaults;
- validation ranges;
- random seeds;
- frequency and time grids;
- screen dimensions and resolution;
- output paths and format;
- selected exact or approximate physical model.

Prefer a human-readable structured format with a schema and clear validation
errors. Do not silently repair invalid physical inputs.

## Output provenance

Every result should store enough metadata to reproduce it:

- complete validated configuration;
- random seed and generator algorithm;
- code version or commit identifier;
- atomic-unit convention and value of \(c\);
- frequency or time coordinates;
- screen coordinates and pixel convention;
- tensor-component ordering;
- electron count and distribution parameters;
- ODE and quadrature methods and tolerances;
- trajectory and integration intervals;
- whether exact geometry or an approximation was used.

Numerical arrays without this metadata should not be treated as reproducible
simulation results.

## Performance and memory

The main workload scales across

$$
N_e\times N_x\times N_y\times N_\omega.
$$

Design early for chunking over electrons, pixels, and frequencies. The stored
frequency-domain screen tensor contains

$$
6N_\omega N_yN_x
$$

complex values, independent of temporary per-electron data.

Parallelization should operate outside the physics kernels. Validate a serial
reference implementation first, then require parallel reductions to agree
within stated floating-point tolerances.

## Development rule

At every stage:

1. implement the smallest testable unit;
2. verify it against an analytic or independent reference;
3. record conventions and assumptions;
4. add regression tests;
5. only then connect it to the next layer.

Scientific correctness and reproducibility take priority over early
performance optimization.
