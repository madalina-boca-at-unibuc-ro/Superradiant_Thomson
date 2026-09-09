# Implemented: Main Entry Point (`main.py`)

Run from the project root:

```sh
.venv/bin/python main.py
.venv/bin/python main.py --show
```

Edit the `INPUTS` dictionary in `main.py` to configure laser parameters, LG mode indices, electron bunch properties, screen geometry, and frequency grid. Bare numbers use atomic units; dimensioned dictionaries use scale registries (e.g., `w_0`, `lambda`, `c`, `fs`, `T`).

## Execution Flow

1. **`initialize(inputs)`**:
   Constructs shared `AtomicUnits`, registers temporal/laser/electron/screen parameter schemas, and returns resolved `(units, parameters, pulse)`.

2. **`sample_pulse()`**:
   Evaluates 1001 temporal factor samples at $z=0$, printing pulse timing and amplitude information.

3. **`main()` Pipeline**:
   - **Temporal Factor**: Evaluates and plots temporal profile $f(t-z/c)$.
   - **LG Spatial Mode**: Evaluates focal-plane scalar intensity $|E_0 u_{pm} f|^2$ and line sample derivatives.
   - **6-Panel Field Diagnostics**: Computes vector components ($E_x/c, E_y/c, E_z/c, B_x, B_y, B_z$) at target evaluation point $\mathbf{r}_{\text{plot}}$.
   - **Fused Streaming Parallel Solver**: Calls `compute_screen_emitted_field_from_laser_and_bunch(...)` to generate initial conditions, integrate ODE trajectories on the fly, and compute 6-component FT Faraday tensors $F^{\mu\nu}(\omega, \mathbf{x})$ on the 2D observation screen across CPU worker processes.
   - **Initial Ensemble Scatter Plotting**: Plots 3D position and momentum scatter plots using full gathered ensemble initial conditions $(r_0, u_0)$.
   - **Trajectory Component Plotting**: Plots 4-panel $r^\mu(\tau), u^\mu(\tau), w^\mu(\tau)$ trajectory component plots for up to 10 stored sample electrons.
   - **Screen Intensity & Breakdown Plots**: Generates total screen emitted intensity heatmap and 18 component breakdown figures across calculated frequencies.
   - **Run Output Directory**: Writes all metadata, parameters, inputs, `.npz` arrays, and `.png` plots to an isolated timestamped directory beneath `~/output/`.

CLI flag `--show` enables interactive Matplotlib plot displays; default runs operate headless (`Agg`).

## BLAS Thread Pinning

Before any other import, `main.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1` via `os.environ.setdefault(...)`, before `numpy` is imported.

Reason: the screen solver (`compute_screen_emitted_field_from_laser_and_bunch` in `screen.py`) already parallelizes across `ProcessPoolExecutor` workers, one electron chunk per worker, sized up to `os.cpu_count()`. Without pinning, a multithreaded BLAS backend (observed with Anaconda's MKL build; can also affect OpenBLAS) makes *each* worker process additionally spawn its own thread pool sized to the full core count. On an $N$-core machine this oversubscribes to roughly $N^2$ threads, and every worker is starved down to a few percent CPU instead of the parallel speedup - much more severe the more cores a machine has, so it is most visible on many-core servers and easy to miss on a low-core laptop.

`os.environ.setdefault` only fills in a var if unset, so it never overrides an explicit environment configuration and is inert wherever the effect wasn't happening (e.g. a `.venv` with a single-threaded-by-default BLAS build). It only affects wall-clock performance, not numerical results.

