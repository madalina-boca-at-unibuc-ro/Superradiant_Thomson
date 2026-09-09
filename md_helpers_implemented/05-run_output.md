# Implemented: Isolated Run Output Directory & Provenance

Every invocation of `main.py` creates a new directory below `Path.home() / 'output'`, independent of the working directory. On this machine this is `/home/madalina/output`, outside the Dropbox source tree.

Names use `SRT_YYYYMMDDTHHMMSS_microsecondsZ_randomsuffix`. Timestamps are UTC; a random suffix plus exclusive directory creation prevents reuse/collisions. Existing results are never overwritten by later runs.

## Saved Run Artifacts

Each simulation run automatically saves:

- `inputs.json`: original raw input configuration dictionary.
- `parameters.json`: resolved atomic-unit values and original units/defaults mapping.
- `run.json`: running/complete/failed status, timestamps, constants, library versions, pulse timing, sampling metadata, screen resolution, residual metrics, and error stack trace on failure.
- `temporal_factor.npz` & `temporal_factor.png`: complex temporal envelope diagnostic.
- `lg_mode.npz`: Laguerre–Gauss spatial mode line evaluation and analytic derivatives.
- `lg_intensity.npz` & `lg_intensity.png`: focal-plane scalar mode intensity heatmap.
- `laser_fields.npz` & `laser_fields.png`: 6-panel time-domain EM field component plot ($E_x/c, E_y/c, E_z/c, B_x, B_y, B_z$).
- `electron_trajectory.npz`: 10-electron sample relativistic trajectories (`tau`, `r`, `u`, `w`) and full ensemble initial condition vectors (`r0_all`, `u0_all`).
- `electron_initial_distribution.png`: 2-panel 3D scatter plot of initial positions $(x,y,z)$ and momenta $(p_x,p_y,p_z)$ for all $N_e$ electrons.
- `electron_position_trajectories.png`, `electron_velocity_trajectories.png`, `electron_acceleration_trajectories.png`: 4-panel proper-time component plots for up to 10 stored sample electrons.
- `screen_emitted_field.npz`: complete 6-component complex FT Faraday tensors $F_l, F_s, F_b, F_{\text{total}}$ on the 2D observation screen across calculated frequencies.
- `screen_emitted_intensity.png`: 2D total emitted intensity heatmap on the screen, $x$/$y$ axes in units of the laser wavelength $\lambda$.
- `screen_breakdown_omega_*/`: subfolders containing 18 4-panel component breakdown plots (Real, Imag, Modulus, Phase) for each component and contribution ($F_l, F_s, F_b$) at calculated frequencies, $x$/$y$ axes in units of the beam waist $w_0$ (so the emission pattern can be compared directly against the electron beam radius `electron.R_beam`).

```sh
.venv/bin/python main.py
.venv/bin/python main.py --show
```

`superradiant_thomson/output.py` owns directory creation and atomic JSON writes. Automated tests use disposable temporary directories (`output_root=...`). No output cleanup is automatic; failed runs retain whatever artifacts were written before failure.

