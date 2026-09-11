"""Initialize the simulation parameters and sample the laser temporal envelope.

Run from the project root: .venv/bin/python main.py
Edit INPUTS below to configure this initial example.
"""
import os

# Electron trajectories/screen fields are parallelized across processes
# (superradiant_thomson.screen.compute_screen_emitted_field_from_laser_and_bunch),
# one electron per worker. Without this, BLAS backends (e.g. MKL, OpenBLAS)
# each spawn a thread pool sized to the full core count *inside every worker*,
# oversubscribing the machine many times over and starving all workers.
for _threads_env in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_threads_env, '1')

from typing import Any

import numpy as np
from superradiant_thomson.lg_mode import (
    LGMode, lg_schema, laser_plot_schema, evaluate_laser_fields,
)
from superradiant_thomson.electron import (
    Electron, electron_schema, solve_electron_trajectory,
)

from superradiant_thomson.screen import (
    ScreenGeometry, ScreenResult, screen_schema, compute_screen_emitted_field,
    compute_screen_emitted_field_from_laser_and_bunch,
)

from superradiant_thomson.parameters import (
    AtomicUnits, ParameterResolver, default_registry, register_laser_scales,
)
from superradiant_thomson.laser import (
    PulseTiming, TemporalFactor, register_temporal_scales, temporal_schema,
    LaserAmplitude, amplitude_schema,
)


INPUTS = {
    'laser.omega': 0.057,  # Angular frequency in atomic units.
    'laser.a_0': 0.10,  # Dimensionless electron-normalized laser strength.
    'laser.flat_top_periods': 10,
    'laser.sigma_l': {'value': 2, 'unit': 'T'},
    'laser.wing_factor': 5,
    'laser.p': 0,
    'laser.m': 1,
    'laser.epsilon': -1,
    'laser.w_0': {'value': 75, 'unit': 'lambda'},
    'laser.zeta_x': 1.0,
    'laser.zeta_y': 0.0 + 1.0j,
    'x_plot_laser': {'value': 0.5, 'unit': 'w_0'},
    'y_plot_laser': {'value': 0.0, 'unit': 'w_0'},
    'z_plot_laser': {'value': 0.0, 'unit': 'w_0'},
    'electron.N': 512,
    'electron.seed': 42,
    'electron.NT': 100,
    'electron.x_0': {'value': 0.0, 'unit': 'w_0'},
    'electron.y_0': {'value': 0.0, 'unit': 'w_0'},
    'electron.z_0': {'value': 0.0, 'unit': 'w_0'},
    'electron.R_beam': {'value': 1.50, 'unit': 'w_0'},
    'electron.h_beam': {'value': 0.0, 'unit': 'w_0'},
    'electron.px_beam': {'value': 0.0, 'unit': 'c'},
    'electron.py_beam': {'value': 0.0, 'unit': 'c'},
    'electron.pz_beam': {'value': -1.0, 'unit': 'c'},
    'electron.sigma_px_beam': {'value': 0.0, 'unit': 'c'},
    'electron.sigma_py_beam': {'value': 0.0, 'unit': 'c'},
    'electron.sigma_pz_beam': {'value': 0.0, 'unit': 'c'},
    'screen.shape': 'rectangular',  # 'rectangular' or 'annular'
    'screen.z_screen': {'value': -144000.0, 'unit': 'lambda'},
    'screen.width': {'value': 400.0, 'unit': 'lambda'},
    'screen.height': {'value': 400.0, 'unit': 'lambda'},
    'screen.Nx': 64,
    'screen.Ny': 64,
    'screen.R_min': {'value': 0.0, 'unit': 'lambda'},
    'screen.R_max': {'value': 400.0, 'unit': 'lambda'},
    'screen.N_R': 64,
    'screen.Phi_min': {'value': 0.0, 'unit': 'pi'},
    'screen.Phi_max': {'value': 2.0, 'unit': 'pi'},
    'screen.N_Phi': 64,
    'screen.N_min': 1,
    'screen.N_max': 3,
    'screen.method': 'simplified',  # 'simplified' (Form 2) or 'direct' (Form 1)
}


def initialize(inputs):
    """Resolve input units and construct the configured temporal laser factor."""
    units = AtomicUnits()
    registry = default_registry(units)
    register_laser_scales(registry, units)
    register_temporal_scales(registry)
    parameters = ParameterResolver(
        registry, temporal_schema() | lg_schema() | laser_plot_schema() | amplitude_schema() | electron_schema() | screen_schema()).resolve(inputs)
    pulse = TemporalFactor(PulseTiming.from_parameters(parameters), units.c)
    return units, parameters, pulse


def sample_pulse(inputs=None):
    inputs = inputs if inputs is not None else INPUTS
    units, parameters, pulse = initialize(inputs)
    timing = pulse.timing
    # At z=0, laboratory time equals laser time s=t-z/c.
    time = np.linspace(0.0, timing.duration, 1001)
    envelope = pulse.envelope(time)
    temporal_factor = pulse(time, z=0.0)

    print('Laser temporal envelope initialized (Hartree atomic units).')
    print(f'Angular frequency: {parameters["laser.omega"]:.8g}')
    print(f'Period: {timing.period:.8g} a.u.')
    print(f'Flat top: {timing.flat_top_periods} periods')
    print(f'Each wing: {timing.wing_duration:.8g} a.u.')
    print(f'Total duration: {timing.duration:.8g} a.u. '
          f'({timing.duration * units.time_in_s / 1e-15:.8g} fs)')
    print(f'Evaluated {time.size} envelope and complex temporal-factor samples at z=0.')
    print(f'Endpoint amplitude: {envelope[0]:.8g}; peak: {envelope.max():.8g}')
    return units, parameters, pulse, time, envelope, temporal_factor


def main(*, show=False, output_root=None, inputs=None):
    inputs = inputs if inputs is not None else INPUTS
    from dataclasses import asdict
    from datetime import datetime, timezone
    import platform
    import scipy
    from superradiant_thomson.output import create_run_directory, write_json

    run_dir = create_run_directory(output_root)
    print(f'Run output: {run_dir}')
    metadata: dict[str, Any] = {
        'status': 'running',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__},
    }
    write_json(run_dir / 'run.json', metadata)
    write_json(run_dir / 'inputs.json', inputs)
    figures = []
    try:
        from superradiant_thomson.plotting import (
            plot_temporal_factor, plot_lg_intensity, plot_laser_fields,
            plot_electron_initial_distribution, plot_electron_ensemble_trajectories,
            plot_screen_emitted_intensity, generate_all_screen_breakdown_plots,
            plot_screen_angular_momentum_flux_density,
        )
        import matplotlib
        if not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        result = sample_pulse(inputs=inputs)
        units, parameters, pulse, time, envelope, temporal_factor = result
        mode = LGMode.from_parameters(parameters, units)
        amplitude = LaserAmplitude.from_parameters(parameters, units)
        metadata['laser_amplitude'] = dict(asdict(amplitude), A_0=amplitude.A_0,
                                           E_0=amplitude.E_0, units='Hartree atomic units',
                                           convention='a_0=|e|A_0/(m_e*c); E_0=omega*A_0')
        print(f'Laser strength a_0: {amplitude.a_0:g}; E_0: {amplitude.E_0:.8g} a.u.')
        # A focal-plane line sample for the initial spatial-mode demonstration.
        x = np.linspace(-3 * mode.w_0, 3 * mode.w_0, 601)
        u, ux, uy = mode.evaluate(x, 0.0, 0.0)
        np.savez_compressed(run_dir / 'lg_mode.npz', x=x, y=0.0, z=0.0,
                            u=u, du_dx=ux, du_dy=uy)
        metadata['lg_mode'] = dict(asdict(mode), z_R=mode.z_R,
                                   wavelength=mode.get_lambda(),
                                   sample_geometry='x line at y=z=0',
                                   mode_units='dimensionless', derivative_units='inverse bohr')
        write_json(run_dir / 'parameters.json', parameters.to_dict())
        metadata.update(constants=asdict(units), timing=asdict(pulse.timing),
                        duration_au=pulse.timing.duration, z_au=0.0,
                        sample_count=time.size, array_units='Hartree atomic units')
        metadata['versions']['matplotlib'] = matplotlib.__version__
        np.savez_compressed(run_dir / 'temporal_factor.npz', time=time,
                            envelope=envelope, temporal_factor=temporal_factor)
        fig, ax = plot_temporal_factor(pulse, time, time_unit='T')
        figures.append(fig)
        fig.savefig(run_dir / 'temporal_factor.png', dpi=180)
        heatmap_time = pulse.timing.wing_duration
        # Half-extent matches the electron beam radius, so the plot shows
        # whether the laser spot is contained within the electron bunch.
        half_extent = float(parameters['electron.R_beam'])
        if half_extent <= 0:
            half_extent = mode.w_0
        xy = np.linspace(-half_extent, half_extent, 401)
        spatial = mode(xy[None, :], xy[:, None], 0.0)
        intensity = np.abs(amplitude.E_0 * spatial * pulse(heatmap_time, z=0.0))**2
        np.savez_compressed(run_dir / 'lg_intensity.npz', x=xy, y=xy,
                            z=0.0, time=heatmap_time, intensity=intensity)
        metadata['lg_intensity'] = {
            'quantity': '|E_0*u_pm*f|^2 (scalar mode, not full vector E squared)',
            'units': 'atomic electric field squared', 'array_order': 'y,x',
            'time_au': heatmap_time, 'z_au': 0.0,
            'extent_in_waists': [-half_extent / mode.w_0, half_extent / mode.w_0],
            'extent_source': 'electron.R_beam' if half_extent == parameters['electron.R_beam'] else 'w_0 (electron.R_beam was 0)',
            'shape': list(intensity.shape),
        }
        fig, ax = plot_lg_intensity(xy, xy, intensity, w_0=mode.w_0,
                                   lambda_scale=mode.get_lambda(),
                                   time_in_periods=heatmap_time / pulse.timing.period)
        figures.append(fig)
        fig.savefig(run_dir / 'lg_intensity.png', dpi=180)

        # Evaluate and plot 6 EM field components at point r_plot
        r_plot = (parameters['x_plot_laser'], parameters['y_plot_laser'], parameters['z_plot_laser'])
        r_plot_w0 = (r_plot[0] / mode.w_0, r_plot[1] / mode.w_0, r_plot[2] / mode.w_0)
        fields = evaluate_laser_fields(mode, amplitude, pulse, r_plot, time)
        np.savez_compressed(run_dir / 'laser_fields.npz', time=time,
                            x=r_plot[0], y=r_plot[1], z=r_plot[2], **fields)
        metadata['laser_fields'] = {
            'point_au': list(r_plot),
            'point_w0': list(r_plot_w0),
            'field_units': {'E': 'Hartree atomic units of electric field',
                            'B': 'Hartree atomic units of magnetic field'},
        }
        fig_fields, _ = plot_laser_fields(fields, time, r_w0=r_plot_w0, pulse=pulse, time_unit='T')
        figures.append(fig_fields)
        fig_fields.savefig(run_dir / 'laser_fields.png', dpi=180)

        # Compute trajectories and multi-frequency FT Faraday tensor on 2D screen in parallel workers
        screen_geom = ScreenGeometry.from_parameters(parameters, units=units)
        screen_method = str(parameters.get('screen.method', 'simplified'))
        print(f'Screen geometry ({screen_geom.shape_type}) initialized at z_screen = {screen_geom.z_screen / mode.get_lambda():g} lambda using calculation method: {screen_method!r}.')
        print('Calculated non-linear Thomson frequencies:')
        for h_idx, w_au in enumerate(screen_geom.omega):
            n_val = screen_geom.harmonics[h_idx] if screen_geom.harmonics is not None else (h_idx + 1)
            w_rel = w_au / parameters['laser.omega']
            print(f'  Harmonic N={n_val}: omega_{n_val} = {w_au:.8g} a.u. ({w_rel:.6g} omega_0)')

        sample_electron, r0_all, u0_all, screen_result = compute_screen_emitted_field_from_laser_and_bunch(
            mode, amplitude, pulse, units, parameters, screen_geom, max_stored_trajectories=10
        )
        wall_sec = screen_result.wall_time_seconds
        per_elec_sec = screen_result.time_per_electron_seconds
        per_elec_per_point_sec = screen_result.time_per_electron_per_point_seconds
        cpu_per_elec_per_point_sec = screen_result.cpu_time_per_electron_per_point_seconds
        n_points = screen_geom.Nx * screen_geom.Ny
        if wall_sec is not None and per_elec_sec is not None:
            print(f'Computation wall-clock time: {wall_sec:.3f} s ({wall_sec / 60.0:.2f} min)')
            print(f'Average real time per electron ({parameters["electron.N"]} electrons): {per_elec_sec * 1000.0:.3f} ms/electron ({per_elec_sec:.6g} s/electron)')
            if per_elec_per_point_sec is not None:
                print(f'Average real time per electron per screen point ({n_points} pixels): {per_elec_per_point_sec * 1e6:.3f} us/electron/point ({per_elec_per_point_sec * 1000.0:.6g} ms/electron/point)')
            if cpu_per_elec_per_point_sec is not None and screen_result.num_workers is not None:
                print(f'Single-thread CPU cost per electron per screen point ({screen_result.num_workers} threads): {cpu_per_elec_per_point_sec * 1000.0:.3f} ms/electron/point/thread ({cpu_per_elec_per_point_sec:.6g} s/electron/point/thread)')

        np.savez_compressed(run_dir / 'electron_trajectory.npz', tau=sample_electron.tau,
                            r=sample_electron.r, u=sample_electron.u, w=sample_electron.w,
                            r0_all=r0_all, u0_all=u0_all)
        metadata['electron_trajectory'] = {
            'N': parameters['electron.N'],
            'seed': parameters['electron.seed'],
            'NT': parameters['electron.NT'],
            'tau_samples': sample_electron.tau.size,
            'duration_au': pulse.timing.duration,
            'max_mass_shell_residual': float(np.max(np.abs(sample_electron.mass_shell_residual(units.c)))),
        }

        # Plot 3D initial electron positions and momenta scatter plots using full ensemble initial conditions
        fig_dist, _ = plot_electron_initial_distribution(r0_all, u0_all,
                                                         w_0=mode.w_0, c=units.c)
        figures.append(fig_dist)
        fig_dist.savefig(run_dir / 'electron_initial_distribution.png', dpi=180)

        # Plot 4-vector trajectories for up to 10 stored sample electrons
        fig_r, fig_u, fig_w = plot_electron_ensemble_trajectories(
            sample_electron, max_electrons=10, tau_unit='T', lambda_scale=mode.get_lambda(), c=units.c, pulse=pulse
        )
        figures.extend([fig_r, fig_u, fig_w])
        fig_r.savefig(run_dir / 'electron_position_trajectories.png', dpi=180)
        fig_u.savefig(run_dir / 'electron_velocity_trajectories.png', dpi=180)
        fig_w.savefig(run_dir / 'electron_acceleration_trajectories.png', dpi=180)

        savez_kwargs = {
            'omega': screen_geom.omega,
            'grid_x': screen_geom.grid_x,
            'grid_y': screen_geom.grid_y,
            'z_screen': screen_geom.z_screen,
            'F_l': screen_result.F_l,
            'F_s': screen_result.F_s,
            'F_b': screen_result.F_b,
            'F_total': screen_result.F_total,
        }
        if screen_geom.shape_type == 'rectangular':
            savez_kwargs['x'] = screen_geom.x
            savez_kwargs['y'] = screen_geom.y

        np.savez_compressed(run_dir / 'screen_emitted_field.npz', **savez_kwargs)

        screen_meta = {
            'shape_type': screen_geom.shape_type,
            'z_screen_au': screen_geom.z_screen,
            'z_screen_lambda': screen_geom.z_screen / mode.get_lambda(),
            'Nx': screen_geom.Nx,
            'Ny': screen_geom.Ny,
            'N_points': n_points,
            'N_min': int(parameters['screen.N_min']),
            'N_max': int(parameters['screen.N_max']),
            'method': str(parameters.get('screen.method', 'simplified')),
            'N_omega': screen_geom.omega.size,
            'omega_harmonics_au': [float(w) for w in screen_geom.omega],
            'shape': list(screen_result.F_total.shape),
            'wall_time_seconds': wall_sec,
            'time_per_electron_seconds': per_elec_sec,
            'time_per_electron_ms': per_elec_sec * 1000.0 if per_elec_sec is not None else None,
            'time_per_electron_per_point_seconds': per_elec_per_point_sec,
            'time_per_electron_per_point_us': per_elec_per_point_sec * 1e6 if per_elec_per_point_sec is not None else None,
            'num_workers': screen_result.num_workers,
            'cpu_time_per_electron_per_point_seconds': cpu_per_elec_per_point_sec,
            'cpu_time_per_electron_per_point_ms': cpu_per_elec_per_point_sec * 1000.0 if cpu_per_elec_per_point_sec is not None else None,
        }
        if screen_geom.shape_type == 'rectangular':
            screen_meta['width_lambda'] = screen_geom.width / mode.get_lambda()
            screen_meta['height_lambda'] = screen_geom.height / mode.get_lambda()
        else:
            screen_meta['R_min_lambda'] = screen_geom.R_min / mode.get_lambda()
            screen_meta['R_max_lambda'] = screen_geom.R_max / mode.get_lambda()
            screen_meta['Phi_min_rad'] = screen_geom.Phi_min
            screen_meta['Phi_max_rad'] = screen_geom.Phi_max
        metadata['screen_emitted_field'] = screen_meta
        fig_screen, _ = plot_screen_emitted_intensity(screen_result, lambda_scale=mode.get_lambda(), w_0=mode.w_0, pulse=pulse)
        figures.append(fig_screen)
        fig_screen.savefig(run_dir / 'screen_emitted_intensity.png', dpi=180)

        fig_am, _ = plot_screen_angular_momentum_flux_density(screen_result, lambda_scale=mode.get_lambda(), w_0=mode.w_0, pulse=pulse, c=units.c)
        figures.append(fig_am)
        fig_am.savefig(run_dir / 'screen_angular_momentum_flux.png', dpi=180)

        # Generate 18 breakdown figures for each calculated frequency in distinct subfolders
        # Each figure is a 2x2 panel: Real, Imag, Modulus, Phase
        breakdown_figs = generate_all_screen_breakdown_plots(
            screen_result, omega_idx=None, lambda_scale=mode.get_lambda(), unit_label='\\lambda', w_0=mode.w_0, pulse=pulse,
            run_dir=run_dir, close_figs=not show
        )
        figures.extend(breakdown_figs)
        metadata['status'] = 'complete'
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / 'run.json', metadata)
        if show:
            plt.show()
        return run_dir
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(run_dir / 'run.json', metadata)
        raise
    finally:
        for fig in figures:
            plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true', help='Open interactive plot windows')
    args = parser.parse_args()
    main(show=args.show)
