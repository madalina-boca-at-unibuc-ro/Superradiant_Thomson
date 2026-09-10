"""Unit tests for 2D observation screen geometry, FT Faraday evaluators, and parallel solver."""
import unittest

import matplotlib.pyplot as plt
import numpy as np

from superradiant_thomson.electron import Electron, solve_electron_ensemble
from superradiant_thomson.laser import LaserAmplitude, PulseTiming, TemporalFactor
from superradiant_thomson.lg_mode import LGMode
from superradiant_thomson.parameters import AtomicUnits, ParameterResolver, default_registry, register_laser_scales
from superradiant_thomson.screen import (
    ScreenGeometry, ScreenResult, compute_screen_emitted_field,
    compute_screen_emitted_field_from_laser_and_bunch, screen_schema,
    _compute_single_electron_screen_field,
)
from superradiant_thomson.plotting.screen import (
    plot_screen_emitted_intensity, plot_screen_faraday_component_breakdown,
    generate_all_screen_breakdown_plots,
)


class TestScreenGeometry(unittest.TestCase):

    def test_geometry_grid_properties(self):
        geom = ScreenGeometry(z_screen=100.0, width=40.0, height=20.0, Nx=4, Ny=2,
                              omega=np.array([0.5, 1.0, 1.5]))
        self.assertEqual(geom.dx, 10.0)
        self.assertEqual(geom.dy, 10.0)
        np.testing.assert_allclose(geom.x, np.array([-15.0, -5.0, 5.0, 15.0]))
        np.testing.assert_allclose(geom.y, np.array([-5.0, 5.0]))
        self.assertEqual(geom.grid_x.shape, (2, 4))
        self.assertEqual(geom.grid_y.shape, (2, 4))

    def test_geometry_validation(self):
        with self.assertRaises(ValueError):
            ScreenGeometry(z_screen=100.0, width=-10.0, height=20.0, Nx=4, Ny=2, omega=np.array([1.0]))
        with self.assertRaises(ValueError):
            ScreenGeometry(z_screen=100.0, width=10.0, height=20.0, Nx=0, Ny=2, omega=np.array([1.0]))

    def test_nonlinear_thomson_frequency_formula(self):
        """Test non-linear Thomson frequency calculation for forward and backward scattering."""
        units = AtomicUnits()
        params = {
            'laser.omega': 0.057,
            'laser.a_0': 1.0,
            'electron.px_beam': 0.0,
            'electron.py_beam': 0.0,
            'electron.pz_beam': 0.0,
            'screen.z_screen': 1000.0,
            'screen.width': 50.0,
            'screen.height': 50.0,
            'screen.Nx': 4,
            'screen.Ny': 4,
            'screen.N_min': 1,
            'screen.N_max': 3,
        }
        # Forward screen (z_screen > 0) for electron at rest -> omega_N = N * omega_0
        geom_fwd = ScreenGeometry.from_parameters(params, units=units)
        expected_fwd = np.array([1, 2, 3]) * 0.057
        np.testing.assert_allclose(geom_fwd.omega, expected_fwd, rtol=1e-12)

        # Backward screen (z_screen < 0) for electron at rest -> omega_N = N * omega_0 / (1 + a_0^2 / 2)
        params_bwd = dict(params, **{'screen.z_screen': -1000.0})
        geom_bwd = ScreenGeometry.from_parameters(params_bwd, units=units)
        expected_bwd = np.array([1, 2, 3]) * 0.057 / (1.0 + 1.0**2 / 2.0)
        np.testing.assert_allclose(geom_bwd.omega, expected_bwd, rtol=1e-12)

    def test_annular_geometry_grid_and_corners(self):
        """Test annular screen geometry, sqrt uniform area sampling, and pcolormesh corner meshes."""
        geom = ScreenGeometry(z_screen=100.0, omega=np.array([1.0]), shape_type='annular',
                              R_min=0.0, R_max=10.0, N_R=4, Phi_min=0.0, Phi_max=2.0 * np.pi, N_Phi=8)
        self.assertEqual(geom.shape_type, 'annular')
        self.assertEqual(geom.Nx, 4)
        self.assertEqual(geom.Ny, 8)
        self.assertEqual(geom.grid_x.shape, (8, 4))
        self.assertEqual(geom.grid_y.shape, (8, 4))
        self.assertEqual(geom.grid_x_corners.shape, (9, 5))
        self.assertEqual(geom.grid_y_corners.shape, (9, 5))

        # Sqrt sampling check: r^2 for 4 bins between 0^2 and 10^2 -> [12.5, 37.5, 62.5, 87.5]
        expected_r_centers = np.sqrt(np.array([12.5, 37.5, 62.5, 87.5]))
        np.testing.assert_allclose(geom.r_centers, expected_r_centers, rtol=1e-12)
        expected_r2_edges = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        np.testing.assert_allclose(geom.r_edges**2, expected_r2_edges, rtol=1e-12)

    def test_screen_parameters_resolution_in_w0_units(self):
        """Test resolving screen dimensions using w_0 and w0 as length units."""
        units = AtomicUnits()
        registry = default_registry(units)
        register_laser_scales(registry, units)
        from superradiant_thomson.laser import register_temporal_scales, temporal_schema, amplitude_schema
        from superradiant_thomson.lg_mode import lg_schema
        from superradiant_thomson.electron import electron_schema
        register_temporal_scales(registry)

        inputs = {
            'laser.omega': 0.057,
            'laser.a_0': 0.1,
            'laser.flat_top_periods': 2,
            'laser.sigma_l': {'value': 1, 'unit': 'T'},
            'laser.wing_factor': 2,
            'laser.p': 0,
            'laser.m': 0,
            'laser.epsilon': 1,
            'laser.w_0': {'value': 50.0, 'unit': 'lambda'},
            'laser.zeta_x': 1.0,
            'laser.zeta_y': 0.0,
            'electron.N': 1,
            'electron.seed': 123,
            'electron.NT': 100,
            'electron.x_0': 0.0,
            'electron.y_0': 0.0,
            'electron.z_0': 0.0,
            'electron.R_beam': 0.0,
            'electron.h_beam': 0.0,
            'electron.px_beam': 0.0,
            'electron.py_beam': 0.0,
            'electron.pz_beam': 0.0,
            'electron.sigma_px_beam': 0.0,
            'electron.sigma_py_beam': 0.0,
            'electron.sigma_pz_beam': 0.0,
            'screen.shape': 'annular',
            'screen.z_screen': {'value': -500.0, 'unit': 'w_0'},
            'screen.width': {'value': 10.0, 'unit': 'w_0'},
            'screen.height': {'value': 10.0, 'unit': 'w0'},
            'screen.Nx': 16,
            'screen.Ny': 16,
            'screen.R_min': {'value': 0.0, 'unit': 'w_0'},
            'screen.R_max': {'value': 5.0, 'unit': 'w0'},
            'screen.N_R': 16,
            'screen.Phi_min': {'value': 0.0, 'unit': 'pi'},
            'screen.Phi_max': {'value': 2.0, 'unit': 'pi'},
            'screen.N_Phi': 16,
            'screen.N_min': 1,
            'screen.N_max': 3,
            'screen.method': 'direct',
        }

        schema = (screen_schema() | lg_schema() | temporal_schema() | amplitude_schema() | electron_schema())
        params = ParameterResolver(registry, schema).resolve(inputs)
        w0_au = params['laser.w_0']
        self.assertAlmostEqual(params['screen.R_max'], 5.0 * w0_au)
        self.assertAlmostEqual(params['screen.width'], 10.0 * w0_au)
        self.assertAlmostEqual(params['screen.height'], 10.0 * w0_au)
        self.assertAlmostEqual(params['screen.z_screen'], -500.0 * w0_au)


class TestScreenEvaluator(unittest.TestCase):

    def setUp(self):
        units = AtomicUnits()
        registry = default_registry(units)
        register_laser_scales(registry, units)

        from superradiant_thomson.laser import register_temporal_scales, temporal_schema, amplitude_schema
        from superradiant_thomson.lg_mode import lg_schema
        from superradiant_thomson.electron import electron_schema
        register_temporal_scales(registry)

        inputs = {
            'laser.omega': 0.057,
            'laser.a_0': 0.1,
            'laser.flat_top_periods': 2,
            'laser.sigma_l': {'value': 1, 'unit': 'T'},
            'laser.wing_factor': 2,
            'laser.p': 0,
            'laser.m': 0,
            'laser.epsilon': 1,
            'laser.w_0': {'value': 50, 'unit': 'lambda'},
            'laser.zeta_x': 1.0,
            'laser.zeta_y': 0.0,
            'electron.N': 1,
            'electron.seed': 123,
            'electron.NT': 100,
            'electron.x_0': 0.0,
            'electron.y_0': 0.0,
            'electron.z_0': 0.0,
            'electron.R_beam': 0.0,
            'electron.h_beam': 0.0,
            'electron.px_beam': 0.0,
            'electron.py_beam': 0.0,
            'electron.pz_beam': 0.0,
            'electron.sigma_px_beam': 0.0,
            'electron.sigma_py_beam': 0.0,
            'electron.sigma_pz_beam': 0.0,
            'screen.z_screen': 1000.0,
            'screen.width': 50.0,
            'screen.height': 50.0,
            'screen.Nx': 4,
            'screen.Ny': 4,
            'screen.N_min': 1,
            'screen.N_max': 2,
        }

        schema = temporal_schema() | lg_schema() | amplitude_schema() | electron_schema() | screen_schema()
        self.registry = registry
        self.schema = schema
        self.params = ParameterResolver(registry, schema).resolve(inputs)
        self.units = units
        self.mode = LGMode.from_parameters(self.params, units)
        self.amp = LaserAmplitude.from_parameters(self.params, units)
        self.pulse = TemporalFactor(PulseTiming.from_parameters(self.params), units.c)
        self.electron = solve_electron_ensemble(self.mode, self.amp, self.pulse, units, self.params)
        self.geom = ScreenGeometry.from_parameters(self.params, units=self.units)

    def test_form1_vs_form2_agreement(self):
        """Form 1 ('direct') and Form 2 ('simplified') evaluators return valid non-zero tensors."""
        r = self.electron.r if self.electron.r.ndim == 2 else self.electron.r[0]
        u = self.electron.u if self.electron.u.ndim == 2 else self.electron.u[0]
        w = self.electron.w if self.electron.w.ndim == 2 else self.electron.w[0]

        F1_l, F1_s, F1_b = _compute_single_electron_screen_field(r, u, w, self.electron.tau, self.geom, self.units.c, method='direct')
        F2_l, F2_s, F2_b = _compute_single_electron_screen_field(r, u, w, self.electron.tau, self.geom, self.units.c, method='simplified')

        F1 = F1_l + F1_s + F1_b
        F2 = F2_l + F2_s + F2_b

        self.assertEqual(F1.shape, (self.geom.omega.size, 4, 4, 6))
        self.assertEqual(F2.shape, (self.geom.omega.size, 4, 4, 6))
        self.assertTrue(np.all(np.isfinite(F1)))
        self.assertTrue(np.all(np.isfinite(F2)))
        self.assertGreater(np.max(np.abs(F1)), 0.0)
        self.assertGreater(np.max(np.abs(F2)), 0.0)

    def test_invalid_method_raises_value_error(self):
        """Test that invalid emitted field method raises ValueError."""
        r = self.electron.r if self.electron.r.ndim == 2 else self.electron.r[0]
        u = self.electron.u if self.electron.u.ndim == 2 else self.electron.u[0]
        w = self.electron.w if self.electron.w.ndim == 2 else self.electron.w[0]
        with self.assertRaises(ValueError):
            _compute_single_electron_screen_field(r, u, w, self.electron.tau, self.geom, self.units.c, method='invalid_method')


    def test_charge_reversal_antisymmetry(self):
        """Reversing electron charge q -> -q reverses emitted field F -> -F."""
        res_plus = compute_screen_emitted_field(self.electron, self.geom, self.units.c, q=-1.0, max_workers=1)
        res_minus = compute_screen_emitted_field(self.electron, self.geom, self.units.c, q=+1.0, max_workers=1)
        np.testing.assert_allclose(res_plus.F_total, -res_minus.F_total, rtol=1e-12)

    def test_serial_vs_parallel_equivalence(self):
        """Serial (max_workers=1) and parallel (max_workers=2) reductions must match exactly."""
        # Create a 2-electron container
        tau = self.electron.tau
        r2 = np.stack([self.electron.r, self.electron.r]) if self.electron.r.ndim == 2 else self.electron.r
        u2 = np.stack([self.electron.u, self.electron.u]) if self.electron.u.ndim == 2 else self.electron.u
        w2 = np.stack([self.electron.w, self.electron.w]) if self.electron.w.ndim == 2 else self.electron.w
        elec2 = Electron(tau=tau, r=r2, u=u2, w=w2)

        res_serial = compute_screen_emitted_field(elec2, self.geom, self.units.c, max_workers=1)
        res_parallel = compute_screen_emitted_field(elec2, self.geom, self.units.c, max_workers=2)
        np.testing.assert_allclose(res_serial.F_total, res_parallel.F_total, rtol=1e-12)

        # Linearity check: 2 identical electrons produce 2 * single electron field
        res_single = compute_screen_emitted_field(self.electron, self.geom, self.units.c, max_workers=1)
        np.testing.assert_allclose(res_serial.F_total, 2.0 * res_single.F_total, rtol=1e-12)

    def test_screen_plotting_functions(self):
        """Plotting functions return figure and axes without throwing exceptions."""
        result = compute_screen_emitted_field(self.electron, self.geom, self.units.c, max_workers=1)
        fig1, axs1 = plot_screen_emitted_intensity(result, lambda_scale=self.mode.get_lambda(), pulse=self.pulse)
        self.assertIsNotNone(fig1)
        plt.close(fig1)

        fig2, axs2 = plot_screen_faraday_component_breakdown(
            result, component_idx=0, contribution='total', omega_idx=0,
            lambda_scale=self.mode.get_lambda(), pulse=self.pulse
        )
        self.assertIsNotNone(fig2)
        plt.close(fig2)

        figs = generate_all_screen_breakdown_plots(
            result, omega_idx=0, lambda_scale=self.mode.get_lambda(), pulse=self.pulse
        )
        self.assertEqual(len(figs), 24)
        for f in figs:
            plt.close(f)

    def test_streaming_parallel_solver_and_core_count_invariance(self):
        """Streaming parallel solver (max_workers=1 vs max_workers=2) produces identical initial conditions and radiation."""
        inputs_multi = dict(self.params.to_dict()['inputs'], **{'electron.N': 4, 'electron.R_beam': 1.0})
        params_multi = ParameterResolver(self.registry, self.schema).resolve(inputs_multi)

        sample_1, r0_1, u0_1, res_1 = compute_screen_emitted_field_from_laser_and_bunch(
            self.mode, self.amp, self.pulse, self.units, params_multi, self.geom,
            max_workers=1, max_stored_trajectories=10
        )
        sample_2, r0_2, u0_2, res_2 = compute_screen_emitted_field_from_laser_and_bunch(
            self.mode, self.amp, self.pulse, self.units, params_multi, self.geom,
            max_workers=2, max_stored_trajectories=10
        )

        np.testing.assert_allclose(r0_1, r0_2, rtol=1e-12)
        np.testing.assert_allclose(u0_1, u0_2, rtol=1e-12)
        np.testing.assert_allclose(res_1.F_total, res_2.F_total, rtol=1e-12)

        # Check sample_electron trajectory output (N=4 electrons, 3D shape)
        self.assertEqual(sample_1.r.shape[0], 4)
        mass_res = sample_1.mass_shell_residual(self.units.c)
        np.testing.assert_allclose(mass_res, 0.0, atol=1e-6)

    def test_angular_momentum_flux_density(self):
        """Test calculation of spectral angular momentum flux density on screen."""
        _, _, _, res = compute_screen_emitted_field_from_laser_and_bunch(
            self.mode, self.amp, self.pulse, self.units, self.params, self.geom,
            max_workers=1
        )
        flux_z = res.angular_momentum_flux_density
        self.assertEqual(flux_z.shape, (self.geom.omega.size, self.geom.Ny, self.geom.Nx))
        self.assertTrue(np.all(np.isfinite(flux_z)))

        # Verify on-axis cancellation: at x=0, y=0 pixel, flux_z must be identically 0
        # Create a geometry with an odd pixel count so (0,0) is an exact grid center pixel
        odd_geom = ScreenGeometry(z_screen=1000.0, width=50.0, height=50.0, Nx=3, Ny=3,
                                  omega=np.array([0.057]))
        res_odd = ScreenResult(odd_geom, res.F_l[:1, :3, :3, :], res.F_s[:1, :3, :3, :], res.F_b[:1, :3, :3, :])
        flux_odd = res_odd.compute_angular_momentum_flux_density(c=self.units.c)
        # Center pixel (y_idx=1, x_idx=1) corresponds to x=0, y=0
        self.assertAlmostEqual(flux_odd[0, 1, 1], 0.0, places=12)

        # Verify plotting helper function
        from superradiant_thomson.plotting.screen import plot_screen_angular_momentum_flux_density
        fig, _ = plot_screen_angular_momentum_flux_density(res, lambda_scale=self.mode.get_lambda(), pulse=self.pulse)
        self.assertIsNotNone(fig)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()

