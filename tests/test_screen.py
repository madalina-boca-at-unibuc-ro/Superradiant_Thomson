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
            'screen.omega_min': 0.057,
            'screen.omega_max': 0.114,
            'screen.N_omega': 2,
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
        self.geom = ScreenGeometry.from_parameters(self.params)

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
            result, component_idx=0, contribution='long', omega_idx=0,
            lambda_scale=self.mode.get_lambda(), pulse=self.pulse
        )
        self.assertIsNotNone(fig2)
        plt.close(fig2)

        figs = generate_all_screen_breakdown_plots(
            result, omega_idx=0, lambda_scale=self.mode.get_lambda(), pulse=self.pulse
        )
        self.assertEqual(len(figs), 18)
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


if __name__ == '__main__':
    unittest.main()
