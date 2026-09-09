"""Unit tests for Electron dataclass, distribution generators, and trajectory ODE solver."""
import unittest
import numpy as np

from main import initialize, INPUTS
from superradiant_thomson.parameters import AtomicUnits
from superradiant_thomson.laser import (
    PulseTiming, TemporalFactor, LaserAmplitude,
)
from superradiant_thomson.lg_mode import LGMode
from superradiant_thomson.electron import (
    Electron, electron_schema, generate_electron_initial_conditions,
    solve_electron_trajectory, solve_electron_ensemble,
)


class ElectronTests(unittest.TestCase):
    def setUp(self):
        units = AtomicUnits()
        self.units, self.parameters, self.pulse = initialize(dict(INPUTS, **{
            'laser.p': 0,
            'laser.m': 0,
            'laser.omega': 0.057,
            'laser.flat_top_periods': 2,
            'laser.sigma_l': 200.0,
            'laser.wing_factor': 3.0,
            'laser.a_0': 1.0,
            'electron.N': 1,
            'electron.seed': 42,
            'electron.NT': 50,
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
        }))
        self.c = self.units.c
        self.mode = LGMode.from_parameters(self.parameters, self.units)
        self.timing = PulseTiming.from_parameters(self.parameters)
        self.amplitude = LaserAmplitude.from_parameters(self.parameters, self.units)

    def test_electron_dataclass_validation(self):
        tau = np.linspace(0.0, 10.0, 5)
        r = np.zeros((5, 4))
        u = np.ones((5, 4))
        w = np.zeros((5, 4))

        e = Electron(tau=tau, r=r, u=u, w=w)
        self.assertEqual(e.tau.shape, (5,))
        self.assertEqual(e.r.shape, (5, 4))

        # Test multi-electron 3D shapes
        r_3d = np.zeros((3, 5, 4))
        u_3d = np.ones((3, 5, 4))
        w_3d = np.zeros((3, 5, 4))
        e3d = Electron(tau=tau, r=r_3d, u=u_3d, w=w_3d)
        self.assertEqual(e3d.r.shape, (3, 5, 4))

        # Test invalid shapes
        with self.assertRaises(ValueError):
            Electron(tau=tau, r=np.zeros((4, 4)), u=u, w=w)

        # Test non-finite input
        with self.assertRaises(ValueError):
            u_bad = u.copy()
            u_bad[0, 0] = np.nan
            Electron(tau=tau, r=r, u=u_bad, w=w)

    def test_cylinder_spatial_distribution(self):
        units, params, _ = initialize(dict(INPUTS, **{
            'electron.N': 1000,
            'electron.seed': 123,
            'electron.x_0': 10.0,
            'electron.y_0': -5.0,
            'electron.z_0': 2.0,
            'electron.R_beam': {'value': 5.0, 'unit': 'w_0'},
            'electron.h_beam': {'value': 4.0, 'unit': 'w_0'},
        }))
        r0, _ = generate_electron_initial_conditions(params, units)
        self.assertEqual(r0.shape, (1000, 4))

        x_vals, y_vals, z_vals = r0[:, 1], r0[:, 2], r0[:, 3]
        R_beam_au = params['electron.R_beam']
        h_beam_au = params['electron.h_beam']

        # Radial bound test
        radii = np.sqrt((x_vals - 10.0)**2 + (y_vals - (-5.0))**2)
        self.assertTrue(np.all(radii <= R_beam_au + 1e-12))

        # Height bound test
        self.assertTrue(np.all(np.abs(z_vals - 2.0) <= 0.5 * h_beam_au + 1e-12))

    def test_gaussian_momentum_distribution(self):
        units, params, _ = initialize(dict(INPUTS, **{
            'electron.N': 10000,
            'electron.seed': 456,
            'electron.px_beam': {'value': 0.1, 'unit': 'c'},
            'electron.py_beam': {'value': -0.05, 'unit': 'c'},
            'electron.pz_beam': {'value': 0.2, 'unit': 'c'},
            'electron.sigma_px_beam': {'value': 0.01, 'unit': 'c'},
            'electron.sigma_py_beam': {'value': 0.0, 'unit': 'c'},  # Zero-width fixed value
            'electron.sigma_pz_beam': {'value': 0.02, 'unit': 'c'},
        }))
        _, u0 = generate_electron_initial_conditions(params, units)
        px, py, pz = u0[:, 1], u0[:, 2], u0[:, 3]

        px_target = 0.1 * units.c
        py_target = -0.05 * units.c
        pz_target = 0.2 * units.c
        sig_x_target = 0.01 * units.c
        sig_z_target = 0.02 * units.c

        # Means match expected values
        self.assertAlmostEqual(np.mean(px), px_target, delta=0.01 * units.c)
        np.testing.assert_allclose(py, py_target, rtol=1e-12)  # Fixed zero width
        self.assertAlmostEqual(np.mean(pz), pz_target, delta=0.01 * units.c)

        # Standard deviations match expected values
        self.assertAlmostEqual(np.std(px), sig_x_target, delta=0.005 * units.c)
        self.assertAlmostEqual(np.std(pz), sig_z_target, delta=0.005 * units.c)

        # 4-velocity norm u . u = c^2 exactly for all generated electrons
        norm_sq = u0[:, 0]**2 - (px**2 + py**2 + pz**2)
        np.testing.assert_allclose(norm_sq, units.c**2, rtol=1e-12)

    def test_seed_reproducibility(self):
        units, params, _ = initialize(dict(INPUTS, **{
            'electron.N': 50,
            'electron.seed': 999,
            'electron.R_beam': {'value': 2.0, 'unit': 'w_0'},
            'electron.sigma_px_beam': {'value': 0.05, 'unit': 'c'},
        }))
        r0_a, u0_a = generate_electron_initial_conditions(params, units)
        r0_b, u0_b = generate_electron_initial_conditions(params, units)

        np.testing.assert_array_equal(r0_a, r0_b)
        np.testing.assert_array_equal(u0_a, u0_b)

    def test_free_particle_motion(self):
        # Set laser amplitude a_0 = 0 for free motion
        _, params, _ = initialize(dict(INPUTS, **{'laser.a_0': 0.0, 'electron.N': 1}))
        amp_free = LaserAmplitude.from_parameters(params, self.units)

        electron = solve_electron_trajectory(self.mode, amp_free, self.pulse, self.units, params)

        # 4-acceleration should be zero everywhere
        np.testing.assert_allclose(electron.w, 0.0, atol=1e-12)

        # 4-velocity should be constant
        u0 = electron.u[0]
        np.testing.assert_allclose(electron.u, np.tile(u0, (electron.u.shape[0], 1)), rtol=1e-10)

        # 4-position should be linear in proper time r(tau) = r0 + u0 * tau
        r_expected = electron.r[0] + np.outer(electron.tau, u0)
        np.testing.assert_allclose(electron.r, r_expected, rtol=1e-8, atol=1e-10)

    def test_laser_field_trajectory_conservation(self):
        electron = solve_electron_trajectory(self.mode, self.amplitude, self.pulse, self.units, self.parameters)

        # Grid size N_tau check
        duration = self.pulse.timing.duration
        period = self.pulse.timing.period
        expected_points = int(round((duration / period) * 50)) + 1
        self.assertEqual(electron.tau.size, expected_points)

        # Mass shell residual u . u = c^2
        mass_shell_res = electron.mass_shell_residual(self.c)
        np.testing.assert_allclose(mass_shell_res, 0.0, atol=1e-7)

        # Orthogonality residual u . w = 0
        ortho_res = electron.acceleration_orthogonality_residual(self.c)
        np.testing.assert_allclose(ortho_res, 0.0, atol=1e-6)

    def test_multi_electron_ensemble_trajectory(self):
        units, params, pulse = initialize(dict(INPUTS, **{
            'electron.N': 3,
            'electron.seed': 77,
            'electron.NT': 20,
            'electron.R_beam': {'value': 0.1, 'unit': 'w_0'},
            'electron.sigma_px_beam': {'value': 0.01, 'unit': 'c'},
        }))
        mode = LGMode.from_parameters(params, units)
        amp = LaserAmplitude.from_parameters(params, units)

        electron = solve_electron_ensemble(mode, amp, pulse, units, params)

        self.assertEqual(electron.r.shape[0], 3)
        self.assertEqual(electron.u.shape[0], 3)
        self.assertEqual(electron.w.shape[0], 3)

        # Mass shell residual holds for all 3 electrons across time
        mass_res = electron.mass_shell_residual(units.c)
        self.assertEqual(mass_res.shape, (3, electron.tau.size))
        np.testing.assert_allclose(mass_res, 0.0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
