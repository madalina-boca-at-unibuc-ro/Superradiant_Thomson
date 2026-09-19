"""Unit tests for incident laser beam screen evaluation and plotting at z=0."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from superradiant_thomson.lg_mode import LGMode, compute_incident_laser_screen_field
from superradiant_thomson.laser import LaserAmplitude, PulseTiming, TemporalFactor
from superradiant_thomson.parameters import AtomicUnits, ParameterResolver, default_registry, register_laser_scales
from superradiant_thomson.screen import ScreenGeometry, ScreenResult
from superradiant_thomson.plotting import generate_incident_laser_screen_plots


class TestIncidentLaserScreen(unittest.TestCase):
    def setUp(self):
        self.units = AtomicUnits()
        self.mode = LGMode(p=1, m=1, epsilon=1, w_0=75.0, omega=0.057, c=self.units.c)
        self.amplitude = LaserAmplitude(a_0=0.1, omega=0.057, c=self.units.c)
        self.pulse = TemporalFactor(
            PulseTiming(omega=0.057, flat_top_periods=10, sigma_l=2.0 * (2 * np.pi / 0.057), wing_factor=5.0),
            self.units.c
        )
        self.geom_annular = ScreenGeometry(
            shape_type='annular', z_screen=25000.0,
            R_min=0.0, R_max=200.0, N_R=8, Phi_min=0.0, Phi_max=2 * np.pi, N_Phi=8,
            omega=np.array([0.057]), harmonics=np.array([1])
        )
        self.geom_rect = ScreenGeometry(
            z_screen=25000.0, width=400.0, height=400.0, Nx=8, Ny=8,
            omega=np.array([0.057]), harmonics=np.array([1]), shape_type='rectangular'
        )

    def test_compute_incident_laser_screen_field_annular(self):
        res = compute_incident_laser_screen_field(self.mode, self.amplitude, self.geom_annular, self.units.c)
        self.assertIsInstance(res, ScreenResult)
        self.assertEqual(res.geometry.z_screen, 0.0)
        self.assertEqual(res.F_total.shape, (1, 8, 8, 6))
        self.assertTrue(np.all(np.isfinite(res.F_total)))

    def test_compute_incident_laser_screen_field_rectangular(self):
        res = compute_incident_laser_screen_field(self.mode, self.amplitude, self.geom_rect, self.units.c)
        self.assertIsInstance(res, ScreenResult)
        self.assertEqual(res.geometry.z_screen, 0.0)
        self.assertEqual(res.F_total.shape, (1, 8, 8, 6))
        self.assertTrue(np.all(np.isfinite(res.F_total)))

    def test_generate_incident_laser_screen_plots(self):
        res = compute_incident_laser_screen_field(self.mode, self.amplitude, self.geom_annular, self.units.c)
        with tempfile.TemporaryDirectory() as tmpdir:
            figs = generate_incident_laser_screen_plots(
                res, c=self.units.c, lambda_scale=self.mode.get_lambda(),
                w_0=self.mode.w_0, pulse=self.pulse, run_dir=tmpdir, close_figs=True
            )
            self.assertEqual(figs, [])
            subdirs = sorted(p for p in Path(tmpdir).iterdir() if p.is_dir())
            self.assertEqual(len(subdirs), 2)  # breakdown and observables subfolders
            breakdown_subdir = [p for p in subdirs if 'breakdown' in p.name][0]
            observables_subdir = [p for p in subdirs if 'observables' in p.name][0]
            self.assertEqual(len(list(breakdown_subdir.glob('*.png'))), 6)
            self.assertEqual(len(list(observables_subdir.glob('*.png'))), 8)


if __name__ == '__main__':
    unittest.main()
