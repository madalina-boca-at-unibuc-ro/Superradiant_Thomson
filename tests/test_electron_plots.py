"""Unit tests for electron initial distribution 3D and 4-vector trajectory plotting functions."""
import unittest
import numpy as np
import matplotlib.pyplot as plt

from superradiant_thomson.parameters import AtomicUnits
from superradiant_thomson.electron import Electron
from superradiant_thomson.plotting import (
    plot_electron_initial_positions,
    plot_electron_initial_momenta,
    plot_electron_initial_distribution,
    plot_electron_position_trajectories,
    plot_electron_velocity_trajectories,
    plot_electron_acceleration_trajectories,
    plot_electron_ensemble_trajectories,
)


class ElectronPlotTests(unittest.TestCase):
    def setUp(self):
        self.c = AtomicUnits().c
        self.w_0 = 100.0

    def tearDown(self):
        plt.close('all')

    def test_initial_positions_plot(self):
        # Test 2D shape (N, 4)
        r0 = np.array([
            [0.0, 10.0, -20.0, 5.0],
            [0.0, -15.0, 25.0, -10.0],
        ])
        fig, ax = plot_electron_initial_positions(r0, w_0=self.w_0)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertIn('w_0', ax.get_xlabel())

        # Test 1D shape (4,) single electron
        fig1, ax1 = plot_electron_initial_positions(r0[0], w_0=self.w_0)
        self.assertIsNotNone(fig1)

    def test_initial_momenta_plot(self):
        # Test 3D shape (N, N_tau, 4)
        u0 = np.zeros((3, 10, 4))
        u0[:, 0, 0] = self.c
        u0[:, 0, 1] = np.array([0.1, -0.2, 0.05]) * self.c

        fig, ax = plot_electron_initial_momenta(u0, c=self.c)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertIn('m_e c', ax.get_xlabel())

    def test_initial_distribution_2panel_plot(self):
        r0 = np.random.randn(20, 4)
        u0 = np.random.randn(20, 4)
        u0[:, 0] = np.sqrt(self.c**2 + np.sum(u0[:, 1:]**2, axis=-1))

        fig, (ax_pos, ax_mom) = plot_electron_initial_distribution(r0, u0, w_0=self.w_0, c=self.c)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax_pos)
        self.assertIsNotNone(ax_mom)
        self.assertIn('Initial Electron Positions', ax_pos.get_title())
        self.assertIn('Initial Electron Momenta', ax_mom.get_title())

    def test_trajectory_4panel_plots(self):
        tau = np.linspace(0.0, 100.0, 50)
        r = np.random.randn(12, 50, 4)
        u = np.random.randn(12, 50, 4)
        u[:, :, 0] = self.c
        w = np.random.randn(12, 50, 4)

        electron = Electron(tau=tau, r=r, u=u, w=w)

        fig_r, axs_r = plot_electron_position_trajectories(electron, max_electrons=10, w_0=self.w_0)
        self.assertEqual(axs_r.shape, (2, 2))

        fig_r_lam, axs_r_lam = plot_electron_position_trajectories(electron, max_electrons=10, lambda_scale=15000.0)
        self.assertIn(r'\lambda', axs_r_lam.flat[0].get_ylabel())

        fig_u, axs_u = plot_electron_velocity_trajectories(electron, max_electrons=5, c=self.c)
        self.assertEqual(axs_u.shape, (2, 2))

        fig_w, axs_w = plot_electron_acceleration_trajectories(electron, max_electrons=10, c=self.c)
        self.assertEqual(axs_w.shape, (2, 2))

        fig_pos, fig_vel, fig_acc = plot_electron_ensemble_trajectories(electron, max_electrons=10, lambda_scale=15000.0, c=self.c)
        self.assertIsNotNone(fig_pos)
        self.assertIsNotNone(fig_vel)
        self.assertIsNotNone(fig_acc)


if __name__ == '__main__':
    unittest.main()
