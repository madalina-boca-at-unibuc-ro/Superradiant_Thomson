import unittest
import numpy as np

from superradiant_thomson.four_vector import (
    minkowski_dot, minkowski_norm_sq, lower_index, raise_index,
    wedge_4d, four_position, four_momentum, four_velocity, four_acceleration,
)
from superradiant_thomson.parameters import AtomicUnits


class FourVectorTests(unittest.TestCase):
    def setUp(self):
        self.c = AtomicUnits().c

    def test_minkowski_dot_and_norms(self):
        a = np.array([5.0, 1.0, 2.0, 3.0])
        b = np.array([2.0, 0.5, 1.0, 1.5])

        # a . b = 5*2 - (1*0.5 + 2*1 + 3*1.5) = 10 - (0.5 + 2 + 4.5) = 10 - 7 = 3
        self.assertAlmostEqual(minkowski_dot(a, b), 3.0)
        # a . a = 25 - (1 + 4 + 9) = 11
        self.assertAlmostEqual(minkowski_norm_sq(a), 11.0)

    def test_lower_and_raise_index(self):
        a = np.array([4.0, 1.5, -2.5, 3.0])
        a_lower = lower_index(a)
        np.testing.assert_allclose(a_lower, [4.0, -1.5, 2.5, -3.0])

        a_raised = raise_index(a_lower)
        np.testing.assert_allclose(a_raised, a)

    def test_wedge_4d(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])

        w = wedge_4d(a, b)
        self.assertEqual(w.shape, (4, 4))
        # Anti-symmetry: w[i,j] == -w[j,i]
        np.testing.assert_allclose(w, -w.T)
        self.assertAlmostEqual(w[0, 1], 1.0 * 6.0 - 2.0 * 5.0)  # -4.0
        self.assertAlmostEqual(w[1, 2], 2.0 * 7.0 - 3.0 * 6.0)  # -4.0

    def test_four_position_and_momentum_kinematics(self):
        ct = 10.0
        r = np.array([1.0, 2.0, 3.0])
        x = four_position(ct, r)
        np.testing.assert_allclose(x, [10.0, 1.0, 2.0, 3.0])

        m = 1.0
        p_spatial = np.array([3.0, 4.0, 0.0])  # |p| = 5
        p = four_momentum(p_spatial, m=m, c=self.c)
        # Mass-shell identity: p . p = m^2 c^2
        np.testing.assert_allclose(minkowski_norm_sq(p), (m * self.c)**2, rtol=1e-12)

        # 4-velocity normalization: u . u = c^2
        u = four_velocity(p, m=m)
        np.testing.assert_allclose(minkowski_norm_sq(u), self.c**2, rtol=1e-12)

    def test_four_acceleration_orthogonality(self):
        # Lorentz force w^\mu = (q/m^2) F^{\mu\nu} u_\nu
        # Since F^{\mu\nu} is anti-symmetric, u^\mu w_\mu = 0
        m, q, c = 1.0, -1.0, self.c
        p_spatial = np.array([10.0, -5.0, 12.0])
        p = four_momentum(p_spatial, m=m, c=c)
        u = four_velocity(p, m=m)

        # Construct a test anti-symmetric F tensor
        Ex, Ey, Ez = 0.05, -0.02, 0.01
        Bx, By, Bz = 0.001, -0.003, 0.002
        Ex_c, Ey_c, Ez_c = Ex / c, Ey / c, Ez / c
        F = np.array([
            [0.0, -Ex_c, -Ey_c, -Ez_c],
            [Ex_c, 0.0, -Bz, By],
            [Ey_c, Bz, 0.0, -Bx],
            [Ez_c, -By, Bx, 0.0],
        ])

        w = four_acceleration(F, u, q=q, m=m)
        # Orthogonality: u . w = 0
        self.assertAlmostEqual(minkowski_dot(u, w), 0.0, places=10)

    def test_batching_dimensions(self):
        # Test 2D batch shape (N_particles, 4)
        N = 100
        p_spatial = np.random.randn(N, 3)
        p = four_momentum(p_spatial, m=1.0, c=self.c)
        self.assertEqual(p.shape, (N, 4))

        p_norm = minkowski_norm_sq(p)
        self.assertEqual(p_norm.shape, (N,))
        np.testing.assert_allclose(p_norm, self.c**2, rtol=1e-12)
