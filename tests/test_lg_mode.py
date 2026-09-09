import unittest
import numpy as np
from scipy.special import hyp1f1, factorial
from superradiant_thomson.lg_mode import LGMode
from superradiant_thomson.parameters import AtomicUnits


class LGTests(unittest.TestCase):
    def mode(self, p=0, m=0, epsilon=1):
        return LGMode(p, m, epsilon, 30., 2., AtomicUnits().c)

    def test_fundamental_and_geometry(self):
        mode = self.mode()
        x = np.array([0, 10, 30])
        np.testing.assert_allclose(mode(x, 0, 0), np.sqrt(2)*np.exp(-(x/30)**2))
        self.assertAlmostEqual(mode.spot_size(mode.z_R), 30*np.sqrt(2))
        self.assertAlmostEqual(mode.gouy_phase(-mode.z_R), -np.pi/4)

    def test_independent_hypergeometric_formula(self):
        for p, m in [(0, 0), (0, 1), (2, 3), (3, 0)]:
            for epsilon in (-1, 1):
                mode = self.mode(p, m, epsilon)
                x, y, z = 8., -11., -0.7*mode.z_R
                w = mode.spot_size(z)
                norm = np.sqrt(2)/factorial(m)*np.sqrt(factorial(p+m)/factorial(p))
                expected = (norm*30/w*(np.sqrt(2)/w)**m
                            *np.exp(-1j*(2*p+m+1)*np.arctan(z/mode.z_R))
                            *np.exp(-mode.k/2*(x*x+y*y)/(mode.z_R+1j*z))
                            *hyp1f1(-p,m+1,2*(x*x+y*y)/w**2)*(x+1j*epsilon*y)**m)
                np.testing.assert_allclose(mode(x,y,z), expected, rtol=1e-12)

    def test_derivatives_including_axis(self):
        for p in (0, 1, 3):
            for m in (0, 1, 2, 4):
                for epsilon in (-1, 1):
                    mode = self.mode(p,m,epsilon)
                    for x,y in [(0.,0.), (7.,-9.)]:
                        z, h = 0.4*mode.z_R, 1e-4
                        _, ux, uy = mode.evaluate(x,y,z)
                        dx = (mode(x+h,y,z)-mode(x-h,y,z))/(2*h)
                        dy = (mode(x,y+h,z)-mode(x,y-h,z))/(2*h)
                        np.testing.assert_allclose([ux,uy], [dx,dy], rtol=2e-7, atol=2e-10)

    def test_broadcast_and_winding(self):
        mode = self.mode(1,2,-1)
        x = np.array([1,4,8])[:,None]
        z = np.array([0,mode.z_R])[None,:]
        self.assertEqual(mode(x,3,z).shape, (3,2))
        theta = 0.3
        np.testing.assert_allclose(mode(10*np.cos(theta),10*np.sin(theta),0),
                                   mode(10,0,0)*np.exp(-2j*theta))

    def test_polarization_normalization(self):
        # Default linear polarization.
        mode1 = LGMode(0, 0, 1, 30., 2., 137., zeta_x=1.0, zeta_y=0.0)
        self.assertAlmostEqual(mode1.zeta_x, 1.0)
        self.assertAlmostEqual(mode1.zeta_y, 0.0)
        self.assertAlmostEqual(abs(mode1.zeta_x)**2 + abs(mode1.zeta_y)**2, 1.0)

        # Unnormalized real inputs (3, 4) -> (0.6, 0.8).
        mode2 = LGMode(0, 0, 1, 30., 2., 137., zeta_x=3.0, zeta_y=4.0)
        self.assertAlmostEqual(mode2.zeta_x, 0.6)
        self.assertAlmostEqual(mode2.zeta_y, 0.8)
        self.assertAlmostEqual(abs(mode2.zeta_x)**2 + abs(mode2.zeta_y)**2, 1.0)

        # Complex circular polarization (1, 1j) -> (1/sqrt(2), 1j/sqrt(2)).
        mode3 = LGMode(0, 0, 1, 30., 2., 137., zeta_x=1.0, zeta_y=1.0j)
        self.assertAlmostEqual(mode3.zeta_x, 1 / np.sqrt(2))
        self.assertAlmostEqual(mode3.zeta_y, 1j / np.sqrt(2))
        self.assertAlmostEqual(abs(mode3.zeta_x)**2 + abs(mode3.zeta_y)**2, 1.0)

        # Zero polarization error.
        with self.assertRaises(ValueError):
            LGMode(0, 0, 1, 30., 2., 137., zeta_x=0.0, zeta_y=0.0)

        # Nonfinite polarization error.
        with self.assertRaises(ValueError):
            LGMode(0, 0, 1, 30., 2., 137., zeta_x=complex(np.nan, 0), zeta_y=1.0)

    def test_parameters_and_invalid_inputs(self):
        from main import initialize, INPUTS
        inputs = {k: v for k, v in INPUTS.items() if k != 'laser.w_0'}
        units, params, _ = initialize(inputs)
        mode = LGMode.from_parameters(params, units)
        self.assertAlmostEqual(mode.w_0 / mode.get_lambda(), 75)
        self.assertAlmostEqual(abs(mode.zeta_x)**2 + abs(mode.zeta_y)**2, 1.0)
        for change in ({'p': -1}, {'m': 0.5}, {'epsilon': 0}, {'w_0': 0}, {'zeta_x': 0, 'zeta_y': 0}):
            args = dict(p=0, m=0, epsilon=1, w_0=30, omega=2, c=137)
            args.update(change)
            with self.assertRaises(ValueError):
                LGMode(**args)
