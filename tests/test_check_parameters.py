"""Unit tests for simulation input parameter validation."""
import unittest

from main import INPUTS
from superradiant_thomson.check_parameters import check_parameters, ParameterCheckError


class TestCheckParameters(unittest.TestCase):
    def test_valid_default_inputs_pass(self):
        errors = check_parameters(INPUTS, raise_on_error=False)
        self.assertEqual(errors, [])
        # Should also not raise when raise_on_error=True
        check_parameters(INPUTS, raise_on_error=True)

    def test_laser_m_negative_fails(self):
        bad_inputs = dict(INPUTS, **{'laser.m': -1})
        errors = check_parameters(bad_inputs, raise_on_error=False)
        self.assertTrue(any('laser.m' in err for err in errors))
        with self.assertRaises(ParameterCheckError) as ctx:
            check_parameters(bad_inputs, raise_on_error=True)
        self.assertIn('laser.m: must be a non-negative integer', str(ctx.exception))

    def test_laser_p_negative_fails(self):
        bad_inputs = dict(INPUTS, **{'laser.p': -0.5})
        errors = check_parameters(bad_inputs, raise_on_error=False)
        self.assertTrue(any('laser.p' in err for err in errors))
        with self.assertRaises(ParameterCheckError):
            check_parameters(bad_inputs, raise_on_error=True)

    def test_laser_omega_nonpositive_fails(self):
        bad_inputs = dict(INPUTS, **{'laser.omega': -0.057})
        errors = check_parameters(bad_inputs, raise_on_error=False)
        self.assertTrue(any('laser.omega' in err for err in errors))
        with self.assertRaises(ParameterCheckError):
            check_parameters(bad_inputs, raise_on_error=True)

    def test_electron_N_invalid_fails(self):
        bad_inputs = dict(INPUTS, **{'electron.N': 0})
        errors = check_parameters(bad_inputs, raise_on_error=False)
        self.assertTrue(any('electron.N' in err for err in errors))
        with self.assertRaises(ParameterCheckError):
            check_parameters(bad_inputs, raise_on_error=True)

    def test_screen_shape_invalid_fails(self):
        bad_inputs = dict(INPUTS, **{'screen.shape': 'triangular'})
        errors = check_parameters(bad_inputs, raise_on_error=False)
        self.assertTrue(any('screen.shape' in err for err in errors))
        with self.assertRaises(ParameterCheckError):
            check_parameters(bad_inputs, raise_on_error=True)

    def test_screen_harmonic_range_invalid_fails(self):
        bad_inputs = dict(INPUTS, **{'screen.N_min': 5, 'screen.N_max': 2})
        errors = check_parameters(bad_inputs, raise_on_error=False)
        self.assertTrue(any('screen.N_min' in err for err in errors))
        with self.assertRaises(ParameterCheckError):
            check_parameters(bad_inputs, raise_on_error=True)


if __name__ == '__main__':
    unittest.main()
