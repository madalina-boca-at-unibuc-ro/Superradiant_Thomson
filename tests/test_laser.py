import unittest
from math import pi

from superradiant_thomson.laser import PulseTiming, register_temporal_scales, temporal_schema
from superradiant_thomson.parameters import ParameterResolver, default_registry


class PulseTests(unittest.TestCase):
    def test_parameter_resolution(self):
        registry = default_registry()
        register_temporal_scales(registry)
        parameters = ParameterResolver(registry, temporal_schema()).resolve({
            'laser.omega': 2 * pi,
            'laser.flat_top_periods': 4,
            'laser.sigma_l': {'value': 2, 'unit': 'T'},
            'laser.wing_factor': 5,
        })
        timing = PulseTiming.from_parameters(parameters)
        self.assertEqual(timing.duration, 24)
        self.assertEqual(timing.wing_duration, 10)
        self.assertEqual(timing.flat_top_duration, 4)

    def test_validation(self):
        for kwargs in ({'omega': 0}, {'sigma_l': -1}, {'wing_factor': float('nan')},
                       {'flat_top_periods': 1.5}, {'flat_top_periods': True}):
            inputs = dict(omega=1, sigma_l=1, wing_factor=5, flat_top_periods=2)
            inputs.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PulseTiming(**inputs)


class EnvelopeTests(unittest.TestCase):
    def test_shape_and_carrier(self):
        import numpy as np
        from superradiant_thomson.laser import TemporalFactor
        pulse = TemporalFactor(PulseTiming(2*pi, 4, 2, 5), c=137)
        np.testing.assert_allclose(pulse.envelope([-1, 0, 8, 10, 12, 14, 16, 24, 25]),
                                   [0, np.exp(-25), np.exp(-1), 1, 1, 1, np.exp(-1), np.exp(-25), 0])
        grid = np.linspace(0, 24, 101)
        np.testing.assert_allclose(pulse.envelope(grid), pulse.envelope(24-grid), atol=1e-14)
        np.testing.assert_allclose(pulse(10.25), -1j, atol=1e-14)
        np.testing.assert_allclose(pulse(grid + 2, 274), pulse(grid), atol=1e-13)
        self.assertEqual(pulse.support_at_z(274), (2, 26))

    def test_zero_plateau_and_configurable_cutoff(self):
        import numpy as np
        from superradiant_thomson.laser import TemporalFactor
        pulse = TemporalFactor(PulseTiming(1, 0, 2, 3), c=137)
        self.assertEqual(pulse.timing.duration, 12)
        self.assertEqual(pulse.envelope(6), 1)
        self.assertAlmostEqual(float(pulse.envelope(0)), np.exp(-9))
        with self.assertRaises(ValueError):
            pulse.envelope(float('nan'))


class AmplitudeTests(unittest.TestCase):
    def test_definition_and_scaling(self):
        from superradiant_thomson.laser import LaserAmplitude
        from superradiant_thomson.parameters import AtomicUnits
        c = AtomicUnits().c
        amplitude = LaserAmplitude(0.3, 0.057, c)
        self.assertAlmostEqual(amplitude.A_0 / c, 0.3)
        self.assertAlmostEqual(amplitude.E_0 / amplitude.omega, amplitude.A_0)
        self.assertAlmostEqual(LaserAmplitude(0.6, 0.057, c).E_0**2 / amplitude.E_0**2, 4)
        self.assertEqual(LaserAmplitude(0, 0.057, c).E_0, 0)

    def test_schema_and_validation(self):
        from main import initialize, INPUTS
        from superradiant_thomson.laser import LaserAmplitude
        units, parameters, _ = initialize(dict(INPUTS, **{'laser.a_0': 2}))
        self.assertAlmostEqual(LaserAmplitude.from_parameters(parameters, units).E_0,
                               2 * units.c * parameters['laser.omega'])
        for value in (-1, float('nan'), True):
            with self.assertRaises(ValueError):
                LaserAmplitude(value, 1, 137)
