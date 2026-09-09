import json
import math
import unittest

from superradiant_thomson.parameters import (
    ANGULAR_FREQUENCY, LENGTH, VELOCITY, AtomicUnits, Dependency,
    Parameter, ParameterResolver, Quantity, ResolutionError, Scale,
    default_registry, register_laser_scales,
)


class ParameterTests(unittest.TestCase):
    def setUp(self):
        self.units = AtomicUnits()
        self.registry = default_registry(self.units)
        register_laser_scales(self.registry, self.units)
        self.schema = {
            'bunch.width': Parameter(LENGTH, validator=lambda x: x >= 0),
            'laser.omega': Parameter(ANGULAR_FREQUENCY, 0.057, lambda x: x > 0),
            'bunch.speed': Parameter(VELOCITY, Quantity(0.9, 'c')),
        }

    def test_dynamic_scale_defaults_and_provenance(self):
        result = ParameterResolver(self.registry, self.schema).resolve(
            {'bunch.width': {'value': 3, 'unit': 'lambda'}})
        self.assertAlmostEqual(result['bunch.width'], 3 * 2 * math.pi * self.units.c / 0.057)
        self.assertAlmostEqual(result['bunch.speed'], 0.9 * self.units.c)
        self.assertEqual(result.to_dict()['inputs']['bunch.width']['unit'], 'lambda')
        json.dumps(result.to_dict())
        with self.assertRaises(TypeError):
            result.values['bunch.width'] = 0

    def test_fixed_conversion(self):
        resolver = ParameterResolver(self.registry, {'width': Parameter(LENGTH)})
        a = resolver.resolve({'width': Quantity(800, 'nm')})
        b = resolver.resolve({'width': Quantity(0.8, 'um')})
        self.assertAlmostEqual(a['width'], b['width'])
        self.assertAlmostEqual(a['width'] * self.units.bohr_in_m, 800e-9)

    def test_errors(self):
        resolver = ParameterResolver(self.registry, self.schema)
        for inputs, message in [
            ({}, 'Missing parameter'),
            ({'extra': 1}, 'Unknown parameters'),
            ({'bunch.width': Quantity(1, 'c')}, 'incompatible dimension'),
            ({'bunch.width': Quantity(1, 'unknown')}, 'Unknown unit'),
            ({'bunch.width': float('nan')}, 'finite real'),
            ({'bunch.width': True}, 'finite real'),
            ({'bunch.width': -1}, 'failed validation'),
            ({'bunch.width': Quantity(1, 'lambda'), 'laser.omega': 0}, 'failed validation'),
        ]:
            with self.subTest(inputs=inputs), self.assertRaisesRegex(ResolutionError, message):
                resolver.resolve(inputs)

    def test_extension_and_cycle(self):
        self.registry.register('width', Scale(LENGTH, lambda p: p['bunch.width'],
                                            (Dependency('bunch.width', LENGTH),)))
        resolver = ParameterResolver(self.registry, {'bunch.width': Parameter(LENGTH),
                                                     'screen.width': Parameter(LENGTH)})
        result = resolver.resolve({'bunch.width': 10, 'screen.width': Quantity(2, 'width')})
        self.assertEqual(result['screen.width'], 20)
        with self.assertRaisesRegex(ResolutionError, 'Circular dependency'):
            resolver.resolve({'bunch.width': Quantity(1, 'width'), 'screen.width': 1})

    def test_missing_and_wrong_dependency(self):
        for spec, message in [(None, 'missing dependency'), (Parameter(LENGTH, 1), 'dependency dimension')]:
            schema = {'bunch.width': Parameter(LENGTH)}
            if spec:
                schema['laser.omega'] = spec
            with self.assertRaisesRegex(ResolutionError, message):
                ParameterResolver(self.registry, schema).resolve({'bunch.width': Quantity(1, 'lambda')})

    def test_complex_parameter(self):
        from superradiant_thomson.parameters import DIMENSIONLESS
        schema = {
            'zeta_x': Parameter(DIMENSIONLESS, 1.0, allow_complex=True),
            'zeta_y': Parameter(DIMENSIONLESS, 0.0, allow_complex=True),
        }
        resolver = ParameterResolver(self.registry, schema)
        res = resolver.resolve({'zeta_x': 1.0 + 2.0j, 'zeta_y': {'value': 3.0j, 'unit': 'au'}})
        self.assertEqual(res['zeta_x'], 1.0 + 2.0j)
        self.assertEqual(res['zeta_y'], 3.0j)
        prov = res.to_dict()
        self.assertEqual(prov['resolved']['zeta_x'], {'real': 1.0, 'imag': 2.0})
        self.assertEqual(prov['resolved']['zeta_y'], {'real': 0.0, 'imag': 3.0})
        json.dumps(prov)

        # Non-complex parameter rejecting complex input.
        real_resolver = ParameterResolver(self.registry, {'width': Parameter(LENGTH)})
        with self.assertRaisesRegex(ResolutionError, 'finite real number'):
            real_resolver.resolve({'width': 1.0 + 2.0j})

    def test_repeated_resolution_is_fresh(self):
        resolver = ParameterResolver(self.registry, self.schema)
        a = resolver.resolve({'bunch.width': Quantity(1, 'lambda')})
        b = resolver.resolve({'bunch.width': Quantity(1, 'lambda'), 'laser.omega': 0.114})
        self.assertAlmostEqual(a['bunch.width'], 2 * b['bunch.width'])
        with self.assertRaises(ValueError):
            self.registry.register('nm', Scale(LENGTH, 1))


if __name__ == '__main__':
    unittest.main()
