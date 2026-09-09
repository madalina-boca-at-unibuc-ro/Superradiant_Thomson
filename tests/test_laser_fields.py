import unittest
import numpy as np
import matplotlib.pyplot as plt

from superradiant_thomson.lg_mode import (
    LGMode, lg_schema, laser_plot_schema, evaluate_laser_fields,
    faraday_tensor_from_fields, evaluate_laser_faraday_tensor, contract_faraday,
)
from superradiant_thomson.laser import (
    PulseTiming, TemporalFactor, LaserAmplitude, amplitude_schema, temporal_schema,
    register_temporal_scales,
)
from superradiant_thomson.parameters import (
    AtomicUnits, ParameterResolver, default_registry, register_laser_scales,
)
from superradiant_thomson.plotting import plot_laser_fields


class LaserFieldsTests(unittest.TestCase):
    def setUp(self):
        self.units = AtomicUnits()
        self.registry = default_registry(self.units)
        register_laser_scales(self.registry, self.units)
        register_temporal_scales(self.registry)
        self.schema = temporal_schema() | lg_schema() | laser_plot_schema() | amplitude_schema()
        self.inputs = {
            'laser.omega': 0.057,
            'laser.a_0': 1.0,
            'laser.flat_top_periods': 5,
            'laser.sigma_l': {'value': 2, 'unit': 'T'},
            'laser.wing_factor': 4,
            'laser.p': 0,
            'laser.m': 0,
            'laser.epsilon': 1,
            'laser.w_0': {'value': 50, 'unit': 'lambda'},
            'laser.zeta_x': 1.0,
            'laser.zeta_y': 0.0,
            'x_plot_laser': {'value': 0.5, 'unit': 'w_0'},
            'y_plot_laser': {'value': 0.0, 'unit': 'w_0'},
            'z_plot_laser': {'value': 0.0, 'unit': 'w_0'},
        }

    def test_laser_plot_schema_and_w0_scale(self):
        # Default (0,0,0) laser plot point
        default_inputs = dict(self.inputs)
        default_inputs.pop('x_plot_laser')
        default_inputs.pop('y_plot_laser')
        default_inputs.pop('z_plot_laser')
        params_def = ParameterResolver(self.registry, self.schema).resolve(default_inputs)
        self.assertEqual(params_def['x_plot_laser'], 0.0)
        self.assertEqual(params_def['y_plot_laser'], 0.0)
        self.assertEqual(params_def['z_plot_laser'], 0.0)

        # Explicit (0.5, 0.0, 0.0) in w_0 unit
        params = ParameterResolver(self.registry, self.schema).resolve(self.inputs)
        w_0 = params['laser.w_0']
        self.assertAlmostEqual(params['x_plot_laser'], 0.5 * w_0)
        self.assertEqual(params['y_plot_laser'], 0.0)
        self.assertEqual(params['z_plot_laser'], 0.0)

    def test_evaluate_laser_fields_physics(self):
        params = ParameterResolver(self.registry, self.schema).resolve(self.inputs)
        pulse = TemporalFactor(PulseTiming.from_parameters(params), self.units.c)
        mode = LGMode.from_parameters(params, self.units)
        amplitude = LaserAmplitude.from_parameters(params, self.units)

        time = np.linspace(0.0, pulse.timing.duration, 201)
        r = (params['x_plot_laser'], params['y_plot_laser'], params['z_plot_laser'])
        fields = evaluate_laser_fields(mode, amplitude, pulse, r, time)

        # Check returned dict keys
        for key in ('E_x', 'E_y', 'E_z', 'B_x', 'B_y', 'B_z'):
            self.assertIn(key, fields)
            self.assertEqual(fields[key].shape, time.shape)

        # For linear polarization along x (zeta_x=1, zeta_y=0) at y=0, z=0:
        # B_x should be 0, B_y = E_x / c
        np.testing.assert_allclose(fields['B_x'], 0.0, atol=1e-15)
        np.testing.assert_allclose(fields['B_y'], fields['E_x'] / mode.c, rtol=1e-12)
        # E_y should be 0 since zeta_y = 0
        np.testing.assert_allclose(fields['E_y'], 0.0, atol=1e-15)

    def test_faraday_tensor_construction_and_properties(self):
        params = ParameterResolver(self.registry, self.schema).resolve(self.inputs)
        pulse = TemporalFactor(PulseTiming.from_parameters(params), self.units.c)
        mode = LGMode.from_parameters(params, self.units)
        amplitude = LaserAmplitude.from_parameters(params, self.units)

        time = np.linspace(0.0, pulse.timing.duration, 50)
        r = (params['x_plot_laser'], params['y_plot_laser'], params['z_plot_laser'])
        fields = evaluate_laser_fields(mode, amplitude, pulse, r, time)
        c = mode.c

        F = evaluate_laser_faraday_tensor(mode, amplitude, pulse, r, time)
        self.assertEqual(F.shape, (50, 4, 4))

        # Check anti-symmetry F^{\mu\nu} = -F^{\nu\mu}
        for t_idx in range(50):
            np.testing.assert_allclose(F[t_idx], -F[t_idx].T, atol=1e-15)
            np.testing.assert_allclose(np.diag(F[t_idx]), 0.0, atol=1e-15)

        # Check specific component mappings
        np.testing.assert_allclose(F[:, 0, 1], -fields['E_x'] / c)
        np.testing.assert_allclose(F[:, 0, 2], -fields['E_y'] / c)
        np.testing.assert_allclose(F[:, 0, 3], -fields['E_z'] / c)
        np.testing.assert_allclose(F[:, 1, 0], fields['E_x'] / c)
        np.testing.assert_allclose(F[:, 1, 2], -fields['B_z'])
        np.testing.assert_allclose(F[:, 1, 3], fields['B_y'])
        np.testing.assert_allclose(F[:, 2, 3], -fields['B_x'])

    def test_contract_faraday_lorentz_force_identity(self):
        # Test F^{\mu\nu} p_\nu = ( E \cdot p / c, \gamma m (E + v \times B) )
        params = ParameterResolver(self.registry, self.schema).resolve(self.inputs)
        pulse = TemporalFactor(PulseTiming.from_parameters(params), self.units.c)
        mode = LGMode.from_parameters(params, self.units)
        amplitude = LaserAmplitude.from_parameters(params, self.units)

        time = np.array([pulse.timing.duration / 2])
        r = (0.1 * mode.w_0, 0.2 * mode.w_0, 0.0)
        fields = evaluate_laser_fields(mode, amplitude, pulse, r, time)
        c = mode.c
        F = faraday_tensor_from_fields(fields, c=c)[0]  # shape (4,4)

        # Particle test state: gamma=2, v=(0.5*c, 0, 0)
        m = 1.0
        v = np.array([0.5 * c, 0.2 * c, -0.1 * c])
        gamma = 1.0 / np.sqrt(1 - np.sum(v**2) / c**2)
        p_spatial = gamma * m * v
        p0 = gamma * m * c
        p_contravariant = np.array([p0, p_spatial[0], p_spatial[1], p_spatial[2]])

        # Contract F^{\mu\nu} p_\nu
        force_4d = contract_faraday(F, p_contravariant)  # shape (4,)

        # Expected 3-vector component: gamma * m * (E + v x B)
        E = np.array([fields['E_x'][0], fields['E_y'][0], fields['E_z'][0]])
        B = np.array([fields['B_x'][0], fields['B_y'][0], fields['B_z'][0]])
        v_cross_B = np.cross(v, B)
        expected_spatial_force = gamma * m * (E + v_cross_B)
        expected_energy_rate = np.dot(E, p_spatial) / c

        np.testing.assert_allclose(force_4d[0], expected_energy_rate, rtol=1e-12)
        np.testing.assert_allclose(force_4d[1:], expected_spatial_force, rtol=1e-12)

    def test_plot_laser_fields(self):
        params = ParameterResolver(self.registry, self.schema).resolve(self.inputs)
        pulse = TemporalFactor(PulseTiming.from_parameters(params), self.units.c)
        mode = LGMode.from_parameters(params, self.units)
        amplitude = LaserAmplitude.from_parameters(params, self.units)

        time = np.linspace(0.0, pulse.timing.duration, 101)
        r_w0 = (0.5, 0.0, 0.0)
        r_au = (r_w0[0] * mode.w_0, r_w0[1] * mode.w_0, r_w0[2] * mode.w_0)
        fields = evaluate_laser_fields(mode, amplitude, pulse, r_au, time)

        fig, axs = plot_laser_fields(fields, time, r_w0=r_w0, pulse=pulse, time_unit='T')
        self.assertEqual(axs.shape, (2, 3))
        plt.close(fig)
