import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from superradiant_thomson.output import create_run_directory
from superradiant_thomson.parameters import AtomicUnits


class OutputTests(unittest.TestCase):
    def test_unique_directories_and_preservation(self):
        with TemporaryDirectory() as root:
            first = create_run_directory(root)
            (first / 'keep').write_text('original')
            second = create_run_directory(root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'keep').read_text(), 'original')
            self.assertTrue(second.name.startswith('SRT_'))

    def test_run_and_failure_metadata(self):
        import main
        fast_inputs = dict(main.INPUTS, **{'electron.N': 2, 'screen.Nx': 4, 'screen.Ny': 4})
        with TemporaryDirectory() as root:
            with patch.object(main, 'INPUTS', fast_inputs):
                run = main.main(output_root=root)
                self.assertEqual(json.loads((run / 'run.json').read_text())['status'], 'complete')
                with np.load(run / 'temporal_factor.npz') as data:
                    self.assertEqual(data['time'].shape, (1001,))
                    np.testing.assert_allclose(np.abs(data['temporal_factor']), data['envelope'])
                self.assertTrue((run / 'temporal_factor.png').stat().st_size > 0)
                with patch.object(main, 'sample_pulse', side_effect=ValueError('test failure')):
                    with self.assertRaises(ValueError):
                        main.main(output_root=root)
                statuses = [json.loads(p.read_text())['status'] for p in Path(root).glob('*/run.json')]
                self.assertCountEqual(statuses, ['complete', 'failed'])

    def test_screen_observable_flux_density_ratio_equals_c(self):
        """Physical identity in run.json: integrated flux / integrated density along Oz is +-c.

        For radiation far from the source, flux = c * n_z * density for any locally-conserved
        quantity it carries (all field components share the same phase velocity c in vacuum), so
        the ratio should be +c if the screen sits on the +Oz side of the source and -c if it sits
        on the -Oz side. Checked for all three plotted density/flux pairs: energy, total
        (spin+orbital) angular momentum, and spin angular momentum alone. The ratios themselves are
        precomputed and stored in run.json (no separate run_log.txt is written anymore).
        """
        import main
        c = AtomicUnits().c
        fast_inputs = dict(main.INPUTS, **{'electron.N': 2, 'screen.Nx': 4, 'screen.Ny': 4})
        z_screen_value = fast_inputs['screen.z_screen']
        z_screen_value = z_screen_value['value'] if isinstance(z_screen_value, dict) else z_screen_value
        expected_sign = 1.0 if z_screen_value >= 0 else -1.0

        with TemporaryDirectory() as root:
            with patch.object(main, 'INPUTS', fast_inputs):
                run = main.main(output_root=root)
            self.assertFalse((run / 'run_log.txt').exists())
            run_data = json.loads((run / 'run.json').read_text())
            rows = run_data['screen_observables']['rows']
            laser_rows = run_data['incident_laser_observables']['rows']

        self.assertGreater(len(rows), 0)
        for row in rows:
            for name in ('energy_flux_density_ratio', 'spin_flux_density_ratio',
                        'angular_momentum_flux_density_ratio'):
                self.assertAlmostEqual(row[name], expected_sign * c, delta=0.01 * c,
                                       msg=f'{name} for harmonic N={row["N"]}')
            self.assertIn('spin_flux_energy_flux_ratio', row)
            self.assertIn('theoretical_spin_flux_energy_flux_ratio', row)
            self.assertIn('orbital_flux_energy_flux_ratio', row)
            self.assertIn('theoretical_orbital_flux_energy_flux_ratio', row)

        self.assertEqual(len(laser_rows), 1)
        for name in ('energy_flux_density_ratio', 'spin_flux_density_ratio',
                    'angular_momentum_flux_density_ratio'):
            self.assertAlmostEqual(laser_rows[0][name], c, delta=0.01 * c,
                                   msg=f'incident laser {name}')
        # Incident laser ratio tests vs theoretical values
        self.assertAlmostEqual(
            laser_rows[0]['spin_flux_energy_flux_ratio'],
            laser_rows[0]['theoretical_spin_flux_energy_flux_ratio'],
            delta=0.01 * laser_rows[0]['theoretical_spin_flux_energy_flux_ratio']
        )
        self.assertAlmostEqual(
            laser_rows[0]['orbital_flux_energy_flux_ratio'],
            laser_rows[0]['theoretical_orbital_flux_energy_flux_ratio'],
            delta=0.01 * laser_rows[0]['theoretical_orbital_flux_energy_flux_ratio']
        )

