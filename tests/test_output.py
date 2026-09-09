import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from superradiant_thomson.output import create_run_directory


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
        with TemporaryDirectory() as root:
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
