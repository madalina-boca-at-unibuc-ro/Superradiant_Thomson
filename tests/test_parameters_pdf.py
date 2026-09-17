"""Unit tests for simulation input parameters PDF generation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from superradiant_thomson.parameters import (
    AtomicUnits, ParameterResolver, default_registry, register_laser_scales,
)
from superradiant_thomson.laser import (
    PulseTiming, register_temporal_scales, temporal_schema, amplitude_schema,
)
from superradiant_thomson.lg_mode import lg_schema, laser_plot_schema
from superradiant_thomson.electron import electron_schema
from superradiant_thomson.screen import screen_schema
from superradiant_thomson.plotting import generate_parameters_pdf
from superradiant_thomson.output import write_parameters_pdf


class TestParametersPdf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from main import INPUTS
        cls.units = AtomicUnits()
        registry = default_registry(cls.units)
        register_laser_scales(registry, cls.units)
        register_temporal_scales(registry)
        schema = (
            temporal_schema() | lg_schema() | laser_plot_schema() |
            amplitude_schema() | electron_schema() | screen_schema()
        )
        test_inputs = dict(INPUTS, **{'electron.N': 4, 'screen.Nx': 8, 'screen.Ny': 8})
        cls.parameters = ParameterResolver(registry, schema).resolve(test_inputs)

    def test_generate_parameters_pdf_creates_valid_file(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / 'test_params.pdf'
            out = generate_parameters_pdf(pdf_path, self.parameters, units=self.units)
            self.assertEqual(out, pdf_path)
            self.assertTrue(pdf_path.is_file())
            # Ensure it is a valid non-empty PDF
            size = pdf_path.stat().st_size
            self.assertGreater(size, 5000)
            content = pdf_path.read_bytes()
            self.assertTrue(content.startswith(b'%PDF-'))
            self.assertIn(b'%%EOF', content)

    def test_write_parameters_pdf_from_output(self):
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / 'test_output_params.pdf'
            out = write_parameters_pdf(pdf_path, self.parameters, units=self.units, run_dir=tmpdir)
            self.assertEqual(out, pdf_path)
            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 5000)


if __name__ == '__main__':
    unittest.main()
