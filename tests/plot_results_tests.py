"""Run with the plotting virtualenv: python tests/plot_results_tests.py."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plot_results', ROOT / 'scripts/plot_results.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class PlotTests(unittest.TestCase):
    def test_api_parser_accepts_nine_columns_and_stops_at_kernel_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.txt'
            path.write_text('CUDA API Statistics:\n90.0 1,000,000 4 250000 250000 1 500000 1 cudaMalloc\n'
                            'CUDA Kernel Statistics:\n10.0 1000 2 500 500 1 900 1 reduce_kernel\n')
            self.assertEqual(plot.nsight_api(path), [('cudaMalloc', 90., 1., 4)])

    def test_ncu_units_and_missing_values_are_not_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            path.write_text('warning\n"ID","Kernel Name","launch__registers_per_thread"\n'
                            '"","","register/thread"\n'
                            '"0","fusion::reduce_kernel(float*)","32"\n'
                            '"1","fusion::reduce_kernel(float*)","N/A"\n')
            self.assertEqual(plot.kernel_counters(path)['reduce_kernel']['launch__registers_per_thread'], [32.])

    def test_cpu_only_report_without_optional_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'timings.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['case', 'dataset', 'width', 'height', 'exposures', 'levels',
                                 'backend', 'metric', 'median_ms', 'min_ms', 'max_ms'])
                writer.writerow(['s0-test-n1-l1', '/scene', 7, 5, 1, 1, 'cpu', plot.CPU, 2, 1, 3])
            pages = plot.make_report(root, root / 'graphs')
            self.assertEqual(pages, 1)
            self.assertTrue((root / 'graphs/report.pdf').read_bytes().startswith(b'%PDF'))
            self.assertTrue((root / 'graphs/interpretation.md').exists())
            self.assertFalse(list((root / 'graphs').glob('*.png')))


if __name__ == '__main__':
    unittest.main()
