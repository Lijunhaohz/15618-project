"""Exercise profiler orchestration without requiring GPU/perf permissions."""
import argparse
import os
import subprocess
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('profiling', ROOT / 'scripts/profiling.py')
profiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiling)
cuda_spec = importlib.util.spec_from_file_location('cuda_environment', ROOT / 'scripts/cuda_environment.py')
cuda_environment = importlib.util.module_from_spec(cuda_spec)
cuda_spec.loader.exec_module(cuda_environment)


class ProfilingTests(unittest.TestCase):
    def options(self, **changes):
        values = dict(profile='all', perf_trials=2, perf_repeats=1,
                      perf_record=True, ncu_launch_count=0, ncu_set='full')
        values.update(changes)
        return argparse.Namespace(**values)

    def test_full_gpu_coverage_and_counter_exports(self):
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)

            def run(command, log, required):
                commands.append(list(map(str, command)))
                log.write_text('ok')
                return 0

            with patch.object(profiling.shutil, 'which', return_value='/tool'), \
                 patch.object(profiling.platform, 'system', return_value='Linux'):
                statuses = profiling.collect(self.options(), ROOT / 'build', case,
                                              [ROOT / 'image.png'], 5, ['cpu', 'cuda'], run)
            ncu = next(cmd for cmd in commands if cmd[:2] == ['ncu', '--set'])
            self.assertEqual(ncu[2], 'full')
            self.assertNotIn('--launch-count', ncu)
            self.assertTrue(any('--cuda-memory-usage=true' in cmd for cmd in commands))
            self.assertTrue(any(cmd[:2] == ['perf', 'record'] for cmd in commands))
            self.assertTrue(any('--csv' in cmd for cmd in commands))
            self.assertTrue(all(s['status'] == 'ok' for s in statuses))
            self.assertTrue((case / 'profile-status.json').exists())

    def test_tool_error_with_zero_exit_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(command, log, required):
                log.write_text("ERROR: profiler could not collect data\n")
                return 0

            with patch.object(profiling.shutil, 'which', return_value='/tool'):
                statuses = profiling.collect(self.options(profile='nsys'), ROOT / 'build',
                                              Path(directory), [], 3, ['cuda'], run)
            nsys = next(s for s in statuses if s['name'] == 'nsys')
            self.assertEqual(nsys['status'], 'failed')

    def test_missing_tools_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(profiling.shutil, 'which', return_value=None):
            statuses = profiling.collect(self.options(), ROOT / 'build', Path(directory),
                                          [], 3, ['cuda'], lambda *a, **k: self.fail('must not run'))
            self.assertTrue(statuses)
            self.assertTrue(all(s['status'] == 'unavailable' for s in statuses))

    def test_counter_permission_failure_does_not_prevent_gpu_profile(self):
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            def run(command, log, required):
                commands.append(list(map(str, command)))
                log.write_text('Permission denied' if command[0] == 'perf' else 'ok')
                return 1 if command[0] == 'perf' else 0

            with patch.object(profiling.shutil, 'which', return_value='/tool'):
                statuses = profiling.collect(self.options(), ROOT / 'build', Path(directory),
                                              [], 3, ['cpu', 'cuda'], run)
            self.assertTrue(any(s['status'] == 'failed' for s in statuses))
            self.assertTrue(any(cmd[0] == 'ncu' for cmd in commands))
            stat_calls = [cmd for cmd in commands if cmd[:2] == ['perf', 'stat']]
            self.assertEqual(len(stat_calls), len(profiling.PERF_GROUPS))


class CudaEnvironmentTests(unittest.TestCase):
    def test_standard_install_and_compatible_host(self):
        def which(name):
            return {'/usr/local/cuda/bin/nvcc': '/usr/local/cuda/bin/nvcc',
                    'g++-11': '/usr/bin/g++-11'}.get(name)
        with patch.dict(os.environ, {'PATH': '/usr/bin'}, clear=True), \
             patch.object(cuda_environment.shutil, 'which', side_effect=which), \
             patch.object(cuda_environment.os, 'access', return_value=True), \
             patch.object(cuda_environment.subprocess, 'run', return_value=
                          subprocess.CompletedProcess([], 0, 'release 11.8, V11.8')):
            self.assertEqual(cuda_environment.configure_cuda_environment(), '/usr/local/cuda/bin/nvcc')
            self.assertEqual(os.environ['CUDAHOSTCXX'], '/usr/bin/g++-11')
            self.assertTrue(os.environ['PATH'].startswith('/usr/local/cuda/bin:'))

    def test_missing_compiler_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(cuda_environment.shutil, 'which', return_value=None), \
             patch.object(cuda_environment.os, 'access', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Set CUDACXX='):
                cuda_environment.configure_cuda_environment()

    def test_explicit_compilers_are_preserved(self):
        with patch.dict(os.environ, {'CUDACXX': '/custom/nvcc', 'CUDAHOSTCXX': '/custom/g++'}, clear=True), \
             patch.object(cuda_environment.shutil, 'which', return_value='/custom/nvcc'), \
             patch.object(cuda_environment.subprocess, 'run', return_value=
                          subprocess.CompletedProcess([], 0, 'release 11.8,')):
            cuda_environment.configure_cuda_environment()
            self.assertEqual(os.environ['CUDAHOSTCXX'], '/custom/g++')


class CompactResultsTests(unittest.TestCase):
    def test_retains_evidence_and_consolidates_intermediates(self):
        import json
        compact_spec = importlib.util.spec_from_file_location('compact_results', ROOT / 'scripts/compact_results.py')
        compact = importlib.util.module_from_spec(compact_spec)
        compact_spec.loader.exec_module(compact)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{}')
            (root / 'timings.csv').write_text('original timings')
            case = root / 's0-scene-n3-l5'
            case.mkdir()
            (case / 'inputs.json').write_text('["a.png"]')
            (case / 'cpu.log').write_text('$ benchmark a.png\nverbose')
            (case / 'cpu-perf-instructions.csv').write_text('100;;cycles;0%;123;100.00;;\n')
            (case / 'cpu-perf-stalls.csv').write_text('<not supported>;;stalled-cycles-frontend;;;;;\n')
            (case / 'cpu-perf.data').write_bytes(b'large raw recording')
            (case / 'cuda.png').write_bytes(b'image')
            (case / 'cpu.png').write_bytes(b'duplicate image')
            (case / 'kernels.ncu-rep').write_bytes(b'kernel evidence')
            (case / 'timeline.nsys-rep').write_bytes(b'timeline evidence')
            (case / 'timeline.sqlite').write_bytes(b'intermediate')
            (case / 'my-notes.txt').write_text('keep user notes')
            compact.compact_results(root)
            self.assertEqual((root / 'timings.csv').read_text(), 'original timings')
            self.assertTrue((case / 'kernels.ncu-rep').exists())
            self.assertTrue((case / 'timeline.nsys-rep').exists())
            self.assertTrue((case / 'cpu-counters.csv').exists())
            self.assertTrue((case / 'my-notes.txt').exists())
            self.assertFalse((case / 'cpu-perf.data').exists())
            self.assertFalse((case / 'cpu.png').exists())
            self.assertFalse((case / 'timeline.sqlite').exists())
            self.assertIn('not supported', (root / 'diagnostics.txt').read_text())
            saved = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(saved['cases'][case.name]['inputs'], ['a.png'])
            # Re-running cleanup must preserve the already-consolidated counters.
            compact.compact_results(root)
            self.assertTrue((case / 'cpu-counters.csv').exists())


if __name__ == '__main__':
    unittest.main()
