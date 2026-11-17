#!/usr/bin/env python3
"""Reproducible CPU/CUDA sweeps and optional Nsight collection (standard library only)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import statistics
from profiling import collect
from cuda_environment import configure_cuda_environment
from compact_results import compact_results

ROOT = Path(__file__).resolve().parents[1]
TIMING = re.compile(r'^(.*?)\s+median=\s*([\d.eE+-]+) ms\s+min=\s*([\d.eE+-]+) ms\s+max=\s*([\d.eE+-]+) ms$')
ERROR = re.compile(r'^(.*?): max_abs=([\d.eE+-]+) rmse=([\d.eE+-]+)')


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def run(command, log, required=True):
    print(shlex.join(map(str, command)), flush=True)
    with log.open('w') as stream:
        stream.write('$ ' + shlex.join(map(str, command)) + '\n')
        stream.flush()
        result = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    if required and result.returncode:
        print(log.read_text()[-12000:], file=sys.stderr)
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', type=Path,
                        help='repeat for each scene directory; default: data/memorial')
    parser.add_argument('--counts', nargs='+', type=positive, default=[1, 3, 5, 9, 15])
    parser.add_argument('--levels', nargs='+', type=positive, default=[1, 2, 3, 4, 5])
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'both'], default='both')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/experiments')
    parser.add_argument('--output', type=Path, help='new results directory (must not exist)')
    parser.add_argument('--trials', type=positive, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--skip-build', action='store_true', help='use an existing compatible Release build')
    parser.add_argument('--profile', choices=['none', 'nsys', 'ncu', 'both', 'perf', 'all'], default='all')
    parser.add_argument('--profile-count', nargs='+', type=positive, default=[3, 15])
    parser.add_argument('--profile-levels', nargs='+', type=positive, default=[1, 5])
    parser.add_argument('--ncu-launch-count', type=int, default=0, help='0 profiles every kernel launch')
    parser.add_argument('--ncu-set', choices=['basic', 'full'], default='full')
    parser.add_argument('--profile-scope', choices=['representative', 'all'], default='representative',
                        help='deep profiling for selected counts/depths per scene, or every case')
    parser.add_argument('--perf-repeats', type=positive, default=3)
    parser.add_argument('--perf-trials', type=positive, default=10)
    parser.add_argument('--perf-record', action='store_true', help='optional CPU call-stack sampling; disabled by default')
    parser.add_argument('--keep-raw', action='store_true', help='retain intermediate logs/debug artifacts')
    parser.add_argument('--save-images', action='store_true', help='save a fused PNG per case/backend')
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error('--warmups must be nonnegative')
    if any(level > 5 for level in args.levels + args.profile_levels):
        parser.error('this project supports at most five pyramid levels')
    if args.ncu_launch_count < 0:
        parser.error('--ncu-launch-count must be nonnegative')
    if args.backend == 'cpu' and args.profile in ('nsys', 'ncu', 'both'):
        parser.error('GPU profiling requires --backend cuda or both')
    datasets = args.dataset or [ROOT / 'data/memorial']
    scenes = []
    for directory in datasets:
        directory = directory.resolve()
        if not directory.is_dir():
            parser.error(f'not a directory: {directory}')
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.png', '.ppm'))
        if not files:
            parser.error(f'no PNG/PPM images: {directory}')
        scenes.append((directory, files))
    if args.backend != 'cpu' and not args.skip_build:
        configure_cuda_environment()
    profiling = args.profile != 'none'
    out = (args.output or ROOT / 'output' / ('experiments-' + datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    backends = ['cpu', 'cuda'] if args.backend == 'both' else [args.backend]
    metadata = {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                'arguments': vars(args).copy(),
                'compiler_environment': {key: os.environ.get(key) for key in
                                         ('CUDACXX', 'CUDAHOSTCXX', 'CC', 'CXX')}, 'datasets': [], 'profile_status': [], 'cases': {}}
    for directory, files in scenes:
        metadata['datasets'].append({'directory': str(directory), 'files': [
            {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]})
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    for name, command in [('git-revision', ['git', 'rev-parse', 'HEAD']),
                          ('git-status', ['git', 'status', '--short']),
                          ('git-diff', ['git', 'diff']),
                          ('cpu-info', ['lscpu']), ('gpu-info', ['nvidia-smi', '-q']),
                          ('cuda-version', ['nvcc', '--version']),
                          ('perf-version', ['perf', '--version']),
                          ('nsys-version', ['nsys', '--version']), ('ncu-version', ['ncu', '--version'])]:
        if shutil.which(command[0]):
            run(command, out / (name + '.txt'), required=False)
    if not args.skip_build:
        run(['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
             '-DFUSION_ENABLE_CUDA=' + ('ON' if 'cuda' in backends else 'OFF')], out / 'configure.log')
        run(['cmake', '--build', build, '--parallel'], out / 'build.log')
        run(['ctest', '--test-dir', build, '--output-on-failure'], out / 'tests.log')
    if (build / 'CMakeCache.txt').exists():
        shutil.copy2(build / 'CMakeCache.txt', out / 'CMakeCache.txt')
    samples_file = (out / 'samples.csv').open('w', newline='')
    memory_file = (out / 'device-memory.csv').open('w', newline='')
    samples = csv.writer(samples_file)
    memory = csv.writer(memory_file)
    samples.writerow(['case', 'backend', 'metric', 'trial', 'ms'])
    memory.writerow(['case', 'trial', 'peak_bytes', 'allocated_bytes', 'allocations', 'live_bytes'])
    with (out / 'timings.csv').open('w', newline='') as timing_file, (out / 'errors.csv').open('w', newline='') as error_file:
        timings = csv.writer(timing_file)
        errors = csv.writer(error_file)
        timings.writerow(['case', 'dataset', 'width', 'height', 'exposures', 'levels', 'backend', 'metric', 'median_ms', 'min_ms', 'max_ms'])
        errors.writerow(['case', 'comparison', 'max_abs', 'rmse'])
        for scene_index, (directory, files) in enumerate(scenes):
            for count in sorted(set(args.counts + (args.profile_count if profiling else []))):
                if count > len(files):
                    print(f'Skip {directory.name}, {count} exposures: only {len(files)} available')
                    continue
                # Spread selections over the full bracket rather than taking only dark exposures.
                selected = [files[len(files) // 2]] if count == 1 else [
                    files[round(i * (len(files) - 1) / (count - 1))] for i in range(count)]
                for levels in sorted(set(args.levels + (args.profile_levels if profiling else []))):
                    case = f's{scene_index}-{directory.name}-n{count}-l{levels}'
                    case_dir = out / case
                    case_dir.mkdir()
                    metadata['cases'][case] = {'inputs': list(map(str, selected)), 'commands': []}
                    for backend in backends:
                        command = [build / (backend + '_benchmark'), '--levels', levels,
                                   '--warmups', args.warmups, '--trials', args.trials, '--', *selected]
                        log = case_dir / (backend + '.log')
                        metadata['cases'][case]['commands'].append(list(map(str, command)))
                        run(command, log)
                        text = log.read_text()
                        shape = re.search(r'(\d+)x(\d+), (\d+) exposures, (\d+) levels', text)
                        if not shape:
                            raise RuntimeError(f'Missing benchmark metadata: {log}')
                        for line in text.splitlines():
                            sample = re.match(r'^Sample: (.*?) trial=(\d+) ms=([\d.eE+-]+)$', line)
                            if sample:
                                samples.writerow([case, backend, *sample.groups()])
                            allocation = re.match(r'^Device memory: trial=(\d+) peak_bytes=(\d+) allocated_bytes=(\d+) allocations=(\d+) live_bytes=(\d+)$', line)
                            if allocation:
                                memory.writerow([case, *allocation.groups()])
                            match = TIMING.match(line)
                            if match:
                                timings.writerow([case, directory, shape[1], shape[2], count, levels,
                                                  backend, *match.groups()])
                            match = ERROR.match(line)
                            if match:
                                errors.writerow([case, *match.groups()])
                        samples_file.flush()
                        memory_file.flush()
                        timing_file.flush()
                        error_file.flush()
                        if args.save_images and backend == backends[-1] and (count == 3 or
                                (count == max((c for c in args.counts if c <= len(files)), default=count) and levels == max(args.levels))):
                            run([build / 'exposure_fusion', '--backend', backend, '--levels', levels,
                                 '--output', case_dir / (backend + '.png'), '--', *selected],
                                case_dir / (backend + '-image.log'))
                    if profiling and (args.profile_scope == 'all' or
                                      (count in args.profile_count and levels in args.profile_levels)):
                        metadata['profile_status'].extend(collect(args, build, case_dir, selected,
                                                                   levels, backends, run))
                        (out / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    metadata['cases'] = {}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    samples_file.close()
    memory_file.close()
    # Preserve distributions and derived throughput rather than only rounded medians.
    groups = {}
    with (out / 'samples.csv').open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault((row['case'], row['backend'], row['metric']), []).append(float(row['ms']))
    with (out / 'distributions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'backend', 'metric', 'trials', 'mean_ms', 'stddev_ms', 'p50_ms', 'p95_ms'])
        for key, values in groups.items():
            ordered = sorted(values)
            # Nearest-rank p95 is descriptive; ten trials cannot establish tail latency.
            writer.writerow([*key, len(values), statistics.mean(values),
                             statistics.stdev(values) if len(values) > 1 else 0,
                             statistics.median(values), ordered[math.ceil(.95 * len(values)) - 1]])
    with (out / 'timings.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (out / 'comparisons.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'cpu_wall_ms', 'cuda_wall_ms', 'speedup', 'cuda_output_megapixels_per_second',
                         'cuda_input_megapixels_per_second'])
        for case in sorted({row['case'] for row in rows}):
            cpu = next((r for r in rows if r['case'] == case and r['metric'] == 'CPU fuse (wall)'), None)
            gpu = next((r for r in rows if r['case'] == case and r['metric'] == 'Fuse incl. transfers (wall)'), None)
            if gpu and float(gpu['median_ms']) > 0:
                duration = float(gpu['median_ms'])
                rate = int(gpu['width']) * int(gpu['height']) / (duration * 1000)
                writer.writerow([case, cpu['median_ms'] if cpu else '', duration,
                                 float(cpu['median_ms']) / duration if cpu else '', rate,
                                 rate * int(gpu['exposures'])])
    if not args.keep_raw:
        compact_results(out)
    print(f'Results: {out}')
    incomplete = [s for s in metadata['profile_status'] if s['status'] != 'ok']
    if incomplete:
        print(f'WARNING: {len(incomplete)} profiling passes unavailable/failed; see manifest.json. Timings retained.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        sys.exit(str(error))
