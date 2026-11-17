#!/usr/bin/env python3
"""Consolidate known experiment artifacts, retaining optimization/report evidence."""
import argparse
import csv
import json
from pathlib import Path
import re


def compact_results(root):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists() or not (root / 'timings.csv').exists():
        raise ValueError('Not a completed experiment directory (manifest.json/timings.csv required)')
    manifest = json.loads(manifest_path.read_text())
    remove = set()
    diagnostic_file = root / 'diagnostics.txt'
    diagnostics = diagnostic_file.read_text().splitlines() if diagnostic_file.exists() else []
    cases = manifest.setdefault('cases', {})
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and re.match(r'^s\d+-.+-n\d+-l\d+$', p.name)):
        case = cases.setdefault(directory.name, {})
        inputs = directory / 'inputs.json'
        if inputs.exists():
            case['inputs'] = json.loads(inputs.read_text())
            remove.add(inputs)
        # Keep exact commands, not duplicate benchmark output already in tables.
        commands = case.setdefault('commands', [])
        for name in ['cpu.log', 'cuda.log']:
            log = directory / name
            if log.exists():
                first = log.read_text().splitlines()
                if first and first[0].startswith('$ ') and first[0][2:] not in commands:
                    commands.append(first[0][2:])
                remove.add(log)
        statuses = directory / 'profile-status.json'
        if statuses.exists():
            case['profile_status'] = json.loads(statuses.read_text())
            remove.add(statuses)
        for status in case.get('profile_status', []):
            if status.get('status') != 'ok':
                diagnostics.append(f"{directory.name}: {status['name']}: {status['status']}: {status.get('reason', '')}")
        # One normalized CPU counter table; CUDA-host perf counters primarily
        # describe startup/I/O and are not GPU bottleneck metrics.
        counters = {}
        for path in sorted(directory.glob('cpu-perf-*.csv')):
            for row in csv.reader(path.read_text().splitlines(), delimiter=';'):
                if len(row) >= 3 and row[0] and not row[0].startswith('#'):
                    if row[0].startswith('<'):
                        diagnostics.append(f'{directory.name}: CPU counter {row[2]} {row[0]}')
                    else:
                        counters[row[2]] = row
            remove.add(path)
        if counters:
            with (directory / 'cpu-counters.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['value', 'unit', 'event', 'variation', 'runtime_ns', 'running_percent', 'derived_value', 'derived_unit'])
                writer.writerows(counters.values())
        for path in directory.glob('cuda-perf-*.csv'):
            remove.add(path)
        for path in directory.glob('*-perf.data'):
            remove.add(path)
        # Preserve the useful text call-stack report if explicitly collected,
        # but remove raw sampling data, which dwarfs the resulting summaries.
        for path in directory.glob('*-resources.txt'):
            target = directory / 'resources.txt'
            with target.open('a') as stream:
                stream.write('\n' + path.name + '\n' + path.read_text())
            remove.add(path)
        timeline = directory / 'nsys.log'
        if timeline.exists():
            lines = timeline.read_text().splitlines()
            clean = [line for line in lines if line.strip() and not re.match(r'^\[\d+/\d+\]', line)]
            (directory / 'timeline-summary.txt').write_text('\n'.join(clean) + '\n')
        metrics = directory / 'ncu-metrics.csv'
        if metrics.exists():
            lines = metrics.read_text().splitlines()
            start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
            if start is not None:
                diagnostics.extend(f'{directory.name}: {line}' for line in lines[:start] if line.strip())
                metrics.write_text('\n'.join(lines[start:]) + '\n')
        for log in directory.glob('*.log'):
            if log.name == 'cpu-perf-report.log':
                log.rename(directory / 'cpu-callstacks.txt')
                continue
            text = log.read_text()
            errors = [line for line in text.splitlines() if line.startswith('ERROR:')]
            diagnostics.extend(f'{directory.name}/{log.name}: {line}' for line in errors)
            # Failed passes retain their output in one diagnostic file.
            if any(s.get('name') == log.stem and s.get('status') == 'failed'
                   for s in case.get('profile_status', [])):
                diagnostics.append(f'{directory.name}/{log.name}:\n{text}')
            remove.add(log)
        for name in ['cpu-profile.png', 'cuda-profile.png', 'timeline.sqlite', 'cpu.png']:
            path = directory / name
            if path.exists():
                remove.add(path)
        # Fused image examples: all depths for a 3-exposure bracket, plus the
        # largest measured bracket at its deepest level. Preserve CPU-only output.
        if not (directory / 'cuda.png').exists():
            remove.discard(directory / 'cpu.png')
        match = re.match(r'^(.*)-n(\d+)-l(\d+)$', directory.name)
        scene, count, level = match[1], int(match[2]), int(match[3])
        siblings = [re.match(r'^(.*)-n(\d+)-l(\d+)$', p.name) for p in root.iterdir() if p.is_dir()]
        related = [m for m in siblings if m and m[1] == scene]
        useful_image = count == 3 or (count == max(int(m[2]) for m in related) and level == max(int(m[3]) for m in related))
        if not useful_image:
            remove.update(p for p in [directory / 'cpu.png', directory / 'cuda.png'] if p.exists())
        # Profile statuses refer to commands/diagnostics in this manifest now.
        for status in case.get('profile_status', []):
            prefix = f"{directory.name}/{status['name']}.log: ERROR:"
            if any(line.startswith(prefix) for line in diagnostics):
                status['status'] = 'failed'
                status['reason'] = 'tool reported ERROR despite returning success; see diagnostics.txt'
            status.pop('log', None)
    environment_names = ['cpu-info.txt', 'gpu-info.txt', 'cuda-version.txt', 'perf-version.txt',
                         'nsys-version.txt', 'ncu-version.txt', 'git-revision.txt', 'git-status.txt',
                         'git-diff.txt', 'CMakeCache.txt', 'configure.log', 'build.log', 'tests.log']
    pieces = []
    for name in environment_names:
        path = root / name
        if path.exists():
            pieces.append(f'=== {name} ===\n{path.read_text()}')
            remove.add(path)
    if pieces:
        with (root / 'environment.txt').open('a') as stream:
            stream.write('\n\n'.join(pieces) + '\n')
    if (root / 'perf-events.txt').exists():
        remove.add(root / 'perf-events.txt')
    manifest['profile_status'] = [dict(status, case=name) for name, case in cases.items()
                                  for status in case.get('profile_status', [])]
    manifest['output_format'] = 'compact-v1'
    (root / 'README.md').write_text('''# Experiment results

Start with comparisons.csv (speedup), timings.csv (stage breakdown),
device-memory.csv (peak requested GPU buffers), and errors.csv (correctness).
Use distributions.csv and samples.csv to assess run-to-run variability.

Profiled case directories contain kernels.ncu-rep (Nsight Compute GUI),
ncu-metrics.csv (kernel counters), timeline.nsys-rep (Nsight Systems GUI),
timeline-summary.txt (API/kernel/transfer statistics), cpu-counters.csv,
and resources.txt where available. Selected PNGs show depth-dependent quality.

manifest.json records exact inputs, commands and profiling status.
environment.txt records hardware, tools, build flags and tests.
diagnostics.txt records unsupported counters and errors, when present.
CPU counters are process totals; GPU memory excludes driver/context overhead.
''')
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    if diagnostics:
        (root / 'diagnostics.txt').write_text('\n'.join(dict.fromkeys(diagnostics)) + '\n')
    # All information is consolidated before removing recognized generated files.
    removed_bytes = sum(p.stat().st_size for p in remove if p.exists())
    for path in remove:
        path.unlink(missing_ok=True)
    for directory in root.iterdir():
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return len(remove), removed_bytes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    count, size = compact_results(args.directory)
    print(f'Consolidated results; removed {count} generated files ({size / 1024**2:.1f} MiB).')
