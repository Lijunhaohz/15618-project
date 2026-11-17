"""Optional profiling passes. Failures are data, never a reason to discard timings."""
import json
from pathlib import Path
import platform
import shutil
import subprocess

# Small independent passes reduce counter multiplexing; unsupported groups get
# retried event by event, preserving all successes and failures in the logs.
PERF_GROUPS = {
    'scheduler': ['task-clock', 'context-switches', 'cpu-migrations', 'page-faults', 'minor-faults', 'major-faults'],
    'instructions': ['cycles', 'instructions', 'branches', 'branch-misses'],
    'cache': ['cache-references', 'cache-misses'],
    'l1': ['L1-dcache-loads', 'L1-dcache-load-misses', 'L1-icache-load-misses'],
    'llc': ['LLC-loads', 'LLC-load-misses', 'LLC-stores', 'LLC-store-misses'],
    'tlb': ['dTLB-loads', 'dTLB-load-misses', 'iTLB-loads', 'iTLB-load-misses'],
}


def collect(args, build, case_dir, selected, levels, backends, run):
    statuses = []

    def attempt(tool, name, command):
        log = case_dir / (name + '.log')
        if not shutil.which(tool):
            status = {'name': name, 'status': 'unavailable', 'reason': tool + ' not found'}
        else:
            code = run(command, log, required=False)
            if code == 0 and log.exists() and any(line.startswith('ERROR:') for line in log.read_text().splitlines()):
                code = 1  # Some Nsight releases return success for failed exports.
            status = {'name': name, 'status': 'ok' if code == 0 else 'failed',
                      'returncode': code, 'log': str(log)}
        if status['status'] == 'ok' and '-perf-' in name:
            counter_file = case_dir / (name + '.csv')
            if counter_file.exists() and any(marker in counter_file.read_text() for marker in
                                             ('<not supported>', '<not counted>')):
                status['status'] = 'partial'
                status['reason'] = 'some counters unsupported/not counted; inspect counter CSV'
        statuses.append(status)
        # Persist immediately so an interrupted full-metric replay remains auditable.
        (case_dir / 'profile-status.json').write_text(json.dumps(statuses, indent=2))
        return status['status'] == 'ok'

    def application(backend):
        return [build / 'exposure_fusion', '--backend', backend, '--levels', levels,
                '--output', case_dir / (backend + '-profile.png'), '--', *selected]

    for backend in backends:
        command = application(backend)
        if platform.system() == 'Linux' and Path('/usr/bin/time').exists():
            attempt('/usr/bin/time', backend + '-resources',
                    ['/usr/bin/time', '-v', '-o', case_dir / (backend + '-resources.txt'), *command])
        if backend != 'cpu' or args.profile not in ('all', 'perf'):
            continue
        # CPU benchmark loads input once, amortizing startup/I/O over multiple fuse calls.
        # CUDA CLI avoids contaminating host counters with the CPU reference validation.
        workload = ([build / 'cpu_benchmark', '--levels', levels, '--warmups', 0,
                     '--trials', args.perf_trials, '--', *selected] if backend == 'cpu' else command)
        for group, events in PERF_GROUPS.items():
            name = backend + '-perf-' + group

            def counter_pass(label, event_list):
                return attempt('perf', label, ['perf', 'stat', '--no-big-num', '-x', ';',
                    '-r', args.perf_repeats, '-e', ','.join(event_list),
                    '-o', case_dir / (label + '.csv'), '--', *workload])

            if not counter_pass(name, events) and shutil.which('perf'):
                # Permission denial affects every event; don't rerun the workload 20 times.
                log = (case_dir / (name + '.log')).read_text()
                result = case_dir / (name + '.csv')
                if result.exists():
                    log += result.read_text()
                if not any(word in log.lower() for word in ('permission', 'perf_event_paranoid', 'access denied')):
                    for event in events:
                        counter_pass(backend + '-perf-' + event, [event])
            if not shutil.which('perf'):
                break
        if args.perf_record:
            data = case_dir / (backend + '-perf.data')
            if attempt('perf', backend + '-perf-record', ['perf', 'record', '-F', 99,
                       '--call-graph', 'dwarf', '-o', data, '--', *workload]):
                attempt('perf', backend + '-perf-report', ['perf', 'report', '--stdio', '-i', data])
    if 'cuda' in backends:
        command = application('cuda')
        if args.profile in ('all', 'both', 'nsys'):
            prefix = case_dir / 'timeline'
            if attempt('nsys', 'nsys', ['nsys', 'profile', '--trace=cuda,osrt',
                       '--sample=none', '--cpuctxsw=none', '--cuda-memory-usage=true',
                       '--stats=true', '--output', prefix, *command]):
                pass  # Built-in --stats output is retained as timeline-summary.txt.
        if args.profile in ('all', 'both', 'ncu'):
            extra = [] if args.ncu_launch_count == 0 else ['--launch-count', args.ncu_launch_count]
            prefix = case_dir / 'kernels'
            if attempt('ncu', 'ncu', ['ncu', '--set', args.ncu_set, *extra,
                       '--export', prefix, *command]):
                attempt('ncu', 'ncu-metrics', ['ncu', '--import', str(prefix) + '.ncu-rep',
                        '--page', 'raw', '--csv', '--log-file', case_dir / 'ncu-metrics.csv'])
    return statuses
