#!/usr/bin/env python3
"""Create a compact PDF of benchmark graphs plus a data-driven interpretation."""
import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'fusion-matplotlib'))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    raise SystemExit('Install plotting dependencies: python3 -m pip install -r scripts/requirements-plot.txt')

STAGES = [('Quality (events)', 'Weights'), ('Normalization (events)', 'Normalize'),
          ('Gaussian pyramids (events)', 'Gaussian'), ('Laplacian pyramids (events)', 'Laplacian'),
          ('Blend (events)', 'Blend'), ('Reconstruction (events)', 'Reconstruct')]
CPU = 'CPU fuse (wall)'
CUDA = 'Fuse incl. transfers (wall)'


def read_csv(root, name, required=False):
    path = root / name
    if not path.exists():
        if required:
            raise ValueError(f'Missing {path}')
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load(root):
    timings = read_csv(root, 'timings.csv', True)
    cases = {}
    for row in timings:
        name = row['case']
        case = cases.setdefault(name, dict(name=name, dataset=row['dataset'], width=int(row['width']),
                                          height=int(row['height']), n=int(row['exposures']),
                                          level=int(row['levels']), metrics={}))
        case['metrics'][row['metric']] = {key: float(row[key + '_ms']) for key in ['median', 'min', 'max']}
    if not cases:
        raise ValueError('No timing rows')
    # Keep datasets separate, including scenes with identical directory names.
    for case in cases.values():
        prefix = re.sub(r'-n\d+-l\d+$', '', case['name'])
        case['scene'] = f"{prefix} ({case['width']}×{case['height']})"
    memory = defaultdict(list)
    for row in read_csv(root, 'device-memory.csv'):
        memory[row['case']].append(row)
    return cases, memory, read_csv(root, 'errors.csv')


def median(case, metric):
    return case['metrics'].get(metric, {}).get('median', math.nan)


def nsight_api(path):
    """Parse old/new built-in text summaries; retain CLI startup scope."""
    if not path.exists():
        return []
    active = False
    rows = []
    for line in path.read_text().splitlines():
        if 'CUDA API Statistics:' in line:
            active = True
            continue
        if active and 'Statistics:' in line:
            active = False
        if active:
            cells = line.split()
            if len(cells) >= 9:
                try:
                    rows.append((cells[-1], float(cells[0]), float(cells[1].replace(',', '')) / 1e6,
                                 int(cells[2].replace(',', ''))))
                except ValueError:
                    pass
    return rows


def kernel_counters(path):
    """Read wide Nsight raw exports, skipping units/header rows and N/A values."""
    if not path.exists():
        return {}
    groups = defaultdict(lambda: defaultdict(list))
    fields = ['sm__throughput.avg.pct_of_peak_sustained_elapsed',
              'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
              'sm__warps_active.avg.pct_of_peak_sustained_active',
              'launch__registers_per_thread']
    with path.open(newline='') as stream:
        lines = stream.readlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        return {}
    for row in csv.DictReader(lines[start:]):
        match = re.search(r'(\w+_kernel)\(', row.get('Kernel Name', ''))
        if not match:
            continue
        for field in fields:
            try:
                value = float(row.get(field, '').replace(',', ''))
                if math.isfinite(value):
                    groups[match[1]][field].append(value)
            except (ValueError, AttributeError):
                pass
    return groups


def make_report(root, out, png=False):
    cases, memory, errors = load(root)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .2, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    notes = ['# How to interpret these results', '',
             'Wall timings exclude image I/O. CUDA wall time includes transfers and allocations. '
             'Event intervals can contain GPU idle time. Every point is a recorded measurement; '
             'lines only connect tested cases. Min/max whiskers are observed ranges, not confidence intervals.', '']
    pages = []
    scenes = defaultdict(list)
    for case in cases.values():
        scenes[case['scene']].append(case)
    with PdfPages(out / 'report.pdf') as pdf:
        def save(fig, slug, caption):
            fig.text(.06, .02, caption, fontsize=8, color='#444444', wrap=True)
            fig.tight_layout(rect=(0, .08, 1, .95))
            pdf.savefig(fig)
            if png:
                fig.savefig(out / (slug + '.png'), dpi=180)
            pages.append(slug)
            plt.close(fig)

        for index, (scene, data) in enumerate(sorted(scenes.items())):
            data.sort(key=lambda c: (c['n'], c['level']))
            counts = sorted({c['n'] for c in data})
            depths = sorted({c['level'] for c in data})
            colors = {n: plt.get_cmap('tab10')(i % 10) for i, n in enumerate(counts)}
            notes += [f'## {scene}', '', f'Measured exposure counts: {counts}; pyramid depths: {depths}.', '']
            fig, axes = plt.subplots(1, 2, figsize=(11.7, 6.5))
            for ax, metric, title in zip(axes, [CPU, CUDA], ['CPU wall latency', 'CUDA wall latency including transfers']):
                for n in counts:
                    subset = [c for c in data if c['n'] == n and metric in c['metrics']]
                    if not subset:
                        continue
                    y = [median(c, metric) for c in subset]
                    low = [max(0, v - c['metrics'][metric]['min']) for c, v in zip(subset, y)]
                    high = [max(0, c['metrics'][metric]['max'] - v) for c, v in zip(subset, y)]
                    ax.errorbar([c['level'] for c in subset], y, yerr=[low, high], marker='o',
                                capsize=3, color=colors[n], label=f'{n} exposures')
                ax.set(title=title, xlabel='Pyramid levels', ylabel='Median time (ms)', xticks=depths)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8)
            fig.suptitle(scene)
            save(fig, f'{index}-latency', 'Lower is better. Whiskers show min–max; CPU and CUDA panels have independent y scales.')

            paired = [c for c in data if CPU in c['metrics'] and CUDA in c['metrics']]
            if paired:
                fig, ax = plt.subplots(figsize=(11.7, 6.5))
                for level in depths:
                    subset = [c for c in paired if c['level'] == level]
                    ax.plot([c['n'] for c in subset], [median(c, CPU) / median(c, CUDA) for c in subset],
                            marker='o', label=f'{level} levels')
                ax.axhline(1, color='black', linestyle='--', label='CPU/CUDA parity')
                ax.set(title=f'{scene}: wall-time speedup', xlabel='Exposure count', ylabel='CPU wall / CUDA wall (×)', xticks=counts)
                ax.legend(ncol=3)
                save(fig, f'{index}-speedup', 'Above 1×: CUDA is faster. Use matching parameters; pipeline-only event time is not an end-to-end speedup denominator.')
                best = max(paired, key=lambda c: median(c, CPU) / median(c, CUDA))
                slow = [c for c in paired if median(c, CPU) < median(c, CUDA)]
                notes += [f"- Best observed wall speedup: **{median(best, CPU) / median(best, CUDA):.2f}×**, "
                          f"{best['n']} exposures, {best['level']} levels.",
                          f'- CUDA is slower than CPU in {len(slow)} of {len(paired)} matched cases.']
            gpu = [c for c in data if CUDA in c['metrics']]
            if gpu:
                focus_n = 3 if 3 in counts else counts[-1]
                selections = [[c for c in gpu if c['level'] == depths[-1]],
                              [c for c in gpu if c['n'] == focus_n]]
                fig, axes = plt.subplots(1, 2, figsize=(11.7, 6.5))
                for ax, subset, title in zip(axes, selections, [f'{depths[-1]} levels: exposure scaling', f'{focus_n} exposures: depth scaling']):
                    bottom = [0.] * len(subset)
                    for metric, label in STAGES:
                        values = [median(c, metric) if metric in c['metrics'] else 0 for c in subset]
                        ax.bar(range(len(subset)), values, bottom=bottom, label=label)
                        bottom = [a + b for a, b in zip(bottom, values)]
                    ax.set(xticks=range(len(subset)), xticklabels=[f"N={c['n']}\nL={c['level']}" for c in subset],
                           title=title, ylabel='Median event interval (ms)')
                axes[1].legend(fontsize=8, ncol=2)
                fig.suptitle(scene + ': CUDA stage breakdown')
                save(fig, f'{index}-stages', 'Stacks sum independent stage medians; they need not equal the pipeline median. Event intervals include possible idle time.')
                focus = next((c for c in gpu if c['n'] == focus_n and c['level'] == depths[-1]), gpu[-1])
                values = [(label, median(focus, metric)) for metric, label in STAGES if metric in focus['metrics']]
                total = sum(v for _, v in values)
                if total:
                    label, duration = max(values, key=lambda pair: pair[1])
                    notes += [f"- At {focus['n']} exposures/{focus['level']} levels, **{label}** is the largest event interval "
                              f'({duration:.3f} ms; {duration / total:.1%} of summed stage medians).']
                fig, ax = plt.subplots(figsize=(11.7, 6.5))
                labels = [f"{c['n']} / {c['level']}" for c in gpu]
                ax.plot(range(len(gpu)), [median(c, CUDA) for c in gpu], 'o-', label='Wall including transfers')
                ax.plot(range(len(gpu)), [median(c, 'Pipeline (events)') for c in gpu], 's-', label='Event pipeline')
                ax.set(title=scene + ': wall versus event timing', xticks=range(len(gpu)), xticklabels=labels,
                       xlabel='Exposure count / pyramid levels', ylabel='Median time (ms)')
                ax.tick_params(axis='x', labelrotation=60)
                ax.legend()
                save(fig, f'{index}-wall-vs-events', 'Separate measurements. Their difference is not a direct transfer-time measurement: allocation, cleanup, host work and run variability contribute.')
                # Flag depth growth at fixed input count for follow-up, without discarding points.
                for n in counts:
                    subset = [c for c in gpu if c['n'] == n]
                    times = [median(c, CUDA) for c in subset]
                    if len(times) > 1 and max(times) > 2 * min(times):
                        notes += [f'- **Investigate:** N={n} CUDA latency spans {min(times):.3f}–{max(times):.3f} ms across depths. '
                                  'Repeat a timing-only --profile none sweep on an idle machine; inspect timeline, GPU clocks/contention and allocation/synchronization before attributing the change to filtering.']
            mem_cases = [c for c in data if memory[c['name']]]
            if mem_cases:
                fig, axes = plt.subplots(1, 2, figsize=(11.7, 6.5))
                for level in depths:
                    subset = [c for c in mem_cases if c['level'] == level]
                    for ax, field, divisor in [(axes[0], 'peak_bytes', 2**20), (axes[1], 'allocations', 1)]:
                        ax.plot([c['n'] for c in subset], [max(int(r[field]) for r in memory[c['name']]) / divisor for c in subset],
                                marker='o', label=f'{level} levels')
                axes[0].set(title='Peak requested GPU buffers', ylabel='MiB')
                axes[1].set(title='Device allocations per fuse', ylabel='Allocation count')
                for ax in axes:
                    ax.set(xlabel='Exposure count', xticks=counts)
                    ax.legend(fontsize=8)
                fig.suptitle(scene)
                save(fig, f'{index}-memory', 'Maximum observed across instrumented trials. Requested buffer sizes exclude CUDA context/driver memory and other processes.')
                maximum = max(int(r['peak_bytes']) for c in mem_cases for r in memory[c['name']])
                leaks = sum(int(r['live_bytes']) != 0 for c in mem_cases for r in memory[c['name']])
                notes += [f'- Largest measured requested buffer peak: **{maximum / 2**20:.2f} MiB**. '
                          f'Trials with nonzero live bytes after cleanup: **{leaks}**.']
            selected_errors = [r for r in errors if r['case'] in {c['name'] for c in data} and r['comparison'] == 'Uninstrumented CUDA vs CPU']
            if selected_errors:
                lookup = {r['case']: r for r in selected_errors}
                ordered = [c for c in data if c['name'] in lookup]
                fig, axes = plt.subplots(1, 2, figsize=(11.7, 6.5))
                for ax, field, tolerance in [(axes[0], 'max_abs', 2e-5), (axes[1], 'rmse', 2e-6)]:
                    ax.plot(range(len(ordered)), [float(lookup[c['name']][field]) for c in ordered], 'o')
                    ax.axhline(tolerance, color='red', linestyle='--', label=f'Test tolerance {tolerance:g}')
                    ax.set(yscale='symlog', title=field, ylabel='Absolute float error', xlabel='Exposure count / pyramid levels',
                           xticks=range(len(ordered)), xticklabels=[f"{c['n']}/{c['level']}" for c in ordered])
                    ax.set_yscale('symlog', linthresh=1e-9)
                    ax.tick_params(axis='x', labelrotation=60)
                    ax.legend(fontsize=8)
                fig.suptitle(scene + ': numerical correctness')
                save(fig, f'{index}-correctness', 'Lower is better. Symlog shows exact zeros without replacing them; linear region below 1e−9. Compare floats, not exported 8-bit PNGs.')
                notes += [f"- Worst ordinary CUDA/CPU error: max absolute **{max(float(r['max_abs']) for r in selected_errors):.3g}**, "
                          f"RMSE **{max(float(r['rmse']) for r in selected_errors):.3g}**."]
            # CLI profile API breakdown: useful evidence for buffer reuse, not wall benchmark totals.
            api_cases = [(c, nsight_api(root / c['name'] / 'timeline-summary.txt')) for c in data]
            api_cases = [(c, rows) for c, rows in api_cases if rows]
            if api_cases:
                fig, axes = plt.subplots(1, len(api_cases), figsize=(11.7, 6.5), squeeze=False)
                for ax, (c, rows) in zip(axes[0], api_cases):
                    top = sorted(rows, key=lambda r: r[2], reverse=True)[:6]
                    ax.barh([r[0] for r in reversed(top)], [r[2] for r in reversed(top)], color='#4878a8')
                    ax.set(title=f"N={c['n']}, L={c['level']}", xlabel='CUDA API host duration (ms)')
                fig.suptitle(scene + ': top CUDA API costs in profiled CLI')
                save(fig, f'{index}-api', 'Profiled process includes CUDA startup and I/O. API duration can include waiting for GPU work; do not add it to kernel time.')
                for c, rows in api_cases:
                    name, share, duration, calls = max(rows, key=lambda r: r[2])
                    notes += [f"- Profiled CLI N={c['n']}/L={c['level']}: largest CUDA API cost is **{name}** "
                              f'({duration:.2f} ms, {share:.1f}% of CUDA API time, {calls} calls). Includes startup/replay effects; validate with steady-state timings.']
            counter_cases = [(c, kernel_counters(root / c['name'] / 'ncu-metrics.csv'))
                             for c in data if c['level'] == depths[-1]]
            for c, groups in counter_cases:
                if not groups:
                    continue
                fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
                fields = [('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'SM throughput (% of peak)'),
                          ('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'DRAM throughput (% of peak)'),
                          ('sm__warps_active.avg.pct_of_peak_sustained_active', 'Achieved occupancy (%)'),
                          ('launch__registers_per_thread', 'Registers per thread')]
                for ax, (field, title) in zip(axes.flat, fields):
                    available = [(name.replace('_kernel', ''), statistics.median(values[field]))
                                 for name, values in sorted(groups.items()) if values[field]]
                    ax.barh([name for name, _ in available], [value for _, value in available], color='#4878a8')
                    ax.set_title(title)
                    if '%' in title:
                        ax.set_xlim(0, max(100, max((v for _, v in available), default=0)))
                fig.suptitle(f"{scene}: kernel counters, N={c['n']}, L={c['level']}")
                save(fig, f"{index}-kernel-counters-n{c['n']}",
                     'Medians across profiled launches of each kernel type, including different pyramid sizes. Replay/cache effects may alter counters; high throughput alone does not prove a bottleneck.')
            images = [(c, root / c['name'] / 'cuda.png') for c in data if c['n'] == 3]
            images = [(c, p) for c, p in images if p.exists()]
            if images:
                fig, axes = plt.subplots(1, len(images), figsize=(11.7, 6.5), squeeze=False)
                for ax, (c, path) in zip(axes[0], images):
                    ax.imshow(plt.imread(path))
                    ax.set_title(f"{c['level']} levels")
                    ax.axis('off')
                fig.suptitle(scene + ': three-exposure visual comparison')
                save(fig, f'{index}-images', 'Compare halos, seams and shadow/highlight detail. Changing depth changes the algorithm’s output, not just performance.')
            notes += ['']
    notes += ['## Choosing the next optimization', '',
              '- Pyramid stages dominate: use kernel counters to check memory traffic/cache behavior, then try tiling/filter fusion.',
              '- Allocation/API overhead dominates: prioritize lifetime-based buffer reuse and fewer synchronizations.',
              '- Peak buffer bytes grow with exposure count: compare incremental processing against the resident baseline.',
              '- Treat anomalous timings as measurements needing reproduction; never silently remove them.',
              '- These results cover the supplied scenes/resolutions only; they do not establish general scaling across resolutions.',
              '', f'Generated {len(pages)} PDF pages. Optional PNG export uses --png.']
    (out / 'interpretation.md').write_text('\n'.join(notes) + '\n')
    return len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, help='default: RESULTS/graphs')
    parser.add_argument('--png', action='store_true', help='also export individual PNG figures for the report/poster')
    args = parser.parse_args()
    root = args.results.resolve()
    out = args.output or root / 'graphs'
    try:
        pages = make_report(root, out, args.png)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, str(error) + '\n')
    print(f'Created {pages} graph pages: {out / "report.pdf"}')
    print(f'Interpretation: {out / "interpretation.md"}')


if __name__ == '__main__':
    main()
