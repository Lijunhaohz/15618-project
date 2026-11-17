# Milestone experiments

Run from the repository root on the Linux CUDA machine:

```sh
python3 scripts/run_experiments.py --save-images
```

Use `--build-dir /tmp/ziyoux-fusion-build --output /tmp/ziyoux-results-RUN`
if AFS quota is limited. Choose a new output directory each run. Copy results
off the machine afterward: `/tmp` is temporary and local to that machine.

The script configures a Release build, runs CTest, then measures CPU/CUDA on
1/3/5/9/15 exposures and depths 1–5. Each case uses two warmups and ten measured
trials. Counts exceeding the available bracket are skipped. Deep profiles cover
3 and 15 exposures at depths 1 and 5 per scene, when available. Full Nsight
Compute sections cover every launch and can be expensive.

## Files to use

| File | Purpose |
|---|---|
| `comparisons.csv` | CPU/CUDA speedup and GPU throughput: start here for report figures |
| `timings.csv` | Median/min/max CPU/CUDA latency and CUDA stage breakdown |
| `device-memory.csv` | Peak live requested GPU-buffer bytes, cumulative bytes and allocation count |
| `errors.csv` | Float maximum absolute error and RMSE |
| `samples.csv`, `distributions.csv` | Individual timing trials, mean/stddev and descriptive p50/p95 |
| `manifest.json` | Exact selected inputs, commands, dataset hashes and profiler status |
| `environment.txt` | Hardware, tools, git state, build configuration and tests |
| `diagnostics.txt` | Unsupported counters and profiler errors, only when present |
| Per-case `cuda.png` (or CPU-only `cpu.png`) | Visual examples: depths for three exposures, plus deepest result for largest measured bracket |
| Per-profile `kernels.ncu-rep`, `ncu-metrics.csv` | Nsight Compute GUI report and machine-readable kernel counters |
| Per-profile `timeline.nsys-rep`, `timeline-summary.txt` | Nsight Systems GUI timeline and built-in API/kernel/transfer statistics |
| Per-profile `cpu-counters.csv` | Consolidated CPU cycles/instructions, branches, caches, TLBs and scheduler counters |
| Per-profile `resources.txt` | CPU/CUDA process host RSS, user/system time, faults and I/O |

Successful benchmark logs are consolidated into tables, commands go in the
manifest, and failed logs go into diagnostics. Duplicate CPU images, profiler
output images, intermediate SQLite databases and per-event logs are removed.
CUDA-host perf counters and generic stalled-cycle events are no longer collected:
the former mostly describe startup/I/O, while GPU stalls come from Nsight
Compute; the latter were unavailable on the measured GHC CPU. CPU call-stack
recording is off by default. `--perf-record` enables it, retaining a text summary
instead of the large raw recording. `--keep-raw` retains intermediate artifacts
for debugging. Unknown/user-created files are not removed by compaction.

Already-transferred old results can be compacted in place:

```sh
python3 scripts/compact_results.py output/RESULT_DIRECTORY
```

## Other experiments

```sh
# More scenes, one aligned bracket per directory
python3 scripts/run_experiments.py --dataset data/memorial \
  --dataset /path/to/another-scene --save-images

# More timing trials and selected deep profiling cases
python3 scripts/run_experiments.py --trials 30 --warmups 5 \
  --profile-count 3 9 --profile-levels 1 5 --save-images

# Expensive deep profiling of every measured case
python3 scripts/run_experiments.py --profile-scope all --save-images

# Timing/memory/correctness only
python3 scripts/run_experiments.py --profile none --save-images

# Local CPU smoke test
python3 scripts/run_experiments.py --backend cpu --profile none \
  --dataset data/basic-test-3 --counts 1 3 --levels 1 3 --trials 1 --warmups 0
```

`--profile` accepts `all` (default), `perf`, `nsys`, `ncu`, `both` (NVIDIA tools),
or `none`. `--ncu-set basic --ncu-launch-count 50` limits counter replay;
`--ncu-launch-count 0` profiles every launch. `--perf-repeats` defaults to three;
`--perf-trials` to ten CPU fuse calls per counter pass. Use `--skip-build
--build-dir PATH` only with an existing tested compatible Release build.

Both supplied directories show the Memorial scene. Add other scenes for
broader visual evidence. To vary resolution, prepare separately resized brackets
and pass their directories with `--dataset`; record identical preprocessing for
every exposure. The script neither downloads nor resizes images. All inputs in
a case must be aligned and match dimensions. Exposure selections span sorted
filenames; one-exposure tests use the middle file. Invalid depths fail explicitly.

## Interpretation

Use matching CPU/CUDA wall times for speedup. I/O is excluded from benchmark
wall timings. CUDA event intervals can include GPU idle time and are not pure
kernel sums. Profilers run separately so their replay/instrumentation overhead
is not part of reported latency. Ten trials give descriptive p95 only.

Device memory counts requested backend buffers, not CUDA context/driver overhead
or other processes. CPU perf counts include startup/loading amortized over ten
fuse calls; resource and NVIDIA profiler passes run the CLI including I/O.
Check perf percent-running values before interpreting multiplexed counters.
Unsupported counters are missing data, never zero. Full Nsight metrics depend
on GPU/tool version; the suite collects available full sections, not every
possible hardware event. Binary timelines are retained even when text exports
are unsupported. Some Nsight versions return success despite reporting an error;
these errors are detected and recorded.

Exit 0 means completion; exit 2 means benchmarks completed with partial,
unavailable or failed profiling passes. Required build/test/benchmark failures
stop the run and retain diagnostic logs. Results are never overwritten.

References: [perf stat](https://man7.org/linux/man-pages/man1/perf-stat.1.html),
[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

## Make graphs and interpret a run

Create a plotting environment once:

```sh
python3 -m venv .venv-plot
.venv-plot/bin/python -m pip install -r scripts/requirements-plot.txt
```

Then generate the report:

```sh
.venv-plot/bin/python scripts/plot_results.py \
  output/ziyoux-fusion-results-20261117-141823
```

This writes only `graphs/report.pdf` and `graphs/interpretation.md` by default.
The PDF includes latency with min/max ranges, speedup, stage breakdown, wall
versus event timing, peak memory/allocation count, numerical error, available
CUDA API costs, kernel throughput/occupancy/registers, and selected fused images.
Unavailable optional evidence is skipped. Each dataset/resolution is handled
separately. The Markdown describes findings from the actual data, highlights
anomalies, and explains which optimization each plot can motivate.

Add `--png` for separate report/poster PNG figures. Use `--output PATH` to
choose another figure directory. Existing generated charts there are replaced.
Timing/error/memory tables and profiler reports remain unchanged.

Read plots in this order: correctness, wall speedup, stage costs, allocation
costs/memory, then kernel counters. GPU event intervals and Nsight CLI timings
have different scope; never add them together or subtract their medians to
claim exact transfer overhead. Nsight kernel charts show medians by kernel type
across launches at different pyramid sizes, not a utilization measure for the
whole pipeline. High bandwidth utilization alone does not prove memory-bound
execution. Use the GUI for per-level launch details and warp-stall analysis.
For anomalous timing points, repeat a timing-only `--profile none` sweep on an
idle machine before reporting a scaling claim.

## Suggested input sources

### Aligned exposure brackets

- **Small starting set:** Browse the
  [OpenCV HDR exposure test images](https://github.com/opencv/opencv_extra/tree/4.x/testdata/cv/hdr/exposures)
  and download several `memorialNN.png` files from the same sequence. Use
  GitHub's download controls or the raw file links. Check their dimensions and
  visually inspect registration before using them; record the source and
  selected filenames.
- **Broader evaluation:** The [SICE dataset](https://github.com/csjcai/SICE#dataset)
  provides downloads for multi-exposure sequences. Follow the dataset page's
  Google Drive or Baidu links, select images belonging to the same sequence,
  and verify alignment and dimensions. Review the dataset's usage terms and
  citation requirements before distributing images or results.
- **Own captures:** Use a tripod and exposure bracketing on a static scene.
  Keep the camera and focus fixed, inspect for movement, and export the
  exposures at the same dimensions. No image alignment is included in the
  baseline.

Keep dataset downloads separate from source code, and record the source,
license or usage terms, resolution, exposure count, and any preprocessing for
each experiment.

### Synthetic inputs

Synthetic test images are more reproducible when generated locally than
downloaded. For example, the following standard-library Python command writes
three odd-sized constant RGB images and two identical images with a corner
impulse as binary PPM files:

```sh
python3 - <<'PY'
from pathlib import Path

out = Path("test-inputs")
out.mkdir(exist_ok=True)
width, height = 7, 5

def write_ppm(name, pixels):
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    (out / name).write_bytes(header + bytes(pixels))

for value in (64, 128, 192):
    write_ppm(f"constant-{value}.ppm", [value, value, value] * width * height)

impulse = [0] * (width * height * 3)
impulse[:3] = [255, 255, 255]
for index in (0, 1):
    write_ppm(f"impulse-{index}.ppm", impulse)
PY
```

Use the constant images as one bracket and the impulse images as another.
The constant bracket exercises zero-measure normalization; identical impulse
images exercise boundary handling and reconstruction. The baseline image
loader must support PPM for these files, or they can be converted losslessly
to a supported RGB format. Also test a single image, one-pixel dimensions,
and mismatched sizes.
