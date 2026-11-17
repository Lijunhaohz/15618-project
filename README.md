# CUDA-Accelerated Multi-Exposure Image Fusion

15-618 project by Estella Xie (`ziyoux`) and Junhao Li (`junhaoli`).

## Introduction

This project explores CUDA-accelerated multi-exposure image fusion: combining
aligned photographs of the same scene taken at different exposure levels to
preserve detail in both bright and dark regions. Following the Mertens exposure
fusion algorithm, the planned pipeline computes image quality weights, blends
multi-scale image pyramids, and reconstructs a single fused image.

We plan to compare a CPU reference with a CUDA implementation, then investigate
how kernel fusion, shared-memory tiling, buffer reuse, and incremental processing
affect runtime and GPU memory usage. A memory-aware execution planner and
overlapping data transfers with computation are stretch goals. See
[proposal.md](docs/proposal.md) for the full proposal and schedule.

## Algorithm Specification

See [Algorithm specification and inputs](docs/algorithm.md)

## Building and running

Requirements: CMake, C99 and C++17 compilers, and libpng. CUDA execution also
requires the CUDA toolkit, a compatible host compiler, and an accessible GPU.
From the project root, build and run the CPU backend:

```sh
./scripts/run_cpu_sample.sh --output fused.png data/basic-test-3/
```

On Unix, build and run the CUDA backend in one command:

```sh
./scripts/run_cuda_sample.sh --output fused.png data/basic-test-3/
```

The script uses `build/cuda` by default. Set `CUDACXX`, `CUDAHOSTCXX`, or
`FUSION_CUDA_BUILD_DIR` to override its compiler, host compiler, or build
directory; use a fresh build directory when changing compilers. A CUDA request
fails explicitly if CUDA is unavailable, rather than falling back to CPU.
The CPU script uses `build/cpu` by default, overridable with
`FUSION_CPU_BUILD_DIR`.

Inputs must be aligned, identically sized 8-bit RGB PNG or binary P6 PPM
images. Files and directories can be mixed; directories are read
non-recursively in filename order, ignoring `.md`, `.txt`, `.json`, `.csv`,
and `.DS_Store` metadata. Other unsupported files and empty directories cause
errors. Output is PNG or PPM; existing output is overwritten, so keep it
outside input directories. Add `--levels N` to select pyramid depth (default:
up to five), or use `--help` for options.

After running the corresponding script, build and run tests and timings:

```sh
./scripts/run_cpu_benchmark.sh
./scripts/run_cuda_benchmark.sh
```

Each benchmark script uses its runner's build directory. Both default to
`data/basic-test-3/`; pass image files or directories as arguments to select
other inputs. The CPU benchmark measures `fuse()` wall time (excluding image
I/O), with two warmups and ten trials; it does not report individual stages.
The real-image test
requires the three supplied Memorial images; their alignment is assumed, not
independently verified. See the [Memorial notes](data/memorial/README.md) for
provenance and usage cautions.

### Milestone experiment sweeps and GPU profiling

See [experiment instructions](docs/experiments.md) for CPU/CUDA sweeps across
pyramid depth, exposure count, and scene directories, CSV output, saved images,
device allocation accounting, Linux perf counters/call stacks, and full Nsight
Systems/Compute profiles. Start with:

```sh
python3 scripts/run_experiments.py --save-images
```

Generate graphs from a completed run with `scripts/plot_results.py`; see the
[plotting setup and interpretation guide](docs/experiments.md#make-graphs-and-interpret-a-run).
The default output is one PDF and one Markdown interpretation.

### Preliminary results

Results for the default sample input, in milliseconds:

| Measurement | Median | Min | Max |
|---|---:|---:|---:|
| CPU `fuse()` (wall) | 733.402 | 732.561 | 733.939 |
| CUDA quality weights (events) | 0.111 | 0.108 | 0.115 |
| CUDA normalization (events) | 0.036 | 0.035 | 0.037 |
| CUDA Gaussian pyramids (events) | 1.462 | 1.455 | 1.469 |
| CUDA Laplacian pyramids (events) | 1.183 | 1.177 | 1.200 |
| CUDA blend (events) | 0.268 | 0.267 | 0.271 |
| CUDA reconstruction (events) | 0.701 | 0.696 | 0.706 |
| CUDA pipeline (events) | 3.764 | 3.745 | 3.775 |
| CUDA `fuse()` including transfers (wall) | 38.867 | 38.810 | 39.020 |

CUDA output versus CPU: maximum absolute error `3.58e-7`, RMSE `1.74e-8`.
CPU and CUDA `fuse()` times exclude image I/O. CUDA event timings can include
GPU idle time and are not pure kernel times.

## Milestone report and project website

The milestone report is [docs/milestone.md](docs/milestone.md), with the full
submission PDF at [docs/milestone.pdf](docs/milestone.pdf). The project homepage
shows the milestone and revised schedule; [proposal.html](proposal.html)
preserves the original proposal.

After editing the report, rebuild the site:

```sh
python3 scripts/build_site.py
python3 -m http.server 8000
```

Open http://localhost:8000. The script reads `docs/milestone.md` and
`docs/proposal.md`; generated pages and report figures should be committed with
the source documents. Publishing requires deploying the repository through
GitHub Pages. The expected project URL is
https://Lijunhaohz.github.io/15618-project/; local generation does not publish it.

To rebuild the PDF, install `scripts/requirements-report.txt` in the plotting
virtual environment, then run:

```sh
.venv-plot/bin/python -m pip install -r scripts/requirements-report.txt
.venv-plot/bin/python scripts/build_milestone_pdf.py
```

The two curated figures are already included in `docs/imgs`. To regenerate them
from the measured run, append
`--results output/ziyoux-fusion-results-20261117-141823`.
Publishing the project page and uploading the full PDF to Gradescope are
separate submission steps.

## References
Mertens, T., Kautz, J., & Van Reeth, F. (2007). **Exposure Fusion**. In *Pacific Graphics 2007* (pp. 382–390). IEEE. https://doi.org/10.1109/PG.2007.17

[Paper PDF](https://www.tommertens.com/old-academic/papers/exposure_fusion_reduced.pdf)