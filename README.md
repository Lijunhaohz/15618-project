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

## Project Website

Edit [proposal.md](proposal.md), then regenerate [index.html](index.html):

```sh
python3 scripts/build_site.py
python3 -m http.server 8000
```

Open http://localhost:8000. Stop the server with Ctrl+C. No packages are required. Rebuild after every proposal edit and commit the generated HTML along with the Markdown. Styles live in `assets/style.css`.

The built-in Markdown renderer supports section headings (`##`), subheadings (`###` through `######`), horizontal rules, blockquotes, lists, tables, bold text, and HTTP(S) links. It is not a full Markdown implementation.

### Put the changes on GitHub

```sh
git status
git add proposal.md index.html assets/style.css scripts/build_site.py README.md .nojekyll
git commit -m "Add revised proposal and project website"
git push
```

### Publish the website

In the GitHub repository, go to **Settings → Pages**. Choose **Deploy from a branch**, select the branch you pushed and **/ (root)**, then save. Wait for GitHub's Pages deployment to finish.

Expected URL after publishing: https://lijunhaohz.github.io/15618-project/

The proposal uses this expected address; it is not verified live. If publishing under another owner or repository, update the proposal URL and the repository link in `scripts/build_site.py`, then rebuild.

## References
Mertens, T., Kautz, J., & Van Reeth, F. (2007). **Exposure Fusion**. In *Pacific Graphics 2007* (pp. 382–390). IEEE. https://doi.org/10.1109/PG.2007.17

[Paper PDF](https://www.tommertens.com/old-academic/papers/exposure_fusion_reduced.pdf)