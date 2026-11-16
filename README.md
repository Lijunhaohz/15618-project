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

The CPU fusion pipeline and image I/O are implemented. The loader accepts
8-bit RGB PNG and binary P6 PPM with maximum value 255; other formats, PNG
alpha, and other bit depths are rejected. Saving supports the same formats,
clamping finite output values to `[0, 1]` and rounding to 8-bit. Encoded RGB
values are used directly without color or gamma conversion.

The command-line application uses the CPU backend by default. Select the CUDA
backend with `--backend cuda` after building with CUDA enabled; unavailable
CUDA devices or a CPU-only build produce an error rather than falling back.
The CUDA pipeline uses separate kernels and keeps its pyramid intermediates on
the device until the final floating-point image is returned.

Build and run the CPU-only project with CMake, C99 and C++17 compilers, and
the libpng development package (including its zlib dependency):

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/exposure_fusion --help
```

For example, fuse all three supplied exposures in a directory:

```sh
mkdir -p output
./build/exposure_fusion --output output/fused.png data/basic-test-3/
```

Individual input files may also be specified:

```sh
mkdir -p output
./build/exposure_fusion --output output/fused.png \
  data/basic-test-3/memorial00.png \
  data/basic-test-3/memorial03.png \
  data/basic-test-3/memorial06.png
```

Add `--levels N` to select a pyramid depth; otherwise the application uses up
to five levels, stopping when both dimensions reach one. Inputs must be aligned
and have the same dimensions. File and directory inputs can be mixed; each
directory is read non-recursively, in filename order. Regular files with `.md`,
`.txt`, `.json`, or `.csv` extensions and `.DS_Store` are ignored as metadata.
All other regular files in an input directory must be supported PNG or PPM
images; an unsupported file causes an error rather than being silently
omitted. Empty input directories also cause an error.

For example, after a CUDA-enabled build:

```sh
./build/exposure_fusion --backend cuda --output output/fused.png data/basic-test-3/
```

Alternatively, on Unix, build and run CUDA in one command from the project root:

```sh
./scripts/run_cuda.sh --output fused.png data/basic-test-3/
```

The script forwards image paths and options such as `--levels` to the CLI,
selects CUDA explicitly, and uses `build/cuda` by default. It checks `PATH`
and `/usr/local/cuda/bin/nvcc` for the CUDA compiler; `CUDACXX`,
`CUDAHOSTCXX`, and `FUSION_CUDA_BUILD_DIR` can override its compiler, host
compiler, and build directory. For CUDA 11.7 or 11.8 it selects `g++-11` when
available and no host compiler was specified. CMake caches compiler choices,
so use a fresh build directory when changing them. The output path must have an
existing parent directory; as with the CLI, an existing output is overwritten.

The output path must not refer to an input file, including an image discovered
in an input directory; an existing separate output file will be overwritten.
Keep output outside input directories. Use `--` before input paths beginning
with a hyphen.

On a machine with the CUDA toolkit, configure with
`cmake -S . -B build -DFUSION_ENABLE_CUDA=ON` to compile the CUDA backend.
CUDA is disabled by default. CUDA tests require an accessible GPU. CUDA
compilation disables fused multiply-add to better match the CPU reference's
floating-point evaluation. If `nvcc` is not on `PATH` or the default host
compiler is unsupported by the toolkit, specify `CUDACXX` and a compatible
`CMAKE_CUDA_HOST_COMPILER` when configuring.

The CUDA weight tests compare every raw and normalized map against the CPU.
Raw weights use an absolute tolerance of `1e-10` plus `3e-5` relative error;
normalized weights use `2e-5` absolute error. These preliminary tolerances allow
for float32 and CPU/GPU math-library differences; they are not final pipeline
correctness tolerances.
The CUDA pyramid tests compare every Gaussian weight, Gaussian image, and
Laplacian image level with the CPU using a `1e-6` absolute tolerance, allowing
for float32 rounding differences.
Blending and reconstruction tests use `1e-6` absolute error with identical
CPU-generated stage inputs, and `2e-5` for the chain of CUDA stages, where
earlier weight-computation differences also contribute.
The CUDA fusion integration test uses the same `2e-5` preliminary absolute
tolerance against CPU fusion and checks agreement with the individual CUDA
stages to `1e-6`.

### CUDA float correctness results

The following comparisons ran on an NVIDIA GeForce RTX 2080 with driver
550.67, CUDA toolkit 11.7.99, GCC 11.5, CMake 3.28.3, CUDA architecture 75,
and an unset `CMAKE_BUILD_TYPE` (no optimization flags). CUDA fused
multiply-add was disabled. Each test calls the CPU and CUDA `fuse()` methods
with the default depth and compares every unclipped float RGB component; RMSE
is over all components. Values below are from one run, not timing results.

| Input | Size | Exposures | Levels | Max absolute error | RMSE |
|---|---:|---:|---:|---:|---:|
| Constant gray bracket | 7 x 5 | 3 | 4 | 0 | 0 |
| Identical corner impulses | 7 x 5 | 2 | 4 | 0 | 0 |
| Odd-sized pattern | 7 x 5 | 3 | 4 | 5.96e-8 | 1.91e-8 |
| One column | 1 x 17 | 3 | 5 | 1.49e-7 | 4.38e-8 |
| One row | 17 x 1 | 3 | 5 | 5.96e-8 | 2.36e-8 |
| One pixel | 1 x 1 | 3 | 1 | 0 | 0 |
| Single exposure | 19 x 11 | 1 | 5 | 0 | 0 |
| Memorial bracket | 484 x 714 | 3 | 5 | 3.58e-7 | 1.74e-8 |

The impulse and single-exposure CUDA results also reproduce their inputs:
maximum errors were `2.91e-11` and `1.49e-8`, respectively. The Memorial test
loads `memorial00.png`, `memorial03.png`, and `memorial06.png` from
[`data/basic-test-3`](data/basic-test-3), with no preprocessing beyond the
loader's 8-bit-to-float conversion. The inputs are assumed aligned; alignment
has not been independently verified. See the [Memorial notes](data/memorial/README.md)
for provenance and usage cautions.

Tests require max absolute error <= `2e-5` and RMSE <= `2e-6`; the single-input
round-trip checks use `1e-6` for both. The CPU and GPU can differ in float32
arithmetic and math-library evaluation, and those differences propagate through
normalization and the pyramids. The limits leave roughly 50x headroom over the
largest observed max error and RMSE while the max limit remains below 1% of one
8-bit step (`1/255`). They are regression thresholds, not guarantees for
other devices or inputs. Reproduce the float comparisons with:

```sh
ctest --test-dir build -R '^cuda_fusion_validation_' -V
```

The real-image test is registered when all three bracket files are present.
The CUDA CLI integration test also loads that bracket and writes a PNG.

### Preliminary CUDA timings

On November 16, 2026, the separate-kernel baseline ran on the RTX 2080
(8 GiB), driver 550.67, CUDA 11.7.99, GCC 11.5, architecture 75, with
`CMAKE_BUILD_TYPE` unset and fused multiply-add disabled. The workload was the
three Memorial images above, 484 x 714 pixels, using five pyramid levels.
Alignment and usage cautions above also apply to this workload.

The benchmark loaded the PNGs and computed the CPU reference before timing.
After two warmups of each path, it ran ten trials, alternating an
event-instrumented fusion call with a normal, uninstrumented `fuse()` call.
The table gives the median and range across those trials, in milliseconds.

| Measurement | Median | Min | Max |
|---|---:|---:|---:|
| Quality weights, CUDA events | 0.111 | 0.109 | 0.115 |
| Normalization, CUDA events | 0.036 | 0.036 | 0.039 |
| Image and weight Gaussian pyramids, CUDA events | 1.476 | 1.455 | 1.554 |
| Laplacian pyramids, CUDA events | 1.189 | 1.177 | 1.222 |
| Blending, CUDA events | 0.271 | 0.267 | 0.277 |
| Reconstruction, CUDA events | 0.703 | 0.693 | 0.740 |
| Full device pipeline, CUDA events | 3.793 | 3.750 | 3.927 |
| Full uninstrumented `fuse()`, host wall clock | 39.996 | 39.749 | 41.058 |

Events bracket the pipeline after the input upload and before the output
download. Stage intervals can include device work for small error-flag
transfers and GPU idle gaps while the host synchronizes, allocates memory, or
submits more work; they are **not** pure kernel execution times. The wall-clock
interval includes
input validation, device allocations, host-to-device and device-to-host image
transfers, synchronization, and cleanup, but excludes PNG I/O and backend
construction. Independent stage medians need not sum to the pipeline median.
The last profiled and uninstrumented outputs each matched the CPU with max
error `3.58e-7` and RMSE `1.74e-8`. These are preliminary measurements on a
shared desktop GPU, not speedup claims.

With a CUDA-enabled build, reproduce them with:

```sh
./build/cuda_benchmark data/basic-test-3/memorial00.png \
  data/basic-test-3/memorial03.png data/basic-test-3/memorial06.png
```

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