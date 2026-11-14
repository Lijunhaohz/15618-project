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

The command-line application currently uses the CPU backend. CUDA is not yet
implemented; calling a CUDA placeholder reports an error.

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

The output path must not refer to an input file, including an image discovered
in an input directory; an existing separate output file will be overwritten.
Keep output outside input directories. Use `--` before input paths beginning
with a hyphen.

On a machine with the CUDA toolkit, configure with
`cmake -S . -B build -DFUSION_ENABLE_CUDA=ON` to compile the CUDA skeleton.
The CUDA stages are placeholders as well; CUDA is disabled by default.

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