#!/bin/sh
set -eu

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; }; then
    echo "Usage: $0 --output FILE [--levels N] [--] INPUT..."
    echo "Configure and build the CUDA backend, then fuse the specified images."
    if [ "$#" -eq 0 ]; then exit 1; fi
    exit 0
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${FUSION_CUDA_BUILD_DIR:-"$project_dir/build/cuda"}

if [ -z "${CUDACXX:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDACXX=$(command -v nvcc)
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDACXX=/usr/local/cuda/bin/nvcc
    else
        echo "CUDA compiler not found; set CUDACXX or add nvcc to PATH." >&2
        exit 1
    fi
    export CUDACXX
fi

# CUDA 11.7 and 11.8 support GCC 11 as a host compiler.
if [ -z "${CUDAHOSTCXX:-}" ] && [ -z "${CXX:-}" ] &&
   [ -x "$CUDACXX" ] && command -v g++-11 >/dev/null 2>&1; then
    case "$("$CUDACXX" --version)" in
        *"release 11.7,"*|*"release 11.8,"*)
            CUDAHOSTCXX=$(command -v g++-11)
            export CUDAHOSTCXX
            ;;
    esac
fi

cmake -S "$project_dir" -B "$build_dir" -DFUSION_ENABLE_CUDA=ON
cmake --build "$build_dir" --target exposure_fusion
exec "$build_dir/exposure_fusion" --backend cuda "$@"
