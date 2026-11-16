#!/bin/sh
set -eu

if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; }; then
    echo "Usage: $0 --output FILE [--levels N] [--] INPUT..."
    echo "Configure and build the CPU backend, then fuse the specified images."
    if [ "$#" -eq 0 ]; then exit 1; fi
    exit 0
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${FUSION_CPU_BUILD_DIR:-"$project_dir/build/cpu"}

cmake -S "$project_dir" -B "$build_dir" -DFUSION_ENABLE_CUDA=OFF
cmake --build "$build_dir" --target exposure_fusion
exec "$build_dir/exposure_fusion" --backend cpu "$@"
