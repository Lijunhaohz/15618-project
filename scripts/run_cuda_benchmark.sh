#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${FUSION_CUDA_BUILD_DIR:-"$project_dir/build/cuda"}

if [ "$#" -eq 0 ]; then
    set -- "$project_dir/data/basic-test-3"
fi

cmake --build "$build_dir"
ctest --test-dir "$build_dir" --output-on-failure
exec "$build_dir/cuda_benchmark" "$@"
