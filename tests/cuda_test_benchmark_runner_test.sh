#!/bin/sh
set -eu

runner=$1
backend=$2
project_dir=$(CDPATH= cd -- "$(dirname -- "$runner")/.." && pwd)
temp_dir=$(mktemp -d)
build_dir="$temp_dir/build directory"
benchmark="$build_dir/${backend}_benchmark"
mkdir "$build_dir"
trap 'rm -f "$temp_dir/cmake" "$temp_dir/ctest" "$benchmark" "$temp_dir/calls" "$temp_dir/expected" "$temp_dir/error"; rmdir "$build_dir" "$temp_dir"' EXIT

cat > "$temp_dir/cmake" <<'EOF'
#!/bin/sh
printf '<cmake>' >> "$CALLS_FILE"
printf ' <%s>' "$@" >> "$CALLS_FILE"
printf '\n' >> "$CALLS_FILE"
if [ "${FAIL_STAGE:-}" = build ]; then
    echo "Simulated build failure" >&2
    exit 42
fi
EOF
cat > "$temp_dir/ctest" <<'EOF'
#!/bin/sh
printf '<ctest>' >> "$CALLS_FILE"
printf ' <%s>' "$@" >> "$CALLS_FILE"
printf '\n' >> "$CALLS_FILE"
if [ "${FAIL_STAGE:-}" = test ]; then
    echo "Simulated test failure" >&2
    exit 42
fi
EOF
cat > "$benchmark" <<'EOF'
#!/bin/sh
printf '<benchmark>' >> "$CALLS_FILE"
printf ' <%s>' "$@" >> "$CALLS_FILE"
printf '\n' >> "$CALLS_FILE"
if [ "${FAIL_STAGE:-}" = benchmark ]; then
    echo "Simulated benchmark failure" >&2
    exit 42
fi
EOF
chmod +x "$temp_dir/cmake" "$temp_dir/ctest" "$benchmark"

export PATH="$temp_dir:$PATH"
export FUSION_CUDA_BUILD_DIR="$build_dir" FUSION_CPU_BUILD_DIR="$build_dir" CALLS_FILE="$temp_dir/calls"

sh "$runner" "input directory" "another image.png"
printf '%s\n' "<cmake> <--build> <$build_dir>" \
    "<ctest> <--test-dir> <$build_dir> <--output-on-failure>" \
    '<benchmark> <input directory> <another image.png>' > "$temp_dir/expected"
cmp "$temp_dir/expected" "$temp_dir/calls"

: > "$temp_dir/calls"
sh "$runner"
printf '%s\n' "<cmake> <--build> <$build_dir>" \
    "<ctest> <--test-dir> <$build_dir> <--output-on-failure>" \
    "<benchmark> <$project_dir/data/basic-test-3>" > "$temp_dir/expected"
cmp "$temp_dir/expected" "$temp_dir/calls"

for stage in build test benchmark; do
    : > "$temp_dir/calls"
    if FAIL_STAGE=$stage sh "$runner" input 2>"$temp_dir/error"; then
        echo "Runner succeeded despite $stage failure" >&2
        exit 1
    fi
    if ! grep -q "Simulated $stage failure" "$temp_dir/error"; then
        echo "Runner did not propagate $stage failure" >&2
        exit 1
    fi
    if { [ "$stage" = build ] && grep -q '<ctest>' "$temp_dir/calls"; } ||
       { [ "$stage" = test ] && grep -q '<benchmark>' "$temp_dir/calls"; }; then
        echo "Runner continued after $stage failure" >&2
        exit 1
    fi
done
