#!/bin/sh
set -eu

runner=$1
backend=$2
temp_dir=$(mktemp -d)
build_dir="$temp_dir/build directory"
mkdir "$build_dir"
trap 'rm -f "$temp_dir/cmake" "$build_dir/exposure_fusion" "$temp_dir/calls" "$temp_dir/cmake_args" "$temp_dir/args" "$temp_dir/expected" "$temp_dir/error"; rmdir "$build_dir" "$temp_dir"' EXIT

cat > "$temp_dir/cmake" <<'EOF'
#!/bin/sh
printf '%s\n' "$1" >> "$CALLS_FILE"
printf '<%s>\n' "$@" >> "$CMAKE_ARGS_FILE"
if { [ "${FAIL_STAGE:-}" = configure ] && [ "$1" = -S ]; } ||
   { [ "${FAIL_STAGE:-}" = build ] && [ "$1" = --build ]; }; then
    echo "Simulated CMake failure" >&2
    exit 42
fi
EOF
cat > "$build_dir/exposure_fusion" <<'EOF'
#!/bin/sh
printf '<%s>\n' "$@" > "$ARGS_FILE"
EOF
chmod +x "$temp_dir/cmake" "$build_dir/exposure_fusion"

export PATH="$temp_dir:$PATH"
export CUDACXX=/bin/true CUDAHOSTCXX=/bin/true
export FUSION_CUDA_BUILD_DIR="$build_dir" FUSION_CPU_BUILD_DIR="$build_dir"
export CALLS_FILE="$temp_dir/calls" CMAKE_ARGS_FILE="$temp_dir/cmake_args" ARGS_FILE="$temp_dir/args"

sh "$runner" --output "output file.png" --levels 2 -- "-input file.png"
printf '%s\n' '<--backend>' "<$backend>" '<--output>' '<output file.png>' \
    '<--levels>' '<2>' '<-->' '<-input file.png>' > "$temp_dir/expected"
cmp "$temp_dir/expected" "$temp_dir/args"
printf '%s\n' '-S' '--build' > "$temp_dir/expected"
cmp "$temp_dir/expected" "$temp_dir/calls"
if [ "$backend" = cpu ]; then
    expected_cuda=OFF
else
    expected_cuda=ON
fi
grep -q -F "<-DFUSION_ENABLE_CUDA=$expected_cuda>" "$temp_dir/cmake_args"

for stage in configure build; do
    rm -f "$temp_dir/args"
    if FAIL_STAGE=$stage sh "$runner" --output out.png input.png 2>"$temp_dir/error"; then
        echo "Runner succeeded despite $stage failure" >&2
        exit 1
    fi
    if [ -e "$temp_dir/args" ] || ! grep -q "Simulated CMake failure" "$temp_dir/error"; then
        echo "Runner did not propagate $stage failure" >&2
        exit 1
    fi
done
