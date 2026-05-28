#!/bin/bash
# Shared utilities for run-benchmark.sh

find_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        [ -f "$dir/CMakeLists.txt" ] && echo "$dir" && return
        dir="$(dirname "$dir")"
    done
    echo "Error: CMakeLists.txt not found (are you inside the fmha_native repo?)" >&2
    exit 1
}

resolve_build() {
    local build_dir="$1"

    BENCH="${build_dir}/bench_fmha_fwd"
    if [ ! -f "$BENCH" ]; then
        echo "Error: bench_fmha_fwd not found at ${BENCH}." >&2
        exit 1
    fi
}
