#!/bin/bash
# Performance sweep: 6-run framework (run 6x, drop 1st, average 2-5).
# Mirrors CK's benchmark_fwd_sp3_compare.sh for apples-to-apples comparison.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(find_root "$SCRIPT_DIR")"
BUILD_DIR=""

SWEEP_SIZES=(
    "8 16 1024"
    "4 16 2048"
    "2 16 4096"
    "1  8 8192"
    "1 16 8192"
    "1  8 16384"
    "1  4 32768"
    "1  2 40000"
)

WARMUP=5
ITERS=20
NRUNS=6
DROP=1
MASK=0
SPLITK=""   # unset = single-pass (legacy behavior); set = pass --splitk $SPLITK

usage() {
    cat <<'USAGE'
Usage: run-benchmark.sh [OPTIONS]

Runs the full 8-config sweep with 6-run averaging (drop 1st, average 2-5).

Options:
  --build-dir DIR   Build directory  [default: <repo>/build]
  --mask N          Mask variant 0 (no mask) or 1 (causal)  [default: 0]
  --splitk G        Split-K split count; appends --splitk G to each run [unset]
  --help            Show this help
USAGE
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --mask)       MASK="$2"; shift 2 ;;
        --splitk)     SPLITK="$2"; shift 2 ;;
        --help)       usage ;;
        *)            echo "Unknown option: $1" >&2; usage ;;
    esac
done

resolve_build "${BUILD_DIR:-${REPO_ROOT}/build}"

# Optional --splitk passthrough: when unset the SPLITK_ARGS array is empty, so the
# bench invocation below is byte-for-byte identical to the legacy single-pass sweep.
SPLITK_ARGS=()
[ -n "$SPLITK" ] && SPLITK_ARGS=(--splitk "$SPLITK")

echo "Benchmark: ${#SWEEP_SIZES[@]} configs (WARMUP=$WARMUP ITERS=$ITERS RUNS=$NRUNS DROP=$DROP MASK=$MASK${SPLITK:+ SPLITK=$SPLITK})"
[ -n "${HIP_VISIBLE_DEVICES:-}" ] && echo "GPU: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo ""

printf "%4s %3s %6s %4s | %10s %10s %10s\n" \
    "B" "H" "S" "D" "Avg(ms)" "Min(ms)" "TFLOPS"
printf -- "---- --- ------ ---- + ---------- ---------- ----------\n"

for cfg in "${SWEEP_SIZES[@]}"; do
    read -r CB CH CS CD <<< "$cfg 64"

    avg_vals=()
    min_vals=()
    tf_vals=()
    err=""

    for run_idx in $(seq 1 $NRUNS); do
        set +e
        out=$("$BENCH" -b "$CB" -h "$CH" -s "$CS" -d "$CD" --mask "$MASK" \
              --warmup "$WARMUP" --iters "$ITERS" "${SPLITK_ARGS[@]}" 2>&1)
        rc=$?
        set -e

        if [ $rc -ne 0 ]; then
            err=$(echo "$out" | grep -iE "HIP error|out of memory|error" | head -1)
            [ -z "$err" ] && err="exited $rc (run $run_idx)"
            break
        fi

        if [ $run_idx -gt $DROP ]; then
            avg_vals+=($(echo "$out" | grep "^Avg:"    | awk '{print $2}'))
            min_vals+=($(echo "$out" | grep "^min="    | sed 's/min=//;s/ .*//'))
            tf_vals+=($( echo "$out" | grep "^TFLOPS:" | awk '{print $2}'))
        fi
    done

    if [ -z "$err" ] && [ ${#tf_vals[@]} -gt 0 ]; then
        avg=$(printf '%s\n' "${avg_vals[@]}" | awk '{s+=$1} END {printf "%.3f", s/NR}')
        min_ms=$(printf '%s\n' "${min_vals[@]}" | awk 'BEGIN{m=99999} {if($1+0<m)m=$1} END {printf "%.3f", m}')
        tf=$(printf '%s\n' "${tf_vals[@]}" | awk '{s+=$1} END {printf "%.2f", s/NR}')
    else
        avg="ERROR"; min_ms="ERROR"; tf="ERROR"
    fi

    printf "%4s %3s %6s %4s | %10s %10s %10s\n" \
        "$CB" "$CH" "$CS" "$CD" "$avg" "$min_ms" "$tf"

    if [ -n "$err" ]; then
        printf "  ↳ %s\n" "$err"
    fi
done

echo ""
