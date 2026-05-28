#!/bin/bash
# Performance sweep: 6-run framework (run 6x, drop 1st, average 2-5).
# Always runs the full config sweep — no single-config mode.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(find_root "$SCRIPT_DIR")"
BUILD_DIR=""

# Defaults
WARMUP=5
ITERS=20
SWEEP_SIZES=(
    "b=8,h=16,s=1024"
    "b=4,h=16,s=2048"
    "b=2,h=16,s=4096"
    "b=1,h=8,s=8192"
    "b=1,h=16,s=8192"
    "b=1,h=8,s=16384"
    "b=1,h=4,s=32768"
    "b=1,h=2,s=40000"
)
SWEEP_SIZES_VL=(
    "h=16,s=1024,varlen_seqs=1024:1024:1024:1024:1024:1024:1024:1024"
    "h=16,s=2048,varlen_seqs=2048:2048:2048:2048"
    "h=16,s=4096,varlen_seqs=4096:4096"
    "h=8,s=8192,varlen_seqs=8192"
    "h=16,s=8192,varlen_seqs=8192"
    "h=8,s=16384,varlen_seqs=16384"
    "h=4,s=32768,varlen_seqs=32768"
    "h=2,s=40000,varlen_seqs=40000"
)
VARLEN=0

usage() {
    cat <<'USAGE'
Usage: run-benchmark.sh [OPTIONS]

Runs the full 8-config sweep with 6-run averaging (drop 1st, average 2-5).

Options:
  --varlen          Run variable-length configs instead of fixed-length

Paths:
  --build-dir DIR   Build directory                             [default: auto-detect]

Other:
  --help            Show this help
USAGE
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --varlen)     VARLEN=1; shift ;;
        --build-dir)  BUILD_DIR="$2"; shift 2 ;;
        --help)       usage ;;
        *)            echo "Unknown option: $1" >&2; usage ;;
    esac
done

resolve_build "${BUILD_DIR:-${REPO_ROOT}/build}"

# Build work list
WORK=()
if [ "$VARLEN" -eq 1 ]; then
    SIZES=("${SWEEP_SIZES_VL[@]}")
else
    SIZES=("${SWEEP_SIZES[@]}")
fi
for sz in "${SIZES[@]}"; do
    WORK+=("${sz},d=64")
done

NRUNS=6
DROP=1

echo "Benchmark: ${#WORK[@]} configs (WARMUP=$WARMUP ITERS=$ITERS RUNS=$NRUNS DROP=$DROP)"
echo ""

printf "%4s %3s %6s %4s | %10s %10s %10s\n" \
    "B" "H" "S" "D" "Avg(ms)" "Min(ms)" "TFLOPS"
printf -- "---- --- ------ ---- + ---------- ---------- ----------\n"

for cfg in "${WORK[@]}"; do
    # Parse key=value pairs
    declare -A P=()
    IFS=',' read -ra PAIRS <<< "$cfg"
    for pair in "${PAIRS[@]}"; do
        key="${pair%%=*}"; val="${pair#*=}"
        P[$key]="$val"
    done

    CB=${P[b]:-2}; CH=${P[h]:-16}; CS=${P[s]:-4096}; CD=${P[d]:-64}

    # Build bench args
    BENCH_ARGS=("-b" "$CB" "-h" "$CH" "-s" "$CS" "-d" "$CD")
    [ -n "${P[kv_heads]:-}" ] && BENCH_ARGS+=("--kv-heads" "${P[kv_heads]}")
    [ -n "${P[mask]:-}" ]     && BENCH_ARGS+=("--mask" "${P[mask]}")
    [ -n "${P[lse]:-}" ]      && BENCH_ARGS+=("--lse" "${P[lse]}")
    if [ -n "${P[varlen_seqs]:-}" ]; then
        vseqs="${P[varlen_seqs]}"
        BENCH_ARGS+=("--varlen-seqs" "${vseqs//:/,}")
    fi
    BENCH_ARGS+=("--warmup" "$WARMUP" "--iters" "$ITERS")

    run_avg="ERROR"; run_min="ERROR"; run_tf="ERROR"
    err_reason=""
    avg_vals=()
    min_vals=()
    tf_vals=()

    for run_idx in $(seq 1 $NRUNS); do
        set +e
        out=$("$BENCH" "${BENCH_ARGS[@]}" 2>&1)
        rc=$?
        set -e

        if [ $rc -ne 0 ]; then
            err_reason=$(echo "$out" | grep -iE "HIP error|out of memory|error" | head -1)
            [ -z "$err_reason" ] && err_reason="bench exited $rc (run $run_idx)"
            break
        fi

        if [ $run_idx -gt $DROP ]; then
            avg_vals+=($(echo "$out" | grep "^Avg:"    | awk '{print $2}'))
            min_vals+=($(echo "$out" | grep "^min="    | sed 's/min=//;s/ .*//'))
            tf_vals+=($( echo "$out" | grep "^TFLOPS:" | awk '{print $2}'))
        fi
    done

    if [ -z "$err_reason" ] && [ ${#tf_vals[@]} -gt 0 ]; then
        run_avg=$(printf '%s\n' "${avg_vals[@]}" | awk '{s+=$1} END {printf "%.3f", s/NR}')
        run_min=$(printf '%s\n' "${min_vals[@]}" | awk 'BEGIN{m=99999} {if($1+0<m)m=$1} END {printf "%.3f", m}')
        run_tf=$( printf '%s\n' "${tf_vals[@]}"  | awk '{s+=$1} END {printf "%.2f", s/NR}')
    fi

    printf "%4s %3s %6s %4s | %10s %10s %10s\n" \
        "$CB" "$CH" "$CS" "$CD" "$run_avg" "$run_min" "$run_tf"

    if [ -n "$err_reason" ]; then
        printf "  ↳ %s\n" "$err_reason"
    fi

    unset P
done

echo ""
