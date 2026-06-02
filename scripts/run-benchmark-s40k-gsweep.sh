#!/bin/bash
# Split-K G-sweep for the single customer config B1 H2 S40000 d64.
# Runs the NATIVE bench across several --splitk G values (G=0 = single-pass
# baseline) using the same 6-run framework as run-benchmark.sh: run NRUNS times,
# drop the first, average the rest. In-container (poyenc-fmha), GPU2.
#
# CK is benchmarked SEPARATELY (its own container/script); compare the printed
# native TFLOPS against CK's number for the same config by hand / by the caller.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(find_root "$SCRIPT_DIR")"
BUILD_DIR=""

# Fixed customer config.
B=1; H=2; S=40000; D=64
MASK=0

# G values to sweep. 0 = single-pass (no --splitk), the baseline.
GVALS=(0 1 4 8 16)

WARMUP=5
ITERS=20
NRUNS=6
DROP=1

usage() {
    cat <<'USAGE'
Usage: run-benchmark-s40k-gsweep.sh [OPTIONS]

Sweeps split-K G for the fixed config B1 H2 S40000 d64 (6-run avg, drop 1st).
G=0 is the single-pass baseline (no --splitk).

Options:
  --build-dir DIR   Build directory          [default: <repo>/build]
  --mask N          Mask 0 (none) or 1 (causal) [default: 0]
  --gvals "A B C"   Space-separated G values   [default: 0 1 4 8 16]
  --help            Show this help
USAGE
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --mask)      MASK="$2"; shift 2 ;;
        --gvals)     read -r -a GVALS <<< "$2"; shift 2 ;;
        --help)      usage ;;
        *)           echo "Unknown option: $1" >&2; usage ;;
    esac
done

resolve_build "${BUILD_DIR:-${REPO_ROOT}/build}"

echo "Split-K G-sweep: B$B H$H S$S D$D mask=$MASK (WARMUP=$WARMUP ITERS=$ITERS RUNS=$NRUNS DROP=$DROP)"
[ -n "${HIP_VISIBLE_DEVICES:-}" ] && echo "GPU: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
echo ""

printf "%-12s | %10s %10s %10s | %8s\n" "config" "Avg(ms)" "Min(ms)" "TFLOPS" "vs G=0"
printf -- "------------ + ---------- ---------- ---------- + --------\n"

base_tf=""   # baseline (G=0) TFLOPS, for the vs-baseline column

for G in "${GVALS[@]}"; do
    avg_vals=(); min_vals=(); tf_vals=(); err=""

    # Single-pass (G=0) drops the --splitk flag entirely so it exercises the
    # exact legacy path; G>0 adds --splitk G (two-pass split+combine, both timed).
    splitk_args=()
    label="single-pass"
    if [ "$G" -gt 0 ]; then splitk_args=(--splitk "$G"); label="splitk G=$G"; fi

    for run_idx in $(seq 1 $NRUNS); do
        set +e
        out=$("$BENCH" -b "$B" -h "$H" -s "$S" -d "$D" --mask "$MASK" \
              "${splitk_args[@]}" --warmup "$WARMUP" --iters "$ITERS" 2>&1)
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
        [ "$G" -eq 0 ] && base_tf="$tf"
        if [ -n "$base_tf" ]; then
            vs=$(awk -v t="$tf" -v b="$base_tf" 'BEGIN{printf "%+.1f%%", (t/b-1)*100}')
        else
            vs="-"
        fi
    else
        avg="ERROR"; min_ms="ERROR"; tf="ERROR"; vs="-"
    fi

    printf "%-12s | %10s %10s %10s | %8s\n" "$label" "$avg" "$min_ms" "$tf" "$vs"
    [ -n "$err" ] && printf "  ↳ %s\n" "$err"
done

echo ""
echo "Note: CK is benchmarked separately (poyenc-ck). Compare the TFLOPS column"
echo "against CK's number for B$B H$H S$S d$D mask=$MASK."
