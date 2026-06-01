#!/bin/bash
# Phase 3.4 gate runner: G1 (build), G2 (tests w/ golden), G3 (extract .s).
# Run INSIDE the build container (poyenc-fmha), which mounts the repo at
# /root/workspace. Example:
#   docker exec poyenc-fmha bash -c "/root/workspace/scripts/run-gates.sh"
#
# G2 bar: 67/67 fused (build/test_fmha_fwd_d64) + 49/49 standalone
# (7 suites x 7 tests) using the golden dumps. Golden dirs default to
# /tmp/fmha-native-isa-match/golden/{full,partial} (already staged in
# poyenc-fmha). Without golden, the MatchesGolden cases GTEST_SKIP and the
# bar is NOT met -- the script fails if the golden dirs are missing.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
REPO_ROOT="$(find_root "$SCRIPT_DIR")"

BUILD_DIR="${REPO_ROOT}/build"
GOLDEN_ROOT="/tmp/fmha-native-isa-match/golden"
STAGES="g1,g2,g3"

STANDALONE=(test_k_lds test_qk_gemm test_row_max test_softmax test_v_lds test_pv_gemm test_epilog)
KERNEL_S_SRC="kernel-hip-amdgcn-amd-amdhsa-gfx942.s"
KERNEL_S_DST="${REPO_ROOT}/native_d64_kernel.s"

usage() {
    cat <<'USAGE'
Usage: run-gates.sh [OPTIONS]

Runs Phase 3.4 verification gates inside the build container.

Options:
  --stages LIST     Comma-separated subset of g1,g2,g3  [default: g1,g2,g3]
  --build-dir DIR   Build directory                     [default: <repo>/build]
  --golden DIR      Golden root holding full/ + partial/ [default: /tmp/fmha-native-isa-match/golden]
  --help            Show this help

Gates:
  g1  Build the fmha_kernel target.
  g2  Run fused (67/67) + 7 standalone suites with golden (49/49). Fails on any
      failure, skip, or missing golden dir.
  g3  Rebuild and copy the kernel's generated .s (kernel.cpp -> kernel-hip-...gfx942.s)
      to <repo>/native_d64_kernel.s for assembly comparison.
USAGE
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stages)    STAGES="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --golden)    GOLDEN_ROOT="$2"; shift 2 ;;
        --help)      usage ;;
        *)           echo "Unknown option: $1" >&2; usage ;;
    esac
done

GOLDEN_FULL="${GOLDEN_ROOT}/full"
GOLDEN_PARTIAL="${GOLDEN_ROOT}/partial"

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }
fail() { echo "GATE FAIL: $*" >&2; exit 1; }

# ---- G1: build ----
# Build the kernel AND every binary G2 runs, so the gate never executes a stale
# test binary. (cmake links libfmha_kernel.a into the test/standalone binaries,
# but does NOT relink them unless they are named as targets — the stale-binary
# trap. Listing them here forces a relink whenever the kernel source changes.)
gate_g1() {
    echo "=== G1: build fmha_kernel + test binaries ==="
    cmake --build "$BUILD_DIR" \
        --target fmha_kernel test_fmha_fwd_d64 "${STANDALONE[@]}" \
        -j"$(nproc)" || fail "G1 build failed"
    echo "G1 OK"
}

# ---- G2: tests with golden ----
# Counts an explicit "[  PASSED  ] N tests." with NO "[  FAILED  ]" and NO
# "[  SKIPPED  ]" line as a pass. Any skip means golden was not applied.
run_gtest() {
    local label="$1"; shift
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ $rc -ne 0 ]; then
        echo "$out" | tail -20
        fail "G2 ${label}: exit ${rc}"
    fi
    if echo "$out" | grep -q '\[  FAILED  \]'; then
        echo "$out" | grep -E '\[  FAILED  \]'
        fail "G2 ${label}: failing cases"
    fi
    if echo "$out" | grep -q '\[  SKIPPED \]'; then
        echo "$out" | grep -E '\[  SKIPPED \]' | head
        fail "G2 ${label}: skipped cases (golden not applied)"
    fi
    local summary
    summary="$(echo "$out" | grep -E '\[  PASSED  \] [0-9]+ tests' | tail -1)"
    [ -n "$summary" ] || fail "G2 ${label}: no PASSED summary"
    printf "  %-22s %s\n" "$label" "$summary"
}

gate_g2() {
    echo "=== G2: tests (fused 67/67 + standalone 49/49 via golden) ==="
    [ -f "${GOLDEN_FULL}/dump_lds.bin" ]    || fail "G2 missing golden full: ${GOLDEN_FULL}"
    [ -f "${GOLDEN_PARTIAL}/dump_lds.bin" ] || fail "G2 missing golden partial: ${GOLDEN_PARTIAL}"

    run_gtest "fused" "${BUILD_DIR}/test_fmha_fwd_d64"
    for t in "${STANDALONE[@]}"; do
        run_gtest "$t" "${BUILD_DIR}/${t}" \
            "--golden-full=${GOLDEN_FULL}" "--golden-partial=${GOLDEN_PARTIAL}"
    done
    echo "G2 OK"
}

# ---- G3: extract assembly ----
gate_g3() {
    echo "=== G3: extract kernel assembly ==="
    cmake --build "$BUILD_DIR" --target fmha_kernel -j"$(nproc)" || fail "G3 build failed"
    [ -f "${BUILD_DIR}/${KERNEL_S_SRC}" ] || fail "G3 missing ${BUILD_DIR}/${KERNEL_S_SRC}"
    cp "${BUILD_DIR}/${KERNEL_S_SRC}" "${KERNEL_S_DST}" || fail "G3 copy failed"
    echo "G3 OK -> ${KERNEL_S_DST}"
}

has_stage g1 && gate_g1
has_stage g2 && gate_g2
has_stage g3 && gate_g3
echo "ALL REQUESTED GATES PASSED (${STAGES})"
