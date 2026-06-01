#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// =============================================================================
// COMPONENT (TEST-ONLY): row-max reduction — pipeline STAGE 3 of 7.
//
// Standalone isolation of the softmax row-max step, used ONLY by
// tests/test_row_max.cpp. Golden-verified, NOT #included by src/fused/.
// CPU oracle: src/components_ref/ref_row_max.{hpp,cpp}.
//
// WHY ONLY ONE cross-lane shuffle: in the TransposedC layout each lane already
// holds its M-row's values for 32 of the 64 N-columns, and all 32 lanes in a
// k_sub half hold the SAME 32 columns. So an intra-lane max over the 32 regs
// already covers half the row; the two halves (lane and lane^32) hold the
// complementary columns. A SINGLE ds_bpermute(lane^32) brings the partner's
// half-max in, and one more fmaxf finishes the full 64-column row max. This
// matches CK's ISA (exactly 1 ds_bpermute for rowmax, 1 for rowsum).
// =============================================================================
//
// Phase 1 Kernel 3 -- test_row_max (cross-half ds_bpermute reduction)
//
// Reproduces CK's row-max reduction over S_acc, matching golden dump_reg
// slot 2 (RMAX). Input is the RAW (unscaled) S_acc in thread-buffer order
// (256*32 fp32), as produced by Kernel 2 (test_qk_gemm).
//
// S_acc distribution (golden-verified):
//   m_row = (lane%32) + 32*warp     -- each lane has ONE M-row
//   n_col = (r/8)*16 + (lane/32)*8 + (r%8)
//   All 32 lanes within a k_sub half hold the SAME 32 N-columns.
//   k_sub=0: N-cols {0-7, 16-23, 32-39, 48-55}
//   k_sub=1: N-cols {8-15, 24-31, 40-47, 56-63}  (complementary)
//
// Algorithm:
//   1. Intra-lane: max over 32 registers = max over 32 N-columns for
//      this thread's single M-row. No cross-lane butterfly needed since
//      lanes in the same k_sub hold identical N-column sets.
//   2. Cross-half: 1 ds_bpermute with lane^32 (k_sub partner) merges
//      the complementary 32 N-columns → full 64-column row max.
//
// This matches CK's ISA: exactly 1 ds_bpermute for rowmax (and 1 for rowsum).
//
// Output: 1 fp32 per thread (256 total). Both k_sub halves get the same value.
// Golden RMAX is RAW max (no scale_s applied).

typedef float v16f __attribute__((ext_vector_type(16)));

// ds_bpermute: read src_lane's copy of val.
__device__ __forceinline__ float rm_bpermute(int src_lane, float val) {
    int src_byte = src_lane * 4;
    int ret;
    asm volatile("ds_bpermute_b32 %0, %1, %2\n"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(ret) : "v"(src_byte), "v"(val) : "memory");
    return __builtin_bit_cast(float, ret);
}

// S_acc in thread-buffer order: s_acc[tid*32+r], 256 threads x 32 regs.
// Output: rmax[tid], 256 floats.
__global__ void test_row_max_kernel(const float* __restrict__ s_acc,
                                    float* __restrict__ rmax) {
    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;

    // Load this thread's 32 S_acc values (16 for n0-tile, 16 for n1-tile)
    float local_max = -INFINITY;
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        float v = s_acc[tid * 32 + r];
        local_max = fmaxf(local_max, v);
    }

    // Cross-half exchange: pair with lane in the other k_sub half.
    int partner = lane_id ^ 32;          // same warp, other k_sub
    int partner_tid = (tid & ~63) | partner;  // full thread index of partner
    float other = rm_bpermute(partner_tid, local_max);
    local_max = fmaxf(local_max, other);

    rmax[tid] = local_max;
}
