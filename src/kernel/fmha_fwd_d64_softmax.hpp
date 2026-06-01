#pragma once
#include "fmha_fwd_d64_gemm.hpp"

// ================================================================
// Phase 2: Softmax functions (from Phase 1 K3/K4, golden-verified)
// ================================================================
//
// S_acc distribution (TransposedC, groups-of-8 via SwizzleA):
//   m_row = (lane%32) + 32*warp  — each lane owns ONE M-row
//   n_col(i, k_sub, n_tile) = n_tile*32 + (i/8)*16 + k_sub*8 + (i%8)
//   where i=0..15 (register index), k_sub = lane/32, n_tile=0 or 1
//
// Reduction: 32 lanes in the same k_sub half hold the SAME 32 N-columns.
// k_sub=0: N-cols {0-7, 16-23, 32-39, 48-55}
// k_sub=1: N-cols {8-15, 24-31, 40-47, 56-63}  (complementary)
// So reduction is:
//   1. Intra-lane: reduce over 32 registers (covers one half of N-columns)
//   2. Cross-half: 1 ds_bpermute with lane^32 merges the complementary half
// NO butterfly needed.

// ---- Mask ----
//
// Applies masking to S_acc in-place (scale deferred to exp).
// Boundary mask: if n_col >= seqlen_k (absolute): -INFINITY
// Causal mask: if n_col > m_row + shift: -INFINITY
//
// n_col = kv_offset + (i/8)*16 + k_sub*8 + (i%8)      [for n0 tile]
//       = kv_offset + 32 + (i/8)*16 + k_sub*8 + (i%8)  [for n1 tile]

template <bool HasMask>
__device__ __forceinline__ void softmax_mask(
    v16f& s_acc_n0, v16f& s_acc_n1,
    int seqlen_k,
    int kv_offset,
    int m_row,         // this thread's M-row index
    int mask_shift,    // seqlen_k - seqlen_q (for causal)
    int m_tile_base)   // m_tile_idx * kM0 (wave-uniform min row of this M-tile)
{
    const int k_sub = (threadIdx.x & 63) >> 5;

    const int col_base = kv_offset + k_sub * 8;
    int limit = seqlen_k - col_base;
    if constexpr (HasMask) {
        int causal = m_row + mask_shift - col_base + 1;
        limit = (causal < limit) ? causal : limit;
    }

    // Full-tile fast path: when the whole 64-column tile is in-bounds, no element
    // is masked. Skipping it removes the per-iteration 32 v_cmp + 30 s_or + 32
    // v_cndmask the compiler emits otherwise. The guard is wave-uniform so this is
    // a single scalar branch. The boundary mask only does work on the last tile
    // (and, for causal, the diagonal tiles), which still take the slow path.
    if constexpr (!HasMask) {
        // Max absolute column in this tile = kv_offset + 63 (k_sub=1, n1, off=23).
        if (kv_offset + kN0 <= seqlen_k)
            return;
    } else {
        // Causal: a tile fully BELOW the diagonal needs no masking. The tightest
        // causal limit is at the topmost row (m_tile_base). If the whole tile is
        // left of that row's diagonal AND in-bounds, every element is valid.
        // Only the diagonal/last edge tile takes the slow path (CK's IsEdgeTile).
        if (kv_offset + kN0 <= m_tile_base + mask_shift + 1 &&
            kv_offset + kN0 <= seqlen_k)
            return;
    }

    // Combined mask: 1 compare per element (no scale — deferred to exp)
    constexpr int offsets[16] = {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int off = offsets[i];
        if (off >= limit)
            s_acc_n0[i] = -INFINITY;
        if (off + 32 >= limit)
            s_acc_n1[i] = -INFINITY;
    }
}

// ---- Row max: intra-lane max + 1 ds_bpermute cross-half ----
//
// Returns 1 fp32 scalar (the row max across all 64 N-columns).
// Both k_sub halves get the same result.
// From Phase 1 K3.

__device__ __forceinline__ float softmax_row_max(
    const v16f& s_acc_n0,
    const v16f& s_acc_n1,
    float rmax = -INFINITY)
{
    // Intra-lane max over 32 registers + previous rmax (avoids -INF when all masked)
    float local_max = rmax;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        local_max = fmaxf(local_max, s_acc_n0[i]);
        local_max = fmaxf(local_max, s_acc_n1[i]);
    }

    // Cross-half exchange: 1 ds_bpermute with lane^32
    int partner = (threadIdx.x & ~63) | ((threadIdx.x & 63) ^ 32);
    float other = bpermute_f32(partner, local_max);
    return fmaxf(local_max, other);
}

// ---- Exp2: P = exp2(scale * S - scale * m_new) ----
//
// S_acc is unscaled (raw GEMM output) and masked (-INF for OOB).
// Computes exp2(fmaf(scale, S, -scale_m)) in-place.
// Compiler maps fmaf → v_fma_f32 (single 1-cycle VALU, guaranteed fused).
// Masked values: fmaf(scale, -INF, -scale_m) = -INF → exp2(-INF) = 0.

__device__ __forceinline__ void softmax_exp2(
    v16f& s_acc_n0, v16f& s_acc_n1,
    float scale, float scale_m)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        s_acc_n0[i] = __builtin_amdgcn_exp2f(fmaf(scale, s_acc_n0[i], -scale_m));
        s_acc_n1[i] = __builtin_amdgcn_exp2f(fmaf(scale, s_acc_n1[i], -scale_m));
    }
}

// ---- Row sum: intra-lane sum + 1 ds_bpermute cross-half ----
//
// Returns 1 fp32 scalar (the row sum of P across all 64 N-columns).
// Both k_sub halves get the same result.
// From Phase 1 K4.

__device__ __forceinline__ float softmax_row_sum(
    const v16f& p_n0,
    const v16f& p_n1)
{
    // Intra-lane sum over 32 P values
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        local_sum += p_n0[i];
        local_sum += p_n1[i];
    }

    // Cross-half exchange: 1 ds_bpermute with lane^32
    int partner = (threadIdx.x & ~63) | ((threadIdx.x & 63) ^ 32);
    float other = bpermute_f32(partner, local_sum);
    return local_sum + other;
}

// ---- Rescale O_acc when max changes between tiles ----
//
// o_acc *= exp2(old_max - new_max) per element.

__device__ __forceinline__ void rescale_o_acc(
    v16f& o_acc_d0, v16f& o_acc_d1,
    float factor)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o_acc_d0[i] *= factor;
        o_acc_d1[i] *= factor;
    }
}

__device__ __forceinline__ void rescale_o_acc(
    v16f& o_acc_d0, v16f& o_acc_d1,
    float old_max, float new_max)
{
    rescale_o_acc(o_acc_d0, o_acc_d1, __builtin_amdgcn_exp2f(old_max - new_max));
}

// ================================================================
// Legacy functions — used by current _device.hpp until Task 2.6.
// DO NOT use in new Phase 2 code. Will be removed after 2.6.
// ================================================================

// Old softmax_row_max with array output (used by current device.hpp)
__device__ __forceinline__ void softmax_row_max(float (&row_max)[16],
                                       const v16f& s_acc_n0,
                                       const v16f& s_acc_n1,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int base_lane = k_sub * 32;

    for (int i = 0; i < 16; i++) {
        float local_max = fmaxf(s_acc_n0[i], s_acc_n1[i]);
        for (int offset = 16; offset >= 1; offset >>= 1) {
            int src = base_lane | ((lane_id & 31) ^ offset);
            float other = bpermute_f32(src, local_max);
            local_max = fmaxf(local_max, other);
        }
        row_max[i] = local_max;
    }
}

// Old softmax_exp with array input
__device__ __forceinline__ void softmax_exp(v16f& s_acc_n0,
                                   v16f& s_acc_n1,
                                   const float (&row_max)[16]) {
    for (int i = 0; i < 16; i++) {
        if (row_max[i] == -INFINITY) {
            s_acc_n0[i] = 0.0f;
            s_acc_n1[i] = 0.0f;
        } else {
            s_acc_n0[i] = __builtin_amdgcn_exp2f(s_acc_n0[i] - row_max[i]);
            s_acc_n1[i] = __builtin_amdgcn_exp2f(s_acc_n1[i] - row_max[i]);
        }
    }
}

// Old softmax_row_sum with array output
__device__ __forceinline__ void softmax_row_sum(float (&row_sum)[16],
                                       const v16f& s_acc_n0,
                                       const v16f& s_acc_n1,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int base_lane = k_sub * 32;

    for (int i = 0; i < 16; i++) {
        float local_sum = s_acc_n0[i] + s_acc_n1[i];
        for (int offset = 16; offset >= 1; offset >>= 1) {
            int src = base_lane | ((lane_id & 31) ^ offset);
            float other = bpermute_f32(src, local_sum);
            local_sum += other;
        }
        row_sum[i] = local_sum;
    }
}

// Old rescale_o_acc with array input
__device__ __forceinline__ void rescale_o_acc(v16f& o_acc_n0, v16f& o_acc_n1,
                                     const float (&old_max)[16],
                                     const float (&new_max)[16]) {
    for (int i = 0; i < 16; i++) {
        float factor = __builtin_amdgcn_exp2f(old_max[i] - new_max[i]);
        o_acc_n0[i] *= factor;
        o_acc_n1[i] *= factor;
    }
}
