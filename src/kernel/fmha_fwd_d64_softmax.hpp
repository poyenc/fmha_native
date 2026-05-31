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

// ---- Scale + mask ----
//
// Applies scaling (by scale_s_log2e) and masking to S_acc in-place.
// Boundary mask: if n_col >= seqlen_k (absolute): -INFINITY
// Causal mask: if n_col > m_row + shift: -INFINITY
//
// n_col = kv_offset + (i/8)*16 + k_sub*8 + (i%8)      [for n0 tile]
//       = kv_offset + 32 + (i/8)*16 + k_sub*8 + (i%8)  [for n1 tile]

template <bool HasMask>
__device__ __forceinline__ void softmax_scale_and_mask(
    v16f& s_acc_n0, v16f& s_acc_n1,
    float scale_s_log2e,
    int seqlen_k,
    int kv_offset,
    int m_row,         // this thread's M-row index
    int mask_shift)    // seqlen_k - seqlen_q (for causal)
{
    const int k_sub = (threadIdx.x & 63) >> 5;

    // Compute comparison thresholds once, then compare against per-register
    // column offsets using inline adds.  This avoids the compiler hoisting
    // 14 pre-computed column-index VGPRs across the tile loop (CK pattern:
    // single base + immediate per comparison).
    const int col_base = kv_offset + k_sub * 8;
    const int seqlen_k_limit = seqlen_k - col_base;        // n_col >= seqlen_k  ⟺  offset >= limit
    const int causal_limit = m_row + mask_shift - col_base; // n_col > threshold ⟺  offset > limit

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Per-register N-column offset relative to col_base.
        // Constant per i, so the compiler can fold into immediates.
        constexpr int offsets[16] = {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};
        const int off = offsets[i];

        // Scale
        s_acc_n0[i] *= scale_s_log2e;
        s_acc_n1[i] *= scale_s_log2e;

        // Boundary mask (seqlen_k OOB)
        if (off >= seqlen_k_limit)
            s_acc_n0[i] = -INFINITY;
        if (off + 32 >= seqlen_k_limit)
            s_acc_n1[i] = -INFINITY;

        // Causal mask
        if constexpr (HasMask) {
            if (off > causal_limit)
                s_acc_n0[i] = -INFINITY;
            if (off + 32 > causal_limit)
                s_acc_n1[i] = -INFINITY;
        }
    }
}

// ---- Row max: intra-lane max + 1 ds_bpermute cross-half ----
//
// Returns 1 fp32 scalar (the row max across all 64 N-columns).
// Both k_sub halves get the same result.
// From Phase 1 K3.

__device__ __forceinline__ float softmax_row_max(
    const v16f& s_acc_n0,
    const v16f& s_acc_n1)
{
    // Intra-lane max over 32 registers (16 from n0, 16 from n1)
    float local_max = -INFINITY;
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

// ---- Exp2: P = exp2(S_scaled - row_max) ----
//
// S_acc is already scaled (by scale_s_log2e) and masked.
// Computes exp2(S - rmax) in-place. Masked values (-INFINITY) become 0.
// From Phase 1 K4.

__device__ __forceinline__ void softmax_exp2(
    v16f& s_acc_n0, v16f& s_acc_n1,
    float row_max)
{
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        if (row_max == -INFINITY) {
            s_acc_n0[i] = 0.0f;
            s_acc_n1[i] = 0.0f;
        } else {
            s_acc_n0[i] = __builtin_amdgcn_exp2f(s_acc_n0[i] - row_max);
            s_acc_n1[i] = __builtin_amdgcn_exp2f(s_acc_n1[i] - row_max);
        }
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

// ---- P bf16 truncation (not RNE) ----
//
// Truncates P fp32 values to bf16 in-place (clear lower 16 bits).
// From Phase 1 K4.

__device__ __forceinline__ void softmax_p_to_bf16(
    v16f& p_n0,
    v16f& p_n1)
{
    // Truncate fp32 to bf16-in-fp32 by clearing lower 16 bits.
    // Uses reinterpret_cast instead of __builtin_bit_cast to avoid
    // a compiler miscompile on ext_vector_type element access.
    auto* u0 = reinterpret_cast<uint32_t*>(&p_n0);
    auto* u1 = reinterpret_cast<uint32_t*>(&p_n1);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        u0[i] &= 0xFFFF0000u;
        u1[i] &= 0xFFFF0000u;
    }
}

// ---- Rescale O_acc when max changes between tiles ----
//
// o_acc *= exp2(old_max - new_max) per element.

__device__ __forceinline__ void rescale_o_acc(
    v16f& o_acc_d0, v16f& o_acc_d1,
    float old_max, float new_max)
{
    float factor = __builtin_amdgcn_exp2f(old_max - new_max);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o_acc_d0[i] *= factor;
        o_acc_d1[i] *= factor;
    }
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
