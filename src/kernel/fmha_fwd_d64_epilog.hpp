#pragma once
#include "fmha_fwd_d64_gemm.hpp"

// ================================================================
// Phase 2: Epilog — normalize O_acc, compute LSE, bf16 store
// ================================================================
//
// O_acc layout (TransposedC + SwizzleA):
//   m_row = (lane%32) + 32*warp   — each lane owns ONE M-row
//   d_col = swz((r/8)*16 + k_sub*8 + (r%8))
//   where swz swaps bits 2,3.
//
// Store: 8 × buffer_store_dwordx2 (4 bf16 per store = 32 bf16 total).
// bf16 truncation (not RNE). From Phase 1 K7.
//
// LSE: log(rsum) + rmax / log2e. Stored to lse_base[m_row].

__device__ __forceinline__ void epilog_store(
    v16f& o_acc_d0, v16f& o_acc_d1,
    float rsum,
    float rmax,
    __amdgpu_buffer_rsrc_t o_srd,
    int stride_o,             // in bf16 elements
    float* lse_base,
    int seqlen_q,
    int m_tile_idx,
    __hip_bfloat16* o_base = nullptr)  // raw pointer for element-wise store
{
    const int lane_id = threadIdx.x & 63;
    const int warp_id = threadIdx.x >> 6;
    const int k_sub   = lane_id >> 5;
    const int m_row   = (lane_id & 31) + 32 * warp_id;

    // Guard: skip if this thread's M-row is OOB
    if (m_tile_idx * kM0 + m_row >= seqlen_q) return;

    // Normalize O_acc by rsum
    float inv_sum = (rsum > 0.0f) ? 1.0f / rsum : 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o_acc_d0[i] *= inv_sum;
        o_acc_d1[i] *= inv_sum;
    }

    // Store LSE
    constexpr float kLog2e = 1.4426950408889634f;
    if (lse_base && k_sub == 0) {
        int abs_m_row = m_tile_idx * kM0 + m_row;
        float lse_val = (rsum > 0.0f)
            ? (__builtin_amdgcn_logf(rsum) + rmax) * 0.6931471805599453f
            : -INFINITY;
        lse_base[abs_m_row] = lse_val;
    }

    // Element-wise store: d_col = swz((r/8)*16 + k_sub*8 + (r%8))
    const int abs_m_row = m_tile_idx * kM0 + m_row;

    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        const v16f& o_acc = (r < 16) ? o_acc_d0 : o_acc_d1;
        int local_r = (r < 16) ? r : r - 16;
        int d_nom = (r / 8) * 16 + k_sub * 8 + (r % 8);
        int d_col = swz(d_nom);
        uint16_t bf = f32_to_bf16_trunc(o_acc[local_r]);
        o_base[abs_m_row * stride_o + d_col] = *reinterpret_cast<const __hip_bfloat16*>(&bf);
    }
}

// ================================================================
// Legacy function — used by current _device.hpp until Task 2.6.
// DO NOT use in new Phase 2 code. Will be removed after 2.6.
// ================================================================

__device__ inline void epilog_store_o(v16f& o_acc_n0, v16f& o_acc_n1,
                                       const float (&rsum)[16],
                                       const float (&rmax)[16],
                                       __hip_bfloat16* o_base,
                                       int stride_o,
                                       float* lse_base,
                                       float log2e,
                                       int seqlen_q,
                                       int m_tile_idx,
                                       int warp_id,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int n_pos = lane_id & 31;

    for (int i = 0; i < 16; i++) {
        int m_row = m_tile_idx * kM0 + warp_id * 32 + k_sub * 16 + i;
        if (m_row >= seqlen_q) continue;

        float inv_sum = (rsum[i] > 0.0f) ? 1.0f / rsum[i] : 0.0f;
        float o_n0 = o_acc_n0[i] * inv_sum;
        float o_n1 = o_acc_n1[i] * inv_sum;

        __hip_bfloat16* o_row = o_base + static_cast<int64_t>(m_row) * stride_o;

        uint16_t bf_n0 = f32_to_bf16_trunc(o_n0);
        uint16_t bf_n1 = f32_to_bf16_trunc(o_n1);

        o_row[n_pos]      = *reinterpret_cast<const __hip_bfloat16*>(&bf_n0);
        o_row[32 + n_pos] = *reinterpret_cast<const __hip_bfloat16*>(&bf_n1);

        if (lse_base && n_pos == 0) {
            float lse_val = (rsum[i] > 0.0f)
                ? __builtin_amdgcn_logf(rsum[i]) * 0.6931471805599453f + rmax[i] / log2e
                : -INFINITY;
            lse_base[m_row] = lse_val;
        }
    }
}
