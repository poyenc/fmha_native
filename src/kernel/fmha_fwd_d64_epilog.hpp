#pragma once
#include "fmha_fwd_d64_gemm.hpp"

// ---- Epilog: normalize O by row sum, cast to bf16, store to DRAM ----
//
// O layout (same as S from GEMM0):
//   o_acc_n0[i] at lane l = O[warp*32 + k_sub*16 + i, l%32]
//   o_acc_n1[i] at lane l = O[warp*32 + k_sub*16 + i, 32 + l%32]
//
// Normalizes by row_sum (from online softmax), converts to bf16,
// and stores to DRAM using simple element-wise writes.

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

        uint16_t bf_n0 = f32_to_bf16_dev(o_n0);
        uint16_t bf_n1 = f32_to_bf16_dev(o_n1);

        o_row[n_pos]      = *reinterpret_cast<const __hip_bfloat16*>(&bf_n0);
        o_row[32 + n_pos] = *reinterpret_cast<const __hip_bfloat16*>(&bf_n1);

        // Store LSE: lse = ln(rsum) + rmax / log2(e)
        if (lse_base && n_pos == 0) {
            float lse_val = (rsum[i] > 0.0f)
                ? __logf(rsum[i]) + rmax[i] / log2e
                : -INFINITY;
            lse_base[m_row] = lse_val;
        }
    }
}
