#pragma once
#include "fmha_fwd_d64_gemm.hpp"

// ---- Epilog: normalize O by row sum, cast to bf16, store to DRAM ----
//
// O layout (TransposedC MFMA output):
//   o_acc_n0[i] at lane l = O[warp*32 + k_sub*16 + i, l%32]
//   o_acc_n1[i] at lane l = O[warp*32 + k_sub*16 + i, 32 + l%32]
//   where k_sub = l / 32
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

        if (lse_base && n_pos == 0) {
            float lse_val = (rsum[i] > 0.0f)
                ? __builtin_amdgcn_logf(rsum[i]) * 0.6931471805599453f + rmax[i] / log2e
                : -INFINITY;
            lse_base[m_row] = lse_val;
        }
    }
}

// ================================================================
// NEW ISA-matched O store via buffer_store_dwordx2.
// Old epilog_store_o above is kept until device.hpp is re-wired.
// ================================================================

// ---- O store via buffer_store_dwordx2 ----
//
// Packs 4 consecutive M-row values (same N column) into 2 dwords (4 bf16),
// then stores via buffer_store_dwordx2 with SRD.
//
// MFMA C-output mapping (TransposedC):
//   o_acc_nX[i]: M-row = warp*32 + k_sub*16 + i, N-col = nX*32 + n_pos
//   4 consecutive registers (i%4 == 0..3) share the same k_sub and differ
//   only in (i/4)*4 offset — they're at 4 consecutive M rows.
//
// Per store: 4 bf16 from registers [4*m0_iter .. 4*m0_iter+3] at one N column.
// 8 stores total: m0_iter (0..3) × n_iter (0..1).
//
// fp32→bf16 conversion via v_perm_b32 with selector 0x07060302:
//   perm(fp32_hi, fp32_lo, 0x07060302) = {trunc(fp32_hi), trunc(fp32_lo)}

__device__ __forceinline__ void epilog_store_o_buffer(
    v16f& o_acc_n0, v16f& o_acc_n1,
    const float (&rsum)[16],
    const float (&rmax)[16],
    __amdgpu_buffer_rsrc_t o_srd,
    int stride_o,             // in bf16 elements
    float* lse_base,
    float log2e,
    int seqlen_q,
    int m_tile_idx,
    int warp_id,
    int lane_id)
{
    constexpr unsigned kFp32ToBf16Sel = 0x07060302;

    int k_sub = lane_id >> 5;
    int n_pos = lane_id & 31;

    // Normalize O by row_sum
    for (int i = 0; i < 16; i++) {
        float inv_sum = (rsum[i] > 0.0f) ? 1.0f / rsum[i] : 0.0f;
        o_acc_n0[i] *= inv_sum;
        o_acc_n1[i] *= inv_sum;
    }

    // Base voffset: encodes the M-row base and N-column position
    // m_base = start of this k_sub's 16-row region within the warp's sub-tile
    int m_base = m_tile_idx * kM0 + warp_id * 32 + k_sub * 16;
    int voffset = m_base * stride_o * 2 + n_pos * 2;

    // 8 stores: m0_iter (0..3) × n_iter (0..1)
    for (int m0_iter = 0; m0_iter < 4; m0_iter++) {
        int reg_base = m0_iter * 4;  // o_acc register group start
        int m_row_first = m_base + m0_iter * 4;
        int m_row_last = m_row_first + 3;  // last row in this group

        // Skip entire store group if all 4 rows are OOB
        if (m_row_first >= seqlen_q)
            continue;

        for (int n_iter = 0; n_iter < 2; n_iter++) {
            // Select o_acc_n0 or o_acc_n1 based on n_iter
            const v16f& o_acc = (n_iter == 0) ? o_acc_n0 : o_acc_n1;

            // Pack 4 consecutive fp32 → 2 dwords of bf16
            // reg[reg_base+0..3] → 4 bf16 at M rows m_base+m0_iter*8+{0,1,2,3}
            unsigned dw0 = __builtin_amdgcn_perm(
                __builtin_bit_cast(unsigned, o_acc[reg_base + 1]),
                __builtin_bit_cast(unsigned, o_acc[reg_base + 0]),
                kFp32ToBf16Sel);
            unsigned dw1 = __builtin_amdgcn_perm(
                __builtin_bit_cast(unsigned, o_acc[reg_base + 3]),
                __builtin_bit_cast(unsigned, o_acc[reg_base + 2]),
                kFp32ToBf16Sel);

            // Pack into v2u for buffer_store_b64
            typedef unsigned v2u __attribute__((ext_vector_type(2)));
            v2u data = {dw0, dw1};

            // Byte offset for this store:
            //   m0_iter * 4 rows * stride_o * 2 bytes + n_iter * 32 positions * 2 bytes
            int soffset = m0_iter * 4 * stride_o * 2 + n_iter * 32 * 2;

            // Guard: skip if the base M row of this 4-row group is OOB
            if (m_row_first < seqlen_q) {
                __builtin_amdgcn_raw_buffer_store_b64(data, o_srd,
                    voffset + soffset, 0, 0);
            }
        }
    }

    // Store LSE
    if (lse_base && n_pos == 0) {
        for (int i = 0; i < 16; i++) {
            int m_row = m_tile_idx * kM0 + warp_id * 32 + k_sub * 16 + i;
            if (m_row >= seqlen_q) continue;
            float lse_val = (rsum[i] > 0.0f)
                ? __builtin_amdgcn_logf(rsum[i]) * 0.6931471805599453f + rmax[i] / log2e
                : -INFINITY;
            lse_base[m_row] = lse_val;
        }
    }
}
