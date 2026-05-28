#pragma once
#include "fmha_fwd_d64_lds.hpp"

typedef int v4i __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef short v4h __attribute__((ext_vector_type(4)));

// Pack two int32 values (containing 4 bf16) into short4 for MFMA.
__device__ inline v4h pack_short4(int lo, int hi) {
    v4h r;
    __builtin_memcpy(&r, &lo, 4);
    __builtin_memcpy(reinterpret_cast<char*>(&r) + 4, &hi, 4);
    return r;
}

// ---- GEMM0: S_acc = Q x K^T ----
//
// Each warp computes 32 x 64 sub-tile of S, split into two N sub-tiles
// (n0: N=0..31, n1: N=32..63).
//
// K dimension (64) is split into 2 k0 iterations of 32.
// Within each k0: 4 K-steps of 8 bf16 -> 8 MFMA per k0 (4 K-steps x 2 N sub-tiles).
// Total: 16 MFMA per warp.
//
// Per K-step: 2 ds_read_b128 (one per N sub-tile) + 2 MFMA.
// Per k0: 4 K-steps x (2 reads + 2 MFMA) = 8 reads + 8 MFMA.

// Perform GEMM0 for one k0 iteration (32 K values).
__device__ inline void gemm0_k0(v16f& s_acc_n0, v16f& s_acc_n1,
                                 char* lds,
                                 int lds_buf_bytes,
                                 const v4i* q_regs,
                                 int q_base,
                                 int lane_id) {
    int n_local = lane_id & 31;
    int k_sub   = lane_id >> 5; // 0 or 1

    for (int kstep = 0; kstep < 4; kstep++) {
        int k_start = kstep * 8;

        // LDS read for N sub-tile 0 (n = 0..31)
        int off_n0 = lds_buf_bytes + k_lds_offset(n_local, k_start) * 2;
        v4i k_n0 = *reinterpret_cast<const v4i*>(lds + off_n0);

        // LDS read for N sub-tile 1 (n = 32..63)
        int off_n1 = lds_buf_bytes + k_lds_offset(32 + n_local, k_start) * 2;
        v4i k_n1 = *reinterpret_cast<const v4i*>(lds + off_n1);

        // Extract A operand (K data) based on k_sub
        v4h a_n0 = pack_short4(k_n0[k_sub * 2], k_n0[k_sub * 2 + 1]);
        v4h a_n1 = pack_short4(k_n1[k_sub * 2], k_n1[k_sub * 2 + 1]);

        // Extract B operand (Q data) based on k_sub
        v4i q_reg = q_regs[q_base + kstep];
        v4h b_val = pack_short4(q_reg[k_sub * 2], q_reg[k_sub * 2 + 1]);

        // MFMA: D[32,32] += A[32,8] * B[8,32]
        // A = K data (lane_id%32 -> K's N position)
        // B = Q data (lane_id%32 -> Q's M position)
        s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_n0, b_val, s_acc_n0, 0, 0, 0);
        s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_n1, b_val, s_acc_n1, 0, 0, 0);
    }
}

// Full GEMM0: process both k0 iterations.
__device__ inline void gemm0(v16f& s_acc_n0, v16f& s_acc_n1,
                              const v4i* q_regs,
                              char* lds,
                              int lds_buf0_bytes,
                              int lds_buf1_bytes,
                              int lane_id) {
    // k0 = 0: Q columns 0..31, K LDS buffer 0
    gemm0_k0(s_acc_n0, s_acc_n1, lds, lds_buf0_bytes, q_regs, 0, lane_id);

    __builtin_amdgcn_sched_barrier(0);

    // k0 = 1: Q columns 32..63, K LDS buffer 1
    gemm0_k0(s_acc_n0, s_acc_n1, lds, lds_buf1_bytes, q_regs, 4, lane_id);
}

// ---- GEMM1: O_acc += P x V (scalar bpermute-based, correctness first) ----
//
// Computes O[m_q, hdim] += P[m_q, seqk] * V[seqk, hdim] for all seqk
// in the current tile.
//
// O layout (same as S from GEMM0):
//   o_acc_n0[i] at lane l = O[warp*32 + k_sub*16 + i, l%32]
//   o_acc_n1[i] at lane l = O[warp*32 + k_sub*16 + i, 32 + l%32]
//
// P layout (from softmax, same as S):
//   s_acc_n0[i] at lane l = P[warp*32 + k_sub*16 + i, l%32]
//   s_acc_n1[i] at lane l = P[warp*32 + k_sub*16 + i, 32 + l%32]
//
// For each seqk position j (0..63):
//   P[m_q_i, j] is held by lane (k_sub*32 + j%32):
//     j < 32: s_acc_n0[i] at lane (k_sub*32 + j)
//     j >= 32: s_acc_n1[i] at lane (k_sub*32 + j - 32)
//   V[kv_offset+j, hdim] is loaded from DRAM.
//
// Truncates P to bf16 before multiply (matching GPU reference precision).

// Convert bf16 (in uint16_t) to float on device.
__device__ inline float bf16_to_f32_dev(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return __builtin_bit_cast(float, u);
}

// Convert float to bf16 by truncation on device.
__device__ inline uint16_t f32_to_bf16_dev(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return static_cast<uint16_t>(u >> 16);
}

__device__ inline void gemm1_bpermute(v16f& o_acc_n0, v16f& o_acc_n1,
                                       const v16f& p_n0, const v16f& p_n1,
                                       const __hip_bfloat16* v_base,
                                       int stride_v,
                                       int kv_offset,
                                       int seqlen_k,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int n_pos = lane_id & 31;

    for (int j = 0; j < kN0; j++) {
        int v_row = kv_offset + j;

        // Load V[v_row, n_pos] and V[v_row, 32+n_pos] as bf16
        float v_val_n0 = 0.0f, v_val_n1 = 0.0f;
        if (v_row < seqlen_k) {
            const __hip_bfloat16* v_row_ptr = v_base + static_cast<int64_t>(v_row) * stride_v;
            uint16_t bf_n0, bf_n1;
            __builtin_memcpy(&bf_n0, &v_row_ptr[n_pos], 2);
            __builtin_memcpy(&bf_n1, &v_row_ptr[32 + n_pos], 2);
            v_val_n0 = bf16_to_f32_dev(bf_n0);
            v_val_n1 = bf16_to_f32_dev(bf_n1);
        }

        // For each m_q sub-row i, gather P[m_q_i, j] via bpermute
        int src_lane = k_sub * 32 + (j & 31);
        for (int i = 0; i < 16; i++) {
            float p_val;
            if (j < 32) {
                p_val = bpermute_f32(src_lane, p_n0[i]);
            } else {
                p_val = bpermute_f32(src_lane, p_n1[i]);
            }
            // Truncate P to bf16 to match GPU reference precision
            float p_trunc = bf16_to_f32_dev(f32_to_bf16_dev(p_val));
            o_acc_n0[i] += p_trunc * v_val_n0;
            o_acc_n1[i] += p_trunc * v_val_n1;
        }
    }
}
