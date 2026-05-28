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
