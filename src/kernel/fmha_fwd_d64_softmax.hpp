#pragma once
#include "fmha_fwd_d64_gemm.hpp"

// ---- Row max across 64 N-columns ----
//
// Each lane holds s_acc_n0[i] and s_acc_n1[i] for the SAME M row i.
// s_acc_n0[i] is column n_pos, s_acc_n1[i] is column 32+n_pos.
// 32 lanes (with the same k_sub) share the same 16 M rows.
// Butterfly reduction across those 32 lanes gives the row max.

__device__ __forceinline__ void softmax_row_max(float (&row_max)[16],
                                       const v16f& s_acc_n0,
                                       const v16f& s_acc_n1,
                                       int lane_id) {
    int k_sub = lane_id >> 5;
    int base_lane = k_sub * 32;

    for (int i = 0; i < 16; i++) {
        float local_max = fmaxf(s_acc_n0[i], s_acc_n1[i]);

        // Butterfly reduction across 32 lanes sharing the same M rows
        for (int offset = 16; offset >= 1; offset >>= 1) {
            int src = base_lane | ((lane_id & 31) ^ offset);
            float other = bpermute_f32(src, local_max);
            local_max = fmaxf(local_max, other);
        }
        row_max[i] = local_max;
    }
}

// ---- Softmax exp: P = exp2(scale * (S - max)) ----

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

// ---- Row sum across 64 N-columns ----

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

// ---- Rescale O_acc when max changes between K-tile iterations ----

__device__ __forceinline__ void rescale_o_acc(v16f& o_acc_n0, v16f& o_acc_n1,
                                     const float (&old_max)[16],
                                     const float (&new_max)[16]) {
    for (int i = 0; i < 16; i++) {
        float factor = __builtin_amdgcn_exp2f(old_max[i] - new_max[i]);
        o_acc_n0[i] *= factor;
        o_acc_n1[i] *= factor;
    }
}

// ---- Cast P (fp32) to bf16 pairs for GEMM1 A operand ----

__device__ __forceinline__ void cast_p_to_bf16(v4h (&p_bf16)[16],
                                      const v16f& s_acc_n0,
                                      const v16f& s_acc_n1) {
    for (int i = 0; i < 16; i++) {
        uint32_t u0 = __builtin_bit_cast(uint32_t, s_acc_n0[i]);
        uint32_t u1 = __builtin_bit_cast(uint32_t, s_acc_n1[i]);
        int lo = static_cast<int>(u0 & 0xFFFF0000u);
        int hi = static_cast<int>(u1 & 0xFFFF0000u);
        p_bf16[i] = pack_short4(lo, hi);
    }
}
