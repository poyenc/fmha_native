#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fmha_fwd_d64_lds.hpp"
#include "fmha_fwd_d64_gemm.hpp"
#include "fmha_fwd_d64_softmax.hpp"

// Build an __amdgpu_buffer_rsrc_t from a pre-offset base pointer.
__device__ inline __amdgpu_buffer_rsrc_t make_buffer_resource(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

// Convert fp32 to bf16 by truncation (device side).
__device__ inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return static_cast<uint16_t>(u >> 16);
}

// FMHA forward D64 device function.
//
// Current implementation: GEMM0 only (S_acc = Q x K^T).
// Processes a single K/V tile (seqlen_k <= 64).
template <bool HasMask, bool IsVarlen>
__device__ void fmha_fwd_d64_device(const FmhaFwdParams& params,
                                    char* lds,
                                    int batch_idx,
                                    int head_idx,
                                    int m_tile_idx) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;

    // M-dimension mapping (Q rows)
    const int m_local = warp_id * 32 + (lane_id & 31);
    const int m_row   = m_tile_idx * kM0 + m_local;

    // GQA head mapping
    const int nhead_ratio = params.nhead_q / params.nhead_k;
    const int kv_head_idx = head_idx / nhead_ratio;

    // ---- Base pointers ----
    const __hip_bfloat16* q_base =
        params.q + static_cast<int64_t>(batch_idx) * params.batch_stride_q
                  + static_cast<int64_t>(head_idx)  * params.nhead_stride_q;
    const __hip_bfloat16* k_base =
        params.k + static_cast<int64_t>(batch_idx) * params.batch_stride_k
                  + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_k;
    __hip_bfloat16* o_base =
        params.o + static_cast<int64_t>(batch_idx) * params.batch_stride_o
                  + static_cast<int64_t>(head_idx)  * params.nhead_stride_o;

    // ---- Build SRDs ----
    auto srd_q = make_buffer_resource(q_base);
    auto srd_k = make_buffer_resource(k_base);

    // ---- Load Q: 8 x buffer_load_b128 = 64 bf16 per thread ----
    const int q_row_bytes = m_row * (params.stride_q * 2);

    v4i q_regs[8];
    if (m_row < params.seqlen_q) {
        for (int i = 0; i < 8; i++) {
            q_regs[i] = __builtin_amdgcn_raw_buffer_load_b128(
                srd_q, q_row_bytes + i * 16, 0, 0);
        }
    } else {
        for (int i = 0; i < 8; i++) {
            q_regs[i] = v4i{0, 0, 0, 0};
        }
    }

    // ---- Copy K to LDS ----
    // Load both k0 slices to LDS buffers 0 and 1.
    const int lds_buf0_bytes = 0;
    const int lds_buf1_bytes = kSingleSmemElements * 2; // 4608 bytes

    copy_k_to_lds_2x(srd_k, params.stride_k, lds, lds_buf0_bytes, lds_buf1_bytes);
    // Ensure all LDS writes are visible
    s_waitcnt_lgkmcnt_0();
    s_barrier();

    // ---- GEMM0: S_acc = Q x K^T ----
    v16f s_acc_n0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    v16f s_acc_n1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    gemm0(s_acc_n0, s_acc_n1, q_regs, lds, lds_buf0_bytes, lds_buf1_bytes, lane_id);

    // Apply scale
    for (int i = 0; i < 16; i++) {
        s_acc_n0[i] *= params.scale;
        s_acc_n1[i] *= params.scale;
    }

    // ---- Softmax: P = softmax(S_acc) via online algorithm with exp2 ----
    //
    // s_acc values are already in log2-scaled space (scale includes log2e).
    // 1. Row max across 64 N-columns
    // 2. exp2(S - max) for each element
    // 3. Row sum of exp values
    // 4. Normalize: P = exp / sum

    float rmax[16];
    softmax_row_max(rmax, s_acc_n0, s_acc_n1, lane_id);

    softmax_exp(s_acc_n0, s_acc_n1, rmax);

    float rsum[16];
    softmax_row_sum(rsum, s_acc_n0, s_acc_n1, lane_id);

    // Normalize by row sum
    for (int i = 0; i < 16; i++) {
        float inv_sum = 1.0f / rsum[i];
        s_acc_n0[i] *= inv_sum;
        s_acc_n1[i] *= inv_sum;
    }

    // ---- Store P to O (debug: bf16 for verification) ----
    const int k_sub = lane_id >> 5;
    const int n_pos = lane_id & 31;

    for (int i = 0; i < 16; i++) {
        int q_row_i = m_tile_idx * kM0 + warp_id * 32 + k_sub * 16 + i;
        if (q_row_i >= params.seqlen_q) continue;

        __hip_bfloat16* o_row = o_base + static_cast<int64_t>(q_row_i) * params.stride_o;

        uint16_t bf16_n0 = fp32_to_bf16(s_acc_n0[i]);
        uint16_t bf16_n1 = fp32_to_bf16(s_acc_n1[i]);

        o_row[n_pos]      = *reinterpret_cast<const __hip_bfloat16*>(&bf16_n0);
        o_row[32 + n_pos] = *reinterpret_cast<const __hip_bfloat16*>(&bf16_n1);
    }
}
