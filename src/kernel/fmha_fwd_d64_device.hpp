#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fmha_fwd_d64_lds.hpp"
#include "fmha_fwd_d64_gemm.hpp"
#include "fmha_fwd_d64_softmax.hpp"
#include "fmha_fwd_d64_epilog.hpp"

// Build an __amdgpu_buffer_rsrc_t from a pre-offset base pointer.
__device__ inline __amdgpu_buffer_rsrc_t make_buffer_resource(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

// FMHA forward D64 device function.
//
// Full FA v2 pipeline: multi-tile K/V loop with online softmax.
// For each K/V tile:
//   1. Copy K to LDS
//   2. GEMM0: S = Q * K^T
//   3. Online softmax update (rescale O_acc, compute P)
//   4. GEMM1: O_acc += P * V (bpermute-based, correctness first)
// Epilog: O = O_acc / row_sum, cast to bf16, store to DRAM.
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
    const __hip_bfloat16* v_base =
        params.v + static_cast<int64_t>(batch_idx) * params.batch_stride_v
                  + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_v;
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

    // ---- Initialize online softmax state ----
    float rmax[16], rsum[16];
    for (int i = 0; i < 16; i++) {
        rmax[i] = -INFINITY;
        rsum[i] = 0.0f;
    }
    v16f o_acc_n0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    v16f o_acc_n1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // ---- LDS buffer offsets ----
    const int lds_buf0_bytes = 0;
    const int lds_buf1_bytes = kSingleSmemElements * 2; // 4608 bytes

    const int num_kv_tiles = (params.seqlen_k + kN0 - 1) / kN0;

    // ---- Main K/V tile loop ----
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_offset = kv_tile * kN0;

        // ---- Copy K[kv_tile] to LDS (2 k0 slices) ----
        copy_k_to_lds_2x_guarded(srd_k, params.stride_k, lds,
                                  lds_buf0_bytes, lds_buf1_bytes,
                                  kv_offset, params.seqlen_k);
        s_waitcnt_lgkmcnt_0();
        s_barrier();

        // ---- GEMM0: S_acc = Q x K^T ----
        v16f s_acc_n0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        v16f s_acc_n1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        gemm0(s_acc_n0, s_acc_n1, q_regs, lds, lds_buf0_bytes, lds_buf1_bytes, lane_id);

        // ---- Scale ----
        for (int i = 0; i < 16; i++) {
            s_acc_n0[i] *= params.scale;
            s_acc_n1[i] *= params.scale;
        }

        // ---- Mask out-of-bounds seqlen_k positions ----
        {
            int n_pos = lane_id & 31;
            // s_acc_n0[i] is at seqlen_k position = n_pos
            // s_acc_n1[i] is at seqlen_k position = 32 + n_pos
            int seqk_n0 = kv_offset + n_pos;
            int seqk_n1 = kv_offset + 32 + n_pos;
            if (seqk_n0 >= params.seqlen_k) {
                for (int i = 0; i < 16; i++) s_acc_n0[i] = -INFINITY;
            }
            if (seqk_n1 >= params.seqlen_k) {
                for (int i = 0; i < 16; i++) s_acc_n1[i] = -INFINITY;
            }
        }

        // ---- Online softmax update ----
        float new_max[16];
        softmax_row_max(new_max, s_acc_n0, s_acc_n1, lane_id);

        // Rescale O_acc and old_sum when max changes
        if (kv_tile > 0) {
            rescale_o_acc(o_acc_n0, o_acc_n1, rmax, new_max);
            for (int i = 0; i < 16; i++)
                rsum[i] *= __builtin_amdgcn_exp2f(rmax[i] - new_max[i]);
        }

        // P = exp2(S - new_max)
        softmax_exp(s_acc_n0, s_acc_n1, new_max);

        // Accumulate row sums
        float new_sum[16];
        softmax_row_sum(new_sum, s_acc_n0, s_acc_n1, lane_id);
        for (int i = 0; i < 16; i++)
            rsum[i] += new_sum[i];

        // Update max
        for (int i = 0; i < 16; i++)
            rmax[i] = new_max[i];

        // ---- GEMM1: O_acc += P * V (bpermute-based) ----
        // s_acc_n0/n1 now contain P (unnormalized exp values).
        // V is loaded directly from DRAM per seqlen_k position.
        gemm1_bpermute(o_acc_n0, o_acc_n1, s_acc_n0, s_acc_n1,
                       v_base, params.stride_v, kv_offset, params.seqlen_k,
                       lane_id);

        // Barrier before next iteration's K copy overwrites LDS
        s_barrier();
    }

    // ---- Epilog: normalize O by row_sum, cast to bf16, store ----
    epilog_store_o(o_acc_n0, o_acc_n1, rsum, o_base, params.stride_o,
                   params.seqlen_q, m_tile_idx, warp_id, lane_id);
}
