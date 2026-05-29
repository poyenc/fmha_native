#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fmha_fwd_d64_lds.hpp"
#include "fmha_fwd_d64_gemm.hpp"
#include "fmha_fwd_d64_softmax.hpp"
#include "fmha_fwd_d64_epilog.hpp"

// Build an __amdgpu_buffer_rsrc_t from a pre-offset base pointer.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_resource(const void* base) {
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
__device__ __forceinline__ void fmha_fwd_d64_device(const FmhaFwdParams& params,
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

    // ---- Per-sequence lengths and offsets (varlen) ----
    int seqlen_q, seqlen_k;
    int offset_q = 0, offset_k = 0;
    if constexpr (IsVarlen) {
        offset_q = params.seqstart_q[batch_idx];
        offset_k = params.seqstart_k[batch_idx];
        seqlen_q = params.seqstart_q[batch_idx + 1] - offset_q;
        seqlen_k = params.seqstart_k[batch_idx + 1] - offset_k;
        if (m_tile_idx * kM0 >= seqlen_q) return;
    } else {
        seqlen_q = params.seqlen_q;
        seqlen_k = params.seqlen_k;
    }

    // ---- Base pointers ----
    const __hip_bfloat16* q_base;
    const __hip_bfloat16* k_base;
    const __hip_bfloat16* v_base;
    __hip_bfloat16* o_base;
    if constexpr (IsVarlen) {
        q_base = params.q + static_cast<int64_t>(head_idx)    * params.nhead_stride_q
                           + static_cast<int64_t>(offset_q)   * params.stride_q;
        k_base = params.k + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_k
                           + static_cast<int64_t>(offset_k)   * params.stride_k;
        v_base = params.v + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_v
                           + static_cast<int64_t>(offset_k)   * params.stride_v;
        o_base = params.o + static_cast<int64_t>(head_idx)    * params.nhead_stride_o
                           + static_cast<int64_t>(offset_q)   * params.stride_o;
    } else {
        q_base = params.q + static_cast<int64_t>(batch_idx) * params.batch_stride_q
                           + static_cast<int64_t>(head_idx)  * params.nhead_stride_q;
        k_base = params.k + static_cast<int64_t>(batch_idx) * params.batch_stride_k
                           + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_k;
        v_base = params.v + static_cast<int64_t>(batch_idx) * params.batch_stride_v
                           + static_cast<int64_t>(kv_head_idx) * params.nhead_stride_v;
        o_base = params.o + static_cast<int64_t>(batch_idx) * params.batch_stride_o
                           + static_cast<int64_t>(head_idx)  * params.nhead_stride_o;
    }

    // ---- Build SRDs ----
    auto srd_q = make_buffer_resource(q_base);
    auto srd_k = make_buffer_resource(k_base);
    auto srd_v = make_buffer_resource(v_base);
    auto srd_o = make_buffer_resource(o_base);

    // ---- Load Q: 8 x buffer_load_b128 = 64 bf16 per thread ----
    const int q_row_bytes = m_row * (params.stride_q * 2);

    v4i q_regs[8];
    if (m_row < seqlen_q) {
        for (int i = 0; i < 8; i++) {
            q_regs[i] = __builtin_amdgcn_raw_buffer_load_b128(
                srd_q, q_row_bytes + i * 16, 0, 0);
        }
    } else {
        for (int i = 0; i < 8; i++) {
            q_regs[i] = v4i{0, 0, 0, 0};
        }
    }

    __builtin_amdgcn_sched_barrier(0);  // Phase 1: Q load complete

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

    const int num_kv_tiles = (seqlen_k + kN0 - 1) / kN0;

    // ---- Main K/V tile loop ----
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_offset = kv_tile * kN0;

        // Tile-level causal skip: if every element in this K/V tile
        // is above the causal diagonal for all Q rows in this workgroup,
        // skip the entire tile.
        if constexpr (HasMask) {
            int last_q_row = m_tile_idx * kM0 + kM0 - 1;
            if (last_q_row >= seqlen_q)
                last_q_row = seqlen_q - 1;
            int shift = seqlen_k - seqlen_q;
            if (kv_offset > last_q_row + shift) continue;
        }

        // ---- Phase: K copy to LDS ----
        copy_k_to_lds_2x_guarded(srd_k, params.stride_k, lds,
                                  lds_buf0_bytes, lds_buf1_bytes,
                                  kv_offset, seqlen_k);
        s_waitcnt_lgkmcnt_0();
        s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // ---- s_acc scope: GEMM0 → scale/mask → softmax → V → GEMM1 ----
        {
            v16f s_acc_n0 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            v16f s_acc_n1 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

            // ---- Phase: GEMM0 ----
            gemm0(s_acc_n0, s_acc_n1, q_regs, lds,
                  lds_buf0_bytes, lds_buf1_bytes, lane_id);
            __builtin_amdgcn_sched_barrier(0);

            // ---- Phase: Scale + mask ----
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                s_acc_n0[i] *= params.scale;
                s_acc_n1[i] *= params.scale;
            }
            {
                int n_pos = lane_id & 31;
                int seqk_n0 = kv_offset + n_pos;
                int seqk_n1 = kv_offset + 32 + n_pos;
                if (seqk_n0 >= seqlen_k) {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) s_acc_n0[i] = -INFINITY;
                }
                if (seqk_n1 >= seqlen_k) {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) s_acc_n1[i] = -INFINITY;
                }
            }
            if constexpr (HasMask) {
                int k_sub = lane_id >> 5;
                int n_pos = lane_id & 31;
                int n_n0  = kv_offset + n_pos;
                int n_n1  = kv_offset + 32 + n_pos;
                int shift = seqlen_k - seqlen_q;
                for (int i = 0; i < 16; i++) {
                    int m_idx = m_tile_idx * kM0 + warp_id * 32 + k_sub * 16 + i;
                    if (n_n0 > m_idx + shift) s_acc_n0[i] = -INFINITY;
                    if (n_n1 > m_idx + shift) s_acc_n1[i] = -INFINITY;
                }
            }

            // ---- Phase: Softmax ----
            {
                float new_max[16];
                softmax_row_max(new_max, s_acc_n0, s_acc_n1, lane_id);
                // Merge new_max with rmax: keep rmax when new tile is fully
                // masked (-INFINITY), preventing exp2f(finite - (-INF)) = INF
                // from destroying accumulated o_acc and rsum.
                for (int i = 0; i < 16; i++) {
                    if (new_max[i] == -INFINITY)
                        new_max[i] = rmax[i];
                }
                if (kv_tile > 0) {
                    rescale_o_acc(o_acc_n0, o_acc_n1, rmax, new_max);
                    for (int i = 0; i < 16; i++)
                        rsum[i] *= __builtin_amdgcn_exp2f(rmax[i] - new_max[i]);
                }
                softmax_exp(s_acc_n0, s_acc_n1, new_max);
                float new_sum[16];
                softmax_row_sum(new_sum, s_acc_n0, s_acc_n1, lane_id);
                for (int i = 0; i < 16; i++)
                    rsum[i] += new_sum[i];
                for (int i = 0; i < 16; i++)
                    rmax[i] = new_max[i];
            } // new_max, new_sum dead
            __builtin_amdgcn_sched_barrier(0);

            // ---- GEMM1: O_acc += P * V (bpermute-based) ----
            gemm1_bpermute(o_acc_n0, o_acc_n1, s_acc_n0, s_acc_n1,
                           v_base, params.stride_v, kv_offset, seqlen_k,
                           lane_id);
        } // s_acc_n0, s_acc_n1 DEAD
        __builtin_amdgcn_sched_barrier(0);

        // Barrier before next iteration's K copy
        s_barrier();
    }

    // ---- Epilog: normalize O by row_sum, cast to bf16, store ----
    constexpr float kLog2e = 1.4426950408889634f;
    float* lse_base = nullptr;
    if (params.lse) {
        if constexpr (IsVarlen) {
            // LSE layout: [H, total_tokens] — nhead_stride_q / stride_q = total_tokens
            int nhead_stride_lse = params.nhead_stride_q / params.stride_q;
            lse_base = params.lse
                + static_cast<int64_t>(head_idx) * nhead_stride_lse
                + offset_q;
        } else {
            lse_base = params.lse
                + static_cast<int64_t>(batch_idx) * (params.nhead_q * params.seqlen_q)
                + static_cast<int64_t>(head_idx) * params.seqlen_q;
        }
    }
    epilog_store_o(o_acc_n0, o_acc_n1, rsum, rmax, o_base, params.stride_o,
                   lse_base, kLog2e,
                   seqlen_q, m_tile_idx, warp_id, lane_id);
}
