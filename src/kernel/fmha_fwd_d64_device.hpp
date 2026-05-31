#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"
#include "fmha_fwd_d64_lds.hpp"
#include "fmha_fwd_d64_gemm.hpp"
#include "fmha_fwd_d64_softmax.hpp"
#include "fmha_fwd_d64_epilog.hpp"

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_resource(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

constexpr int LdsSeq[4] = {1, 2, 1, 0};

template <bool HasMask, bool IsVarlen>
__device__ __forceinline__ void fmha_fwd_d64_device(const FmhaFwdParams& params,
                                    char* lds,
                                    int batch_idx,
                                    int head_idx,
                                    int m_tile_idx) {
    const int lane_id = threadIdx.x & 63;
    const int warp_id = threadIdx.x >> 6;
    const int k_sub   = lane_id >> 5;
    const int m_row   = (lane_id & 31) + 32 * warp_id;

    const int nhead_ratio = params.nhead_q / params.nhead_k;
    const int kv_head_idx = head_idx / nhead_ratio;

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

    auto srd_q = make_buffer_resource(q_base);
    auto srd_k = make_buffer_resource(k_base);
    auto srd_v = make_buffer_resource(v_base);

    int seqlen_k_start = 0;
    int seqlen_k_end   = seqlen_k;
    int mask_shift = seqlen_k - seqlen_q;

    if constexpr (HasMask) {
        int last_q_row = m_tile_idx * kM0 + kM0 - 1;
        if (last_q_row >= seqlen_q) last_q_row = seqlen_q - 1;
        int raw_end = last_q_row + mask_shift + 1;
        if (raw_end > seqlen_k) raw_end = seqlen_k;
        seqlen_k_end = ((raw_end + kN0 - 1) / kN0) * kN0;
        if (seqlen_k_end > seqlen_k) seqlen_k_end = seqlen_k;
        seqlen_k_start = 0;
    }

    int num_total_loop = (seqlen_k_end - seqlen_k_start + kN0 - 1) / kN0;

    v16f o_acc_d0, o_acc_d1;
    clear_acc(o_acc_d0);
    clear_acc(o_acc_d1);

    if (num_total_loop <= 0) {
        float* lse_base = nullptr;
        if (params.lse) {
            if constexpr (IsVarlen) {
                int nhead_stride_lse = params.nhead_stride_q / params.stride_q;
                lse_base = params.lse + static_cast<int64_t>(head_idx) * nhead_stride_lse + offset_q;
            } else {
                lse_base = params.lse
                    + static_cast<int64_t>(batch_idx) * (params.nhead_q * params.seqlen_q)
                    + static_cast<int64_t>(head_idx) * params.seqlen_q;
            }
        }
        epilog_store(o_acc_d0, o_acc_d1, 0.0f, -INFINITY,
                     params.stride_o, lse_base, seqlen_q, m_tile_idx, o_base);
        return;
    }

    const int abs_m_row = m_tile_idx * kM0 + m_row;
    const int q_stride_bytes = params.stride_q * 2;

    v4i q_regs[4];
    if (abs_m_row < seqlen_q) {
        #pragma unroll
        for (int kstep = 0; kstep < 4; ++kstep) {
            int hd = kstep * 16 + k_sub * 8;
            int voff = abs_m_row * q_stride_bytes + hd * 2;
            q_regs[kstep] = __builtin_amdgcn_raw_buffer_load_b128(srd_q, voff, 0, 0);
        }
    } else {
        #pragma unroll
        for (int kstep = 0; kstep < 4; ++kstep)
            q_regs[kstep] = v4i{0, 0, 0, 0};
    }

    float rmax = -INFINITY;
    float rsum = 0.0f;

    int kv_offset = seqlen_k_start;
    int k_col_offset = 0;

    // After Q load, before K prefetch — match CK prologue barriers 1-2
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_sched_barrier(0);

    // PROLOGUE
    async_copy_k_subtile(lds, srd_k, params.stride_k, kv_offset, k_col_offset, LdsSeq[0]);
    k_col_offset += kK0;

    __builtin_amdgcn_sched_barrier(0); // prologue barrier 3

    // TILE LOOP
    int i_total_loops = 0;
    __builtin_amdgcn_sched_barrier(0); // prologue barrier 4
    do {
        // GEMM0
        v16f s_acc_n0, s_acc_n1;
        clear_acc(s_acc_n0);
        clear_acc(s_acc_n1);

        {
            async_copy_k_subtile(lds, srd_k, params.stride_k, kv_offset, k_col_offset, LdsSeq[1]);
            k_col_offset += kK0;
            async_copy_fence();
            s_barrier();
            __builtin_amdgcn_sched_barrier(0); // hot-loop barrier 5 — GEMM0 entry
            gemm0_subtile(s_acc_n0, s_acc_n1, slice_q(q_regs, 0), lds, LdsSeq[0]);
        }

        {
            async_copy_fence();
            s_barrier();
            v2i v_k3_0, v_k3_1;
            load_v_from_dram(v_k3_0, v_k3_1, srd_v, params.stride_v, kv_offset);
            __builtin_amdgcn_sched_barrier(0); // hot-loop barrier 6 — after V prefetch
            gemm0_subtile(s_acc_n0, s_acc_n1, slice_q(q_regs, 1), lds, LdsSeq[1]);

            __builtin_amdgcn_sched_barrier(0); // hot-loop barrier 7 — GEMM0 tail

            __builtin_amdgcn_sched_barrier(0x1); // hot-loop barrier 8 — GEMM0 exit, VALU-only

            // V staging
            s_waitcnt_vmcnt_0();
            store_v_to_lds(v_k3_0, v_k3_1, lds, LdsSeq[2]);
            v2i v1_k3_0, v1_k3_1;
            load_v_from_dram(v1_k3_0, v1_k3_1, srd_v, params.stride_v, kv_offset + 32);
            s_waitcnt_vmcnt_0();

            __builtin_amdgcn_sched_barrier(0); // hot-loop barrier 10 — post V-staging

            // Softmax
            float scale_s_log2e = params.scale;
            softmax_scale_and_mask<HasMask>(s_acc_n0, s_acc_n1, scale_s_log2e,
                                            seqlen_k, kv_offset, abs_m_row, mask_shift);
            float m_new = softmax_row_max(s_acc_n0, s_acc_n1);
            __builtin_amdgcn_sched_barrier(0x7F); // hot-loop barrier 9 — after bpermute, all non-MFMA
            if (m_new == -INFINITY) m_new = rmax;
            softmax_exp2(s_acc_n0, s_acc_n1, m_new);
            float l_new = softmax_row_sum(s_acc_n0, s_acc_n1);

            if (i_total_loops > 0) {
                if (rmax != m_new) {
                    rescale_o_acc(o_acc_d0, o_acc_d1, rmax, m_new);
                    rsum = __builtin_amdgcn_exp2f(rmax - m_new) * rsum + l_new;
                } else {
                    rsum += l_new;
                }
            } else {
                rsum = l_new;
            }
            rmax = m_new;

            softmax_p_to_bf16(s_acc_n0, s_acc_n1);

            // GEMM1
            v4h p_packed_0[4];
            pack_p_subtile(p_packed_0, s_acc_n0);

            {
                s_barrier();
                gemm1_subtile(o_acc_d0, o_acc_d1, p_packed_0, lds, LdsSeq[2]);
                s_waitcnt_vmcnt_0();
                store_v_to_lds(v1_k3_0, v1_k3_1, lds, LdsSeq[3]);
            }

            i_total_loops++;
            if (i_total_loops < num_total_loop) {
                kv_offset += kN0;
                k_col_offset = 0;
                s_barrier();
                async_copy_k_subtile(lds, srd_k, params.stride_k, kv_offset, k_col_offset, LdsSeq[0]);
                k_col_offset += kK0;
            }

            {
                v4h p_packed_1[4];
                pack_p_subtile(p_packed_1, s_acc_n1);
                s_barrier();
                gemm1_subtile(o_acc_d0, o_acc_d1, p_packed_1, lds, LdsSeq[3]);
            }


        }

    } while (i_total_loops < num_total_loop);

    // EPILOGUE
    float* lse_base = nullptr;
    if (params.lse) {
        if constexpr (IsVarlen) {
            int nhead_stride_lse = params.nhead_stride_q / params.stride_q;
            lse_base = params.lse + static_cast<int64_t>(head_idx) * nhead_stride_lse + offset_q;
        } else {
            lse_base = params.lse
                + static_cast<int64_t>(batch_idx) * (params.nhead_q * params.seqlen_q)
                + static_cast<int64_t>(head_idx) * params.seqlen_q;
        }
    }

    epilog_store(o_acc_d0, o_acc_d1, rsum, rmax,
                 params.stride_o, lse_base, seqlen_q, m_tile_idx, o_base);
}
