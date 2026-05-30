#pragma once
#include "fmha_fwd_d64_lds.hpp"

typedef float v16f __attribute__((ext_vector_type(16)));
typedef short v4h __attribute__((ext_vector_type(4)));

// Pack two dwords (4 bf16) into v4h for MFMA operand.
__device__ __forceinline__ v4h pack_short4(int lo, int hi) {
    v4h r;
    __builtin_memcpy(&r, &lo, 4);
    __builtin_memcpy(reinterpret_cast<char*>(&r) + 4, &hi, 4);
    return r;
}

// SwizzleA: swap bits 2 and 3 of x (x in [0,32)).
// Applied to K(A) operand lane index in GEMM0 to produce the groups-of-8
// accumulator layout that matches CK's golden S_acc.
__device__ __forceinline__ int swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

// K LDS element offset (padded layout, single-buffer relative).
// This is the layout produced by async DRAM→LDS copy (buffer_load_dword...lds).
// Verified in Phase 0/1 K1/K2:
//   offset(j,d) = (j%4)*136 + ((j/4)%4)*32 + (j/16)*544 + (d%32) + (d/32)*2304
__device__ __forceinline__ int lds_elem_offset(int j, int d) {
    return (j % 4) * 136 + ((j / 4) % 4) * 32 + (j / 16) * 544
         + (d % 32) + (d / 32) * 2304;
}

// V LDS element offset (padded layout, single-buffer relative).
// This is the layout produced by v_perm_b32 + ds_write2_b32 V staging.
// Verified in Phase 1 K5/K6:
//   k = n % 32; offset(n,d) = (k/8)*576 + (d/8)*72 + (d%8)*8 + (k%8)
__device__ __forceinline__ int v_lds_elem_offset(int n, int d) {
    int k = n % 32;
    return (k / 8) * 576 + (d / 8) * 72 + (d % 8) * 8 + (k % 8);
}

// Convert float to bf16 by truncation.
__device__ __forceinline__ uint16_t f32_to_bf16_trunc(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return static_cast<uint16_t>(u >> 16);
}

// Convert bf16 (in uint16_t) to float.
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return __builtin_bit_cast(float, u);
}

// ---- Accumulator helpers ----

__device__ __forceinline__ void clear_acc(v16f& acc) {
    acc = v16f{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
}

// ================================================================
// Phase 2: GEMM0 — S_acc = Q × K^T (sub-tile granularity)
// ================================================================

// Extract the Q register slice for a given K sub-tile index.
// Q is loaded as 4 v4i covering full 64 hdim:
//   q_regs[kstep] = Q[m_row, kstep*16 + k_sub*8 + 0..7]
// For k_subtile_idx=0 (hdim 0..31): ksteps 0,1
// For k_subtile_idx=1 (hdim 32..63): ksteps 2,3
__device__ __forceinline__ const v4i* slice_q(const v4i* q_regs, int k_subtile_idx) {
    return q_regs + k_subtile_idx * 2;
}

// One sub-tile of GEMM0: read K from LDS, execute 2 ksteps × 2 passes × 2 N-tiles = 8 MFMA.
//
// K is A-operand (LDS), Q is B-operand (register).
// SwizzleA on K reads: lane reads K at seqk = swz(lane & 31).
// Each kstep covers 16 hdim (2 passes of 8 bf16 each).
//
// Parameters:
//   s_acc_n0, s_acc_n1: accumulator for N-tile 0 (seqk 0..31) and 1 (seqk 32..63)
//   q_slice: 2 v4i from slice_q() for this sub-tile
//   lds: LDS base pointer
//   buf_idx: which LDS buffer (0, 1, or 2)
__device__ __forceinline__ void gemm0_subtile(
    v16f& s_acc_n0, v16f& s_acc_n1,
    const v4i* q_slice,
    char* lds,
    int buf_idx)
{
    const int lane_id = threadIdx.x & 63;
    const int k_sub   = lane_id >> 5;

    const int seqk0 = swz(lane_id & 31);        // SwizzleA: N-tile 0
    const int seqk1 = 32 + swz(lane_id & 31);   // SwizzleA: N-tile 1
    const int buf_byte_base = buf_base_bytes(buf_idx);

    #pragma unroll
    for (int kstep = 0; kstep < 2; ++kstep) {
        const int d_base = kstep * 16 + k_sub * 8;

        // K reads from LDS (A operand) for both N-tiles
        const v4i k0 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + lds_elem_offset(seqk0, d_base) * 2);
        const v4i k1 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + lds_elem_offset(seqk1, d_base) * 2);
        const v4i q = q_slice[kstep];

        // Pass 0: d_base+0..3 (4 bf16)
        {
            v4h a0 = pack_short4(k0[0], k0[1]);
            v4h a1 = pack_short4(k1[0], k1[1]);
            v4h b  = pack_short4(q[0],  q[1]);
            s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, s_acc_n0, 0, 0, 0);
            s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, s_acc_n1, 0, 0, 0);
        }
        // Pass 1: d_base+4..7 (4 bf16)
        {
            v4h a0 = pack_short4(k0[2], k0[3]);
            v4h a1 = pack_short4(k1[2], k1[3]);
            v4h b  = pack_short4(q[2],  q[3]);
            s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, s_acc_n0, 0, 0, 0);
            s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, s_acc_n1, 0, 0, 0);
        }
    }
}

// ================================================================
// Phase 2: GEMM1 — O_acc += P × V (sub-tile granularity)
// ================================================================

// Pack P (fp32 accumulator) to bf16 for MFMA B-operand.
// P truncation: fp32 → bf16 via v_perm_b32 selector 0x07060302.
// Produces 4 v4h per N-tile half (8 total for both halves of one sub-tile).
//
// For v_subtile_idx=0: P from p_n0[0..15] (seqk 0..31)
// For v_subtile_idx=1: P from p_n1[0..15] (seqk 32..63)
__device__ __forceinline__ void pack_p_subtile(
    v4h (&p_packed)[4],
    const v16f& p_half)
{
    constexpr unsigned kFp32ToBf16Sel = 0x07060302;
    const auto* u = reinterpret_cast<const unsigned*>(&p_half);

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        unsigned lo = __builtin_amdgcn_perm(
            u[4 * s + 1], u[4 * s], kFp32ToBf16Sel);
        unsigned hi = __builtin_amdgcn_perm(
            u[4 * s + 3], u[4 * s + 2], kFp32ToBf16Sel);
        p_packed[s] = pack_short4(lo, hi);
    }
}

// Extract the P slice for a given V sub-tile index.
// P is split into two N-tile halves (p_n0 = seqk 0..31, p_n1 = seqk 32..63).
// v_subtile_idx selects which half.
__device__ __forceinline__ const v16f& slice_p(
    const v16f& p_n0, const v16f& p_n1, int v_subtile_idx)
{
    return (v_subtile_idx == 0) ? p_n0 : p_n1;
}

// One sub-tile of GEMM1: read V from LDS, execute 2 ksteps × 2 passes × 2 hdim-tiles = 8 MFMA.
//
// V is A-operand (LDS), P is B-operand (register).
// NO SwizzleA on V reads — bare lane%32 for hdim position.
// The SwizzleA effect on O_acc comes from P being in SwizzleA'd layout
// (inherited from GEMM0's output).
//
// V LDS layout: same padded layout as K.
//   V element offset: lds_elem_offset(seqk_local, hdim_pos)
//   where seqk_local = seqk within the 32-row slice stored in this buffer.
//
// Parameters:
//   o_acc_d0, o_acc_d1: accumulator for hdim-tile 0 (0..31) and 1 (32..63)
//   p_packed: 4 v4h from pack_p_subtile() for this sub-tile
//   lds: LDS base pointer
//   buf_idx: which LDS buffer holds V for this sub-tile
__device__ __forceinline__ void gemm1_subtile(
    v16f& o_acc_d0, v16f& o_acc_d1,
    const v4h* p_packed,
    char* lds,
    int buf_idx)
{
    const int lane_id = threadIdx.x & 63;
    const int k_sub   = lane_id >> 5;

    // V hdim read position: bare lane%32, NO SwizzleA.
    const int hdim_pos = lane_id & 31;
    const int buf_byte_base = buf_base_bytes(buf_idx);

    #pragma unroll
    for (int kstep = 0; kstep < 2; ++kstep) {
        // seqk position within the 32-row V slice in this buffer
        const int seqk_local = kstep * 16 + k_sub * 8;

        // V(A) reads from LDS — both hdim tiles from same buffer
        // V uses the ds_write2 layout (v_lds_elem_offset), NOT the async copy layout.
        const v4i v0 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + v_lds_elem_offset(seqk_local, hdim_pos) * 2);
        const v4i v1 = *reinterpret_cast<const v4i*>(
            lds + buf_byte_base + v_lds_elem_offset(seqk_local, hdim_pos + 32) * 2);

        // P(B) operand — pre-packed bf16
        v4h b_val = p_packed[kstep * 2];

        // Pass 0: seqk +0..3
        {
            v4h a0 = pack_short4(v0[0], v0[1]);
            v4h a1 = pack_short4(v1[0], v1[1]);
            o_acc_d0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b_val, o_acc_d0, 0, 0, 0);
            o_acc_d1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b_val, o_acc_d1, 0, 0, 0);
        }

        // Pass 1: seqk +4..7
        b_val = p_packed[kstep * 2 + 1];
        {
            v4h a0 = pack_short4(v0[2], v0[3]);
            v4h a1 = pack_short4(v1[2], v1[3]);
            o_acc_d0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b_val, o_acc_d0, 0, 0, 0);
            o_acc_d1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b_val, o_acc_d1, 0, 0, 0);
        }
    }
}

// ================================================================
// Legacy functions — used by current _device.hpp until Task 2.6.
// DO NOT use in new Phase 2 code. Will be removed after 2.6.
// ================================================================

__device__ inline void gemm0(v16f& s_acc_n0, v16f& s_acc_n1,
                              const v4i* q_regs,
                              char* lds,
                              int lds_buf0_bytes,
                              int lds_buf1_bytes,
                              int lane_id) {
    int n_local = lane_id & 31;
    int k_sub   = lane_id >> 5;

    auto do_k0 = [&](int lds_buf_bytes, int q_base) {
        for (int kstep = 0; kstep < 4; kstep++) {
            int k_start = kstep * 8;
            int off_n0 = lds_buf_bytes + k_lds_offset(n_local, k_start) * 2;
            v4i k_n0 = *reinterpret_cast<const v4i*>(lds + off_n0);
            int off_n1 = lds_buf_bytes + k_lds_offset(32 + n_local, k_start) * 2;
            v4i k_n1 = *reinterpret_cast<const v4i*>(lds + off_n1);
            v4h a_n0 = pack_short4(k_n0[k_sub * 2], k_n0[k_sub * 2 + 1]);
            v4h a_n1 = pack_short4(k_n1[k_sub * 2], k_n1[k_sub * 2 + 1]);
            v4i q_reg = q_regs[q_base + kstep];
            v4h b_val = pack_short4(q_reg[k_sub * 2], q_reg[k_sub * 2 + 1]);
            s_acc_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_n0, b_val, s_acc_n0, 0, 0, 0);
            s_acc_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_n1, b_val, s_acc_n1, 0, 0, 0);
        }
    };
    do_k0(lds_buf0_bytes, 0);
    __builtin_amdgcn_sched_barrier(0);
    do_k0(lds_buf1_bytes, 4);
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
        float v_val_n0 = 0.0f, v_val_n1 = 0.0f;
        if (v_row < seqlen_k) {
            const __hip_bfloat16* v_row_ptr = v_base + static_cast<int64_t>(v_row) * stride_v;
            uint16_t bf_n0, bf_n1;
            __builtin_memcpy(&bf_n0, &v_row_ptr[n_pos], 2);
            __builtin_memcpy(&bf_n1, &v_row_ptr[32 + n_pos], 2);
            v_val_n0 = bf16_to_f32(bf_n0);
            v_val_n1 = bf16_to_f32(bf_n1);
        }
        int src_lane = k_sub * 32 + (j & 31);
        for (int i = 0; i < 16; i++) {
            float p_val;
            if (j < 32) {
                p_val = bpermute_f32(src_lane, p_n0[i]);
            } else {
                p_val = bpermute_f32(src_lane, p_n1[i]);
            }
            float p_trunc = bf16_to_f32(f32_to_bf16_trunc(p_val));
            o_acc_n0[i] += p_trunc * v_val_n0;
            o_acc_n1[i] += p_trunc * v_val_n1;
        }
    }
}
