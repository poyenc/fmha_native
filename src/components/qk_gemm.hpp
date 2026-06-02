#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// =============================================================================
// COMPONENT (TEST-ONLY): GEMM0 (S = Q * K^T) — pipeline STAGE 2 of 7.
//
// Standalone isolation of attention's first matmul, used ONLY by
// tests/test_qk_gemm.cpp. Golden-verified, NOT #included by src/fused/.
// CPU oracle: src/components_ref/cpu_ref_qk_gemm.{hpp,cpp}.
//
// MENTAL MODEL of the output layout (shared across ALL components — learn it
// once here):
//   - 256 threads = 4 warps x 64 lanes. The MFMA produces "TransposedC" output:
//       m_row (seqq) = (lane%32) + 32*warp      -> each thread owns ONE M-row
//       free-dim reg r maps to col = (r/8)*16 + (lane/32)*8 + (r%8)
//   - k_sub = lane/32 (0 or 1) splits each lane into a "half"; the two halves
//     of a warp hold COMPLEMENTARY column sets, merged later via ds_bpermute.
//
// GEMM CONVENTION (CK): A operand lives in LDS, B operand in registers. For
// GEMM0: A = K (LDS), B = Q (regs). The HW MFMA arg order is swapped relative
// to the CK A/B naming, i.e. mfma(hw_a=K, hw_b=Q, C=S_acc).
//
// SwizzleA: the bare 32x32x8 MFMA emits its register dim in groups-of-4, but
// golden S_acc is groups-of-8. The fix is to read the K(A) row with bits 2,3 of
// (lane%32) swapped (see qk_swz below). With that single permutation the 16
// MFMA reproduce golden bit-for-bit. This is the ONE non-obvious trick in GEMM0.
//
// bf16 cast throughout the repo = TRUNCATION (drop low 16 bits, no rounding).
// =============================================================================
//
// Phase 1 Kernel 2 — test_qk_gemm (native GEMM0: S = Q * K^T)
//
// Reproduces CK's GEMM0 MFMA C-output (S_acc) byte/value-exact against golden
// dump_reg slot 1. Uses verified layouts:
//   K LDS  (0.6): offset_elems(j,d) = (j%4)*136 + ((j/4)%4)*32 + (j/16)*544
//                                   + (d%32) + (d/32)*2304
//   Q load (0.7): q_regs[kstep] (8 bf16) = Q[m_row, kstep*16 + (lane/32)*8 + 0..7]
//   S_acc  (0.8): m_row = (lane%32)+32*warp ; n_col = (r/8)*16+(lane/32)*8+(r%8)
//
// ★ GEMM0 SwizzleA (impl-derived, forward-model 0/4096 vs golden): the bare HW
//   v_mfma_f32_32x32x8bf16 acc layout is acc(L,r)=C[m=(r/4)*8+(L/32)*4+(r%4), n=L%32]
//   (groups-of-4 in the register dim). Golden S_acc is groups-of-8. The bridge is
//   reading the K(A) operand row with bits 2,3 of (lane%32) swapped:
//       seqk(A) = tile*32 + swz(lane&31),  swz swaps bit2<->bit3.
//   With A=K(seqk swizzled), B=Q(seqq), 16 MFMA reproduce golden S_acc exactly.
//
// Golden S_ACC is RAW Q*K^T (NO 0.125 softmax scale). CPU ref + compare use raw.

typedef int   v4i  __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef short v4h  __attribute__((ext_vector_type(4)));

constexpr int kQKLdsRegionElems = 8192; // single K region (2 hdim chunks)

using qk_lds_ptr_t = __attribute__((address_space(3))) void*;

__device__ __forceinline__ __amdgpu_buffer_rsrc_t
qk_make_srd(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

// Pack two dwords (4 bf16) into v4h for the MFMA operand.
__device__ __forceinline__ v4h qk_pack(int lo, int hi) {
    v4h r;
    __builtin_memcpy(&r, &lo, 4);
    __builtin_memcpy(reinterpret_cast<char*>(&r) + 4, &hi, 4);
    return r;
}

// SwizzleA: swap bits 2 and 3 of x (x in [0,32)).
__device__ __forceinline__ int qk_swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

// Verified K LDS element offset (single buffer, base 0).
__device__ __forceinline__ int qk_k_lds_offset(int j, int d) {
    return (j % 4) * 136 + ((j / 4) % 4) * 32 + (j / 16) * 544
         + (d % 32) + (d / 32) * 2304;
}

// Stage one 64x64 K tile DRAM->LDS via the verified async copy (Kernel 1 logic).
__device__ __forceinline__ void qk_stage_k(__amdgpu_buffer_rsrc_t k_srd,
                                           char* smem_bytes,
                                           int stride_k, int kv_offset,
                                           int seqlen_k,
                                           int warp_id, int lane_id) {
    const int stride_bytes = stride_k * 2;
    const int d_in_chunk = (lane_id & 15) * 2;
    const int n_base     = (lane_id >> 4) * 4 + warp_id; // verified factor order
    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int k_col_offset = chunk * 32;
        const int chunk_base_bytes = chunk * 4608;
        #pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            const int m0_bytes = chunk_base_bytes + warp_id * 0x110 + issue * 0x440;
            qk_lds_ptr_t lds_dst = (qk_lds_ptr_t)(smem_bytes + m0_bytes);
            const int j = issue * 16 + n_base;
            const int row = kv_offset + j;
            if (row < seqlen_k) {
                const int voffset = row * stride_bytes + (k_col_offset + d_in_chunk) * 2;
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    k_srd, lds_dst, 4, voffset, 0, 0, 0);
            }
        }
    }
}

// Q, K: row-major [seqlen, 64] bf16. out: 256*32 fp32 (S_acc raw thread-buffer order).
__global__ void test_qk_gemm_kernel(const uint16_t* __restrict__ Q,
                                    const uint16_t* __restrict__ K,
                                    float* __restrict__ out,
                                    int stride_q, int stride_k,
                                    int seqlen_q, int seqlen_k) {
    __shared__ __attribute__((aligned(16))) uint16_t smem[kQKLdsRegionElems];

    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;
    const int warp_id = tid >> 6;
    const int k_sub   = lane_id >> 5;       // 0/1 (lane half)
    const int m_row   = (lane_id & 31) + 32 * warp_id;

    char* smem_bytes = reinterpret_cast<char*>(smem);

    // Pre-zero so OOB K rows read as 0 (partial-tile S cols for seqk>=seqlen_k = 0).
    for (int e = tid; e < kQKLdsRegionElems; e += blockDim.x) smem[e] = 0;
    __syncthreads();

    // ---- Stage K into LDS (verified async copy) ----
    __amdgpu_buffer_rsrc_t k_srd = qk_make_srd(K);
    qk_stage_k(k_srd, smem_bytes, stride_k, /*kv_offset=*/0, seqlen_k, warp_id, lane_id);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    // ---- Load Q per verified 0.7 distribution: 4 v4i (kstep), 8 bf16 each ----
    // q_regs[kstep] holds headdim kstep*16 + k_sub*8 + 0..7 for this lane's m_row.
    __amdgpu_buffer_rsrc_t q_srd = qk_make_srd(Q);
    const int q_stride_bytes = stride_q * 2;
    v4i q_regs[4];
    #pragma unroll
    for (int kstep = 0; kstep < 4; ++kstep) {
        if (m_row < seqlen_q) {
            const int hd = kstep * 16 + k_sub * 8;
            const int voff = m_row * q_stride_bytes + hd * 2;
            q_regs[kstep] = __builtin_amdgcn_raw_buffer_load_b128(q_srd, voff, 0, 0);
        } else {
            q_regs[kstep] = v4i{0, 0, 0, 0};
        }
    }

    // ---- GEMM0: 16 MFMA (2 n-tiles x 4 kstep x 2 pass) ----
    v16f s_n0 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // tile0: seqk 0..31
    v16f s_n1 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // tile1: seqk 32..63

    const int seqk0 = 0  + qk_swz(lane_id & 31);  // SwizzleA on A(K) operand
    const int seqk1 = 32 + qk_swz(lane_id & 31);

    #pragma unroll
    for (int kstep = 0; kstep < 4; ++kstep) {
        const int d_base = kstep * 16 + k_sub * 8; // 8 contiguous headdim
        // K reads (A operand) for both n-tiles
        const v4i k0 = *reinterpret_cast<const v4i*>(
            smem_bytes + qk_k_lds_offset(seqk0, d_base) * 2);
        const v4i k1 = *reinterpret_cast<const v4i*>(
            smem_bytes + qk_k_lds_offset(seqk1, d_base) * 2);
        const v4i q = q_regs[kstep];

        // pass 0: headdim d_base+0..3
        {
            v4h a0 = qk_pack(k0[0], k0[1]);
            v4h a1 = qk_pack(k1[0], k1[1]);
            v4h b  = qk_pack(q[0],  q[1]);
            s_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, s_n0, 0, 0, 0);
            s_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, s_n1, 0, 0, 0);
        }
        // pass 1: headdim d_base+4..7
        {
            v4h a0 = qk_pack(k0[2], k0[3]);
            v4h a1 = qk_pack(k1[2], k1[3]);
            v4h b  = qk_pack(q[2],  q[3]);
            s_n0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, s_n0, 0, 0, 0);
            s_n1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, s_n1, 0, 0, 0);
        }
    }

    // ---- Dump S_acc in raw thread-buffer order: out[tid*32 + r] ----
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        out[tid * 32 + r]      = s_n0[r];
        out[tid * 32 + 16 + r] = s_n1[r];
    }
}
