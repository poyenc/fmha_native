#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// =============================================================================
// COMPONENT (TEST-ONLY): GEMM1 (O = P * V) — pipeline STAGE 6 of 7.
//
// Standalone isolation of attention's second matmul, used ONLY by
// tests/test_pv_gemm.cpp. Golden-verified, NOT #included by src/fused/.
// CPU oracle: src/components_ref/cpu_ref_pv_gemm.{hpp,cpp}.
//
// CONTRAST WITH GEMM0 — read this if GEMM0 already makes sense:
//   - Same CK convention (A=LDS, B=reg) and same HW arg swap, but here
//     A = V (staged into LDS by the v_lds path inlined below) and B = P (the
//     softmax output, already in registers from STAGE 4).
//   - NO explicit SwizzleA on the V reads (bare hdim = lane%32). The swizzle
//     "comes for free" because P is ALREADY in GEMM0's SwizzleA'd TransposedC
//     layout, so the resulting O_acc lands in the correct distribution:
//       m = (lane%32)+32*warp ,  d = swz((r/8)*16+(lane/32)*8+(r%8)).
//   - Contraction is over seqk (0..63), so this kernel first stages TWO 32-row
//     V slices into two LDS buffers (the inlined v_lds code), then runs 16 MFMA.
//
// P arrives as bf16-clean fp32 (low 16 bits zero); the loader repacks pairs of
// P regs into dwords by taking their upper 16 bits.
// =============================================================================
//
// Phase 1 Kernel 6 -- test_pv_gemm (GEMM1: P·V → O_acc)
//
// Reproduces CK's GEMM1 MFMA C-output (O_acc) bit-exact against golden
// dump_reg slot 5. Uses verified layouts:
//   P (CK A-operand, register): TransposedC, m=(lane%32)+32*warp,
//       n=(r/8)*16+(lane/32)*8+(r%8). 32 bf16 regs (promoted to fp32 in golden).
//   V LDS (CK B-operand): offset = base + (k/8)*576 + (d/8)*72 + (d%8)*8 + (k%8)
//   O_acc (C-output): TransposedC layout,
//       m=(lane%32)+32*warp, d=swz((r/8)*16+(lane/32)*8+(r%8))
//       where swz swaps bits 2,3 (same SwizzleA effect as GEMM0).
//
// CK convention: A=P(reg), B=V(LDS). Both GEMMs are A-in-reg / B-in-LDS.
// HW MFMA operand order is swapped: mfma(hw_a=V(LDS), hw_b=P(reg), C=O_acc).
// Same pattern as GEMM0: mfma(hw_a=K(LDS), hw_b=Q(reg), C=S_acc).
// The SwizzleA effect comes from P already being in TransposedC layout
// (produced by GEMM0's SwizzleA). No explicit swizzle on V LDS reads.
// The bare-lane V read (hdim = lane%32, un-swizzled) produces the correct
// O_acc distribution because the P data encodes the SwizzleA permutation.
//
// Structure: 16 MFMA = 2 hdim-tiles × 4 ksteps × 2 passes.
// V LDS read: v4i (4 dwords = 8 bf16) per kstep, along seqk contraction dim.

typedef int   v4i  __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef short v4h  __attribute__((ext_vector_type(4)));

constexpr int kPVLdsRegionElems = 8192;

__device__ __forceinline__ __amdgpu_buffer_rsrc_t
pv_make_srd(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

__device__ __forceinline__ v4h pv_pack(int lo, int hi) {
    v4h r;
    __builtin_memcpy(&r, &lo, 4);
    __builtin_memcpy(reinterpret_cast<char*>(&r) + 4, &hi, 4);
    return r;
}

// SwizzleA: swap bits 2 and 3 of x (x in [0,32)).
__device__ __forceinline__ int pv_swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

// V LDS element offset (same padded layout as K LDS).
__device__ __forceinline__ int pv_v_lds_offset(int n, int d) {
    int k = n % 32;
    return (k / 8) * 576 + (d / 8) * 72 + (d % 8) * 8 + (k % 8);
}

// P: bf16-promoted fp32 in thread-buffer order [256*32].
// V: raw bf16 row-major [seqlen_k, D=64].
// out: 256*32 fp32 (O_acc raw thread-buffer order).
__global__ void test_pv_gemm_kernel(const float* __restrict__ P,
                                    const uint16_t* __restrict__ V,
                                    float* __restrict__ out,
                                    int stride_v, int seqlen_k) {
    __shared__ __attribute__((aligned(16))) uint16_t smem[kPVLdsRegionElems];

    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;
    const int warp_id = tid >> 6;
    const int k_sub   = lane_id >> 5;

    char* smem_bytes = reinterpret_cast<char*>(smem);

    // Pre-zero LDS
    for (int e = tid; e < kPVLdsRegionElems; e += blockDim.x) smem[e] = 0;
    __syncthreads();

    // ---- Stage V into LDS using the verified v_perm + ds_write2 path ----
    // Load + repack + store for one 32-row k1 slice (n=0..31)
    {
        typedef int v2i __attribute__((ext_vector_type(2)));
        __amdgpu_buffer_rsrc_t v_srd = pv_make_srd(V);
        int n_hdim   = lane_id >> 2;
        int k_within = lane_id & 3;
        int stride_bytes = stride_v * 2;
        int v_row_base = warp_id * 8 + k_within * 2;
        int v_col_bytes = (n_hdim * 4) * 2;

        v2i load_k3_0, load_k3_1;
        if (v_row_base < seqlen_k) {
            load_k3_0 = __builtin_amdgcn_raw_buffer_load_b64(
                v_srd, v_row_base * stride_bytes + v_col_bytes, 0, 0);
        } else {
            load_k3_0 = (v2i){0, 0};
        }
        if (v_row_base + 1 < seqlen_k) {
            load_k3_1 = __builtin_amdgcn_raw_buffer_load_b64(
                v_srd, (v_row_base + 1) * stride_bytes + v_col_bytes, 0, 0);
        } else {
            load_k3_1 = (v2i){0, 0};
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        constexpr unsigned kPermSel0 = 0x01000504;
        constexpr unsigned kPermSel1 = 0x03020706;
        unsigned in0_lo = static_cast<unsigned>(load_k3_0[0]);
        unsigned in0_hi = static_cast<unsigned>(load_k3_0[1]);
        unsigned in1_lo = static_cast<unsigned>(load_k3_1[0]);
        unsigned in1_hi = static_cast<unsigned>(load_k3_1[1]);
        unsigned out0 = __builtin_amdgcn_perm(in0_lo, in1_lo, kPermSel0);
        unsigned out1 = __builtin_amdgcn_perm(in0_lo, in1_lo, kPermSel1);
        unsigned out2 = __builtin_amdgcn_perm(in0_hi, in1_hi, kPermSel0);
        unsigned out3 = __builtin_amdgcn_perm(in0_hi, in1_hi, kPermSel1);

        constexpr int kBufBase = 2304;
        int lds_buf_byte_offset = kBufBase * 2 - 128 * 4;
        int v86 = ((lane_id >> 3) & 7) * 144
                + ((lane_id >> 2) & 1) * 64
                + (lane_id & 3) * 4;
        int lds_addr = warp_id * 1152 + v86 + lds_buf_byte_offset;

        asm volatile("ds_write2_b32 %0, %1, %2 offset0:128 offset1:132"
                     : : "v"(lds_addr), "v"(out0), "v"(out1) : "memory");
        asm volatile("ds_write2_b32 %0, %1, %2 offset0:136 offset1:140"
                     : : "v"(lds_addr), "v"(out2), "v"(out3) : "memory");
    }

    // Stage second k1 slice (n=32..63) into second buffer
    {
        typedef int v2i __attribute__((ext_vector_type(2)));
        __amdgpu_buffer_rsrc_t v_srd = pv_make_srd(V);
        int n_hdim   = lane_id >> 2;
        int k_within = lane_id & 3;
        int stride_bytes = stride_v * 2;
        int v_row_base = 32 + warp_id * 8 + k_within * 2;
        int v_col_bytes = (n_hdim * 4) * 2;

        v2i load_k3_0, load_k3_1;
        if (v_row_base < seqlen_k) {
            load_k3_0 = __builtin_amdgcn_raw_buffer_load_b64(
                v_srd, v_row_base * stride_bytes + v_col_bytes, 0, 0);
        } else {
            load_k3_0 = (v2i){0, 0};
        }
        if (v_row_base + 1 < seqlen_k) {
            load_k3_1 = __builtin_amdgcn_raw_buffer_load_b64(
                v_srd, (v_row_base + 1) * stride_bytes + v_col_bytes, 0, 0);
        } else {
            load_k3_1 = (v2i){0, 0};
        }

        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        constexpr unsigned kPermSel0 = 0x01000504;
        constexpr unsigned kPermSel1 = 0x03020706;
        unsigned in0_lo = static_cast<unsigned>(load_k3_0[0]);
        unsigned in0_hi = static_cast<unsigned>(load_k3_0[1]);
        unsigned in1_lo = static_cast<unsigned>(load_k3_1[0]);
        unsigned in1_hi = static_cast<unsigned>(load_k3_1[1]);
        unsigned out0 = __builtin_amdgcn_perm(in0_lo, in1_lo, kPermSel0);
        unsigned out1 = __builtin_amdgcn_perm(in0_lo, in1_lo, kPermSel1);
        unsigned out2 = __builtin_amdgcn_perm(in0_hi, in1_hi, kPermSel0);
        unsigned out3 = __builtin_amdgcn_perm(in0_hi, in1_hi, kPermSel1);

        constexpr int kBufBase2 = 2304 + 2304;
        int lds_buf_byte_offset = kBufBase2 * 2 - 128 * 4;
        int v86 = ((lane_id >> 3) & 7) * 144
                + ((lane_id >> 2) & 1) * 64
                + (lane_id & 3) * 4;
        int lds_addr = warp_id * 1152 + v86 + lds_buf_byte_offset;

        asm volatile("ds_write2_b32 %0, %1, %2 offset0:128 offset1:132"
                     : : "v"(lds_addr), "v"(out0), "v"(out1) : "memory");
        asm volatile("ds_write2_b32 %0, %1, %2 offset0:136 offset1:140"
                     : : "v"(lds_addr), "v"(out2), "v"(out3) : "memory");
    }

    __syncthreads();

    // ---- Load P from DRAM (bf16-promoted fp32, truncate to bf16 pairs) ----
    // P is in TransposedC layout: p[tid*32+r].
    // For MFMA A operand: need 4 bf16 per kstep/pass.
    // P register r corresponds to seqk = (r/8)*16 + k_sub*8 + (r%8).
    // Per kstep k, pass p: read P regs at k*8 + p*4 + {0,1,2,3}.

    // Load P as bf16 pairs (pack from golden fp32-promoted format).
    // P values are bf16-clean fp32 (lower 16 bits = 0). Extract the upper 16
    // bits (the bf16) and pack pairs into dwords: lo16 = P[r], hi16 = P[r+1].
    int p_raw[16]; // 16 packed dwords = 32 bf16
    #pragma unroll
    for (int r = 0; r < 32; r += 2) {
        float p0 = P[tid * 32 + r];
        float p1 = P[tid * 32 + r + 1];
        uint32_t u0 = __builtin_bit_cast(uint32_t, p0);
        uint32_t u1 = __builtin_bit_cast(uint32_t, p1);
        // bf16 pair: lo16 = bf16(p0), hi16 = bf16(p1)
        p_raw[r / 2] = static_cast<int>((u1 & 0xFFFF0000u) | (u0 >> 16));
    }

    // ---- GEMM1: 16 MFMA (2 hdim-tiles × 4 ksteps × 2 passes) ----
    //
    // CK: A=P(reg), B=V(LDS). HW: mfma(hw_a=V(LDS), hw_b=P(reg), C=O_acc).
    // No SwizzleA on V reads — bare lane%32 for hdim. SwizzleA effect on O_acc
    // comes from P being in SwizzleA'd TransposedC layout (from GEMM0).
    // kstep iterates over the contraction dimension (seqk, 0..63 in groups of 16).
    // Two hdim-tiles: tile0 = hdim 0..31 (V buf 0), tile1 = hdim 32..63 (V buf 1).

    v16f o_d0 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // hdim 0..31
    v16f o_d1 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // hdim 32..63

    // V hdim read position: bare lane%32, NO SwizzleA on V read.
    // The SwizzleA effect on O_acc comes from P being in SwizzleA'd layout.
    const int hdim_pos = lane_id & 31;

    // V buffers: each holds 32 seqk × 64 hdim. Two k1 slices.
    constexpr int kVBuf0 = 2304;         // seqk 0..31
    constexpr int kVBuf1 = 2304 + 2304;  // seqk 32..63

    #pragma unroll
    for (int kstep = 0; kstep < 4; ++kstep) {
        // Contraction dim (seqk) position for this kstep
        const int seqk_abs = kstep * 16 + k_sub * 8; // absolute seqk (0..56)
        const int seqk_local = seqk_abs % 32;        // position within 32-row k1 slice
        // Select V buffer based on which k1 slice (0..31 → buf0, 32..63 → buf1)
        const int v_buf = (seqk_abs < 32) ? kVBuf0 : kVBuf1;

        // V(A) reads from LDS. Both hdim tiles from the SAME V buffer.
        // Tile 0: hdim 0..31 (hdim_pos in [0,31])
        // Tile 1: hdim 32..63 (hdim_pos + 32)
        const v4i v0 = *reinterpret_cast<const v4i*>(
            smem_bytes + (v_buf + pv_v_lds_offset(seqk_local, hdim_pos)) * 2);
        const v4i v1 = *reinterpret_cast<const v4i*>(
            smem_bytes + (v_buf + pv_v_lds_offset(seqk_local, hdim_pos + 32)) * 2);

        // P(B) operand: per kstep, P regs at seqk = kstep*16 + k_sub*8 + {0..7}
        // These map to P registers kstep*8 + {0..7}, packed as 4 dwords.
        const v4i p = {p_raw[kstep * 4 + 0], p_raw[kstep * 4 + 1],
                       p_raw[kstep * 4 + 2], p_raw[kstep * 4 + 3]};

        // pass 0: seqk seqk_base+0..3
        {
            v4h a0 = pv_pack(v0[0], v0[1]);  // V from LDS = A operand
            v4h a1 = pv_pack(v1[0], v1[1]);
            v4h b  = pv_pack(p[0], p[1]);     // P from register = B operand
            o_d0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, o_d0, 0, 0, 0);
            o_d1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, o_d1, 0, 0, 0);
        }
        // pass 1: seqk seqk_base+4..7
        {
            v4h a0 = pv_pack(v0[2], v0[3]);
            v4h a1 = pv_pack(v1[2], v1[3]);
            v4h b  = pv_pack(p[2], p[3]);
            o_d0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a0, b, o_d0, 0, 0, 0);
            o_d1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a1, b, o_d1, 0, 0, 0);
        }
    }

    // ---- Dump O_acc in raw thread-buffer order ----
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        out[tid * 32 + r]      = o_d0[r];
        out[tid * 32 + 16 + r] = o_d1[r];
    }
}
