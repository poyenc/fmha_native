#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// =============================================================================
// COMPONENT (TEST-ONLY): V-LDS staging — pipeline STAGE 5 of 7.
//
// Standalone isolation of the "stage V into LDS" step, used ONLY by
// tests/test_v_lds.cpp. Golden-verified, NOT #included by src/fused/.
// CPU oracle: src/components_ref/cpu_ref_v_lds.{hpp,cpp}.
//
// WHY V needs a shuffle but K does not: GEMM1 contracts over seqk, so V must be
// laid out in LDS with headdim as the row axis and seqk innermost. The DRAM
// layout is the opposite, so each thread loads a 4x2 bf16 patch, transposes it
// to 2x4 with two v_perm_b32 (a byte-permute), then ds_write2 the result into
// the padded LDS slots. Unlike K (loaded via direct DRAM->LDS), V passes
// through VGPRs because of this transpose.
//
// SCOPE: this component stages just ONE 32-row k1 slice (golden V_LDS dump
// captures exactly that). The 72-element row stride in the offset formula is
// 64 real headdim + 8 pad (kKPack) to avoid LDS bank conflicts.
// =============================================================================
//
// Phase 1 Kernel 5 -- test_v_lds (V DRAM → shuffle → LDS staging)
//
// Reproduces CK's V load+repack+store path for ONE 32-row k1 slice,
// matching golden dump_lds slot 1 (V_LDS). The 3-stage pipeline:
//
// 1. LOAD: buffer_load_dwordx2 (4 bf16/load) from V DRAM into registers.
//    Thread mapping: n_hdim = lane_id/4 (hdim group 0..15),
//                    k_within = lane_id%4 (seqk sub-position 0..3).
//    Warp W loads rows [kv_offset + W*8 .. + (W+1)*8 - 1].
//    Per k3 (0,1): v_row = kv_offset + warp*8 + k_within*2 + k3.
//
// 2. SHUFFLE: v_perm_b32 with selectors 0x01000504, 0x03020706.
//    Transposes 2×2 bf16 within each dword pair (4×2 → 2×4 per thread).
//
// 3. STORE: 2 × ds_write2_b32 per thread (4 dwords total) into LDS.
//    LDS address: warp*1152 + ((lane>>3)&7)*144 + ((lane>>2)&1)*64 + (lane&3)*4
//    ds_write2_b32 offsets: {128,132} and {136,140} (dword units).
//
// Golden-verified V LDS formula (Phase 0.10):
//   k = n % 32; offset_elems(n,d) = buf_base + (k/8)*576 + (d/8)*72 + (d%8)*8 + (k%8)
//   buf_base = 2304. One 32-row slice per golden dump.

typedef int v2i __attribute__((ext_vector_type(2)));

#include "components_ref/cpu_ref_v_lds.hpp"  // kVLdsRegionElems, kVLdsBufBase

__device__ __forceinline__ __amdgpu_buffer_rsrc_t
vlds_make_srd(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

// V: row-major [seqlen_k, D=64] bf16. stride_v in bf16 elements.
// out: device buffer >= kVLdsRegionElems bf16 elements (dumped V LDS region).
__global__ void test_v_lds_kernel(const uint16_t* __restrict__ V,
                                  uint16_t* __restrict__ out,
                                  int stride_v,
                                  int kv_offset,
                                  int seqlen_k) {
    __shared__ __attribute__((aligned(16))) uint16_t smem[kVLdsRegionElems];

    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;
    const int warp_id = tid >> 6;

    // Pre-zero so OOB/padding slots are deterministic 0
    for (int e = tid; e < kVLdsRegionElems; e += blockDim.x) smem[e] = 0;
    __syncthreads();

    // ---- V DRAM load (buffer_load_dwordx2) ----
    __amdgpu_buffer_rsrc_t v_srd = vlds_make_srd(V);
    int n_hdim   = lane_id >> 2;     // 0..15 (hdim group)
    int k_within = lane_id & 3;      // 0..3 (seqk sub-position)
    int stride_bytes = stride_v * 2;

    int v_row_base = kv_offset + warp_id * 8 + k_within * 2;
    int v_col_bytes = (n_hdim * 4) * 2;

    // Load k3=0 and k3=1 (2 loads per thread = 4 dwords = 8 bf16)
    v2i load_k3_0, load_k3_1;
    int row0 = v_row_base;
    int row1 = v_row_base + 1;
    if (row0 < seqlen_k) {
        load_k3_0 = __builtin_amdgcn_raw_buffer_load_b64(
            v_srd, row0 * stride_bytes + v_col_bytes, 0, 0);
    } else {
        load_k3_0 = v2i{0, 0};
    }
    if (row1 < seqlen_k) {
        load_k3_1 = __builtin_amdgcn_raw_buffer_load_b64(
            v_srd, row1 * stride_bytes + v_col_bytes, 0, 0);
    } else {
        load_k3_1 = v2i{0, 0};
    }

    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // ---- v_perm_b32 repack (4×2 → 2×4 transpose) ----
    constexpr unsigned kPermSel0 = 0x01000504;  // {src1_lo, src0_lo}
    constexpr unsigned kPermSel1 = 0x03020706;  // {src1_hi, src0_hi}

    unsigned in0_lo = static_cast<unsigned>(load_k3_0[0]);
    unsigned in0_hi = static_cast<unsigned>(load_k3_0[1]);
    unsigned in1_lo = static_cast<unsigned>(load_k3_1[0]);
    unsigned in1_hi = static_cast<unsigned>(load_k3_1[1]);

    unsigned out0 = __builtin_amdgcn_perm(in0_lo, in1_lo, kPermSel0);
    unsigned out1 = __builtin_amdgcn_perm(in0_lo, in1_lo, kPermSel1);
    unsigned out2 = __builtin_amdgcn_perm(in0_hi, in1_hi, kPermSel0);
    unsigned out3 = __builtin_amdgcn_perm(in0_hi, in1_hi, kPermSel1);

    // ---- ds_write2_b32 to LDS ----
    // The ds_write2_b32 offsets (128..140 dwords = 512..560 bytes) are baked
    // into the instruction. Subtract 512 from the target buffer byte address
    // so that base + 128*4 = kVLdsBufBase * 2 (= 4608 bytes).
    const int lds_buf_byte_offset = kVLdsBufBase * 2 - 128 * 4;

    int v86 = ((lane_id >> 3) & 7) * 144
            + ((lane_id >> 2) & 1) * 64
            + (lane_id & 3) * 4;
    int lds_addr = warp_id * 1152 + v86 + lds_buf_byte_offset;

    asm volatile("ds_write2_b32 %0, %1, %2 offset0:128 offset1:132"
                 : : "v"(lds_addr), "v"(out0), "v"(out1) : "memory");
    asm volatile("ds_write2_b32 %0, %1, %2 offset0:136 offset1:140"
                 : : "v"(lds_addr), "v"(out2), "v"(out3) : "memory");

    __syncthreads();

    // Cooperative dump of the V LDS region to DRAM
    for (int e = tid; e < kVLdsRegionElems; e += blockDim.x) out[e] = smem[e];
}
