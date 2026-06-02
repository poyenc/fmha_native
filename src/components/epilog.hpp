#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// =============================================================================
// COMPONENT (TEST-ONLY): epilogue — pipeline STAGE 7 of 7.
//
// Standalone isolation of the output epilogue, used ONLY by
// tests/test_epilog.cpp. Golden-verified, NOT #included by src/fused/.
// CPU oracle: src/components_ref/cpu_ref_epilog.{hpp,cpp}.
//
// WHAT it does: divide the raw GEMM1 accumulator O_acc by the softmax row-sum
// RSUM (the deferred normalization of flash attention), truncate to bf16, and
// store to DRAM. No LDS, no shuffle — straight register->DRAM.
//
// LAYOUT NOTE: O_acc is in GEMM1's SwizzleA'd TransposedC layout, so register r
// maps to DRAM column d_col = swz((r/8)*16 + k_sub*8 + (r%8)). The 32 regs are
// emitted as 8 buffer_store_dwordx2 of 4 contiguous bf16 each; the long inline
// comment below works the swz() mapping out explicitly for k_sub=0.
// =============================================================================
//
// Phase 1 Kernel 7 -- test_epilog (Default2D epilogue: normalize + bf16 store)
//
// Reproduces CK's Default2DEpilogue: O_acc / RSUM → bf16 → buffer_store.
// No shuffle, no LDS staging — direct register→DRAM.
//
// O_acc layout (TransposedC+SwizzleA):
//   m = (lane%32) + 32*warp
//   d = swz((r/8)*16 + (lane/32)*8 + (r%8))
//
// DRAM store: buffer_store_dwordx2 (2 dwords = 4 bf16 per store, 8 stores/thread).
// bf16 truncation (not RNE). Store at O[m_row * stride + d_col].
//
// Input: O_acc (256×32 fp32), RSUM (256 fp32), seqlen_q, stride_o.
// Output: O DRAM (bf16 row-major [seqlen_q, D=64], stored as fp32 bf16-promoted).

typedef int v2i __attribute__((ext_vector_type(2)));

__device__ __forceinline__ __amdgpu_buffer_rsrc_t
ep_make_srd(void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(base, 0, 0xFFFFFFFF, 0x00027000);
}

// SwizzleA: swap bits 2,3
__device__ __forceinline__ int ep_swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

// bf16 truncation: zero lower 16 bits
__device__ __forceinline__ uint16_t ep_f32_to_bf16(float v) {
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    return static_cast<uint16_t>(u >> 16);
}

// O_acc, rsum: device fp32 arrays. o_dram: output device buffer.
// o_final_dump: optional fp32 dump of normalized O (for golden slot 6 comparison).
__global__ void test_epilog_kernel(const float* __restrict__ o_acc,
                                   const float* __restrict__ rsum,
                                   uint16_t* __restrict__ o_dram,
                                   float* __restrict__ o_final_dump,
                                   int seqlen_q,
                                   int stride_o) {  // in bf16 elements
    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;
    const int warp_id = tid >> 6;
    const int k_sub   = lane_id >> 5;
    const int m_row   = (lane_id & 31) + 32 * warp_id;

    // Load RSUM and compute inverse
    float rs = rsum[tid];
    float inv_rs = (rs != 0.0f) ? (1.0f / rs) : 0.0f;

    // Normalize O_acc and optionally dump O_final
    float o_norm[32];
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        o_norm[r] = o_acc[tid * 32 + r] * inv_rs;
    }

    // Dump O_final (fp32 pre-store) if requested
    if (o_final_dump) {
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            o_final_dump[tid * 32 + r] = o_norm[r];
        }
    }

    // bf16 truncation + DRAM store
    if (m_row >= seqlen_q) return;

    __amdgpu_buffer_rsrc_t o_srd = ep_make_srd(o_dram);
    const int stride_bytes = stride_o * 2;
    const int row_byte_offset = m_row * stride_bytes;

    // 8 stores of buffer_store_dwordx2 (4 bf16 per store).
    // 32 bf16 values → 8 groups of 4.
    // Within each group of 4: consecutive d_col values.
    // The 32 registers map to d_col via SwizzleA'd TransposedC:
    //   r → d_col = swz((r/8)*16 + k_sub*8 + (r%8))
    //
    // Group the 32 regs into 8 stores of 4 consecutive bf16 each.
    // r = {0,1,2,3}, {4,5,6,7}, {8,...,11}, ..., {28,...,31}
    // d_cols for k_sub=0: swz({0,1,2,3}) = {0,1,2,3}
    //                     swz({4,5,6,7}) = {8,9,10,11}
    //                     swz({8,9,10,11}) = {16,...,19} (actually swz(16)=16, etc)
    // Wait: swz operates on d_nom, not r. d_nom = (r/8)*16 + k_sub*8 + (r%8).
    // For k_sub=0, r=0..3: d_nom = 0+0+{0,1,2,3} = {0,1,2,3}. swz = {0,1,2,3}. ✓
    // For k_sub=0, r=4..7: d_nom = 0+0+{4,5,6,7} = {4,5,6,7}. swz = {8,9,10,11}.
    // For k_sub=0, r=8..11: d_nom = 16+0+{0,1,2,3} = {16,17,18,19}. swz = {16,17,18,19}. ✓
    // For k_sub=0, r=12..15: d_nom = 16+0+{4,5,6,7} = {20,21,22,23}. swz = {24,25,26,27}.
    // Groups of 4 consecutive d_col: {0-3}, {8-11}, {16-19}, {24-27} for k_sub=0.
    // These are NOT contiguous in DRAM (gap of 4 between groups).
    // buffer_store_dwordx2 stores 2 dwords at a contiguous address.
    // So each store writes 4 bf16 at d_col, d_col+1, d_col+2, d_col+3.

    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        int r_base = g * 4;
        int d_nom_base = (r_base / 8) * 16 + k_sub * 8 + (r_base % 8);
        int d_col = ep_swz(d_nom_base);

        // Pack 4 bf16 into 2 dwords
        uint16_t b0 = ep_f32_to_bf16(o_norm[r_base + 0]);
        uint16_t b1 = ep_f32_to_bf16(o_norm[r_base + 1]);
        uint16_t b2 = ep_f32_to_bf16(o_norm[r_base + 2]);
        uint16_t b3 = ep_f32_to_bf16(o_norm[r_base + 3]);

        uint32_t dw0 = (static_cast<uint32_t>(b1) << 16) | b0;
        uint32_t dw1 = (static_cast<uint32_t>(b3) << 16) | b2;

        int voffset = row_byte_offset + d_col * 2;
        v2i store_data;
        store_data[0] = static_cast<int>(dw0);
        store_data[1] = static_cast<int>(dw1);
        __builtin_amdgcn_raw_buffer_store_b64(store_data, o_srd, voffset, 0, 0);
    }
}
