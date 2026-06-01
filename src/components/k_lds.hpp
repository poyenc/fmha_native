#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Phase 1 Kernel 1 — test_k_lds
//
// Stages a 64x64 K tile from DRAM into LDS using CK's GOLDEN-VERIFIED async
// copy layout (buffer_load_dword ... lds), then cooperatively dumps the LDS
// K region to DRAM for byte-exact comparison.
//
// VERIFIED layout (impl-checked 0/4096 vs golden):
//   offset_elems(j,d) = (j%4)*136 + ((j/4)%4)*32 + (j/16)*544
//                     + (d%32)   + (d/32)*2304
//
// Hardware auto-distribution: for warp w, issue i, lane L the data element is
//   j  = i*16 + (L>>4)*4 + w        (seqlen row in tile)
//   d0 = chunk*32 + (L&15)*2        (2 contiguous bf16 along headdim)
// and the HW writes the loaded dword to LDS byte (m0 + L*4), where
//   m0 = chunk*4608 + w*0x110 + i*0x440   (bytes; 0x110=136 elems, 0x440=544).
// One can verify  m0 + L*4 == 2*offset_elems(j,d0)  identically.
//
// NOTE (impl): the existing src/kernel/fmha_fwd_d64_lds.hpp::async_copy_k_slice
// uses n_base = warp*4 + (lane>>4), which swaps the warp/lane-group factors and
// reproduces only 1024/4096 slots correctly. This kernel uses the CORRECT
// n_base = (lane>>4)*4 + warp. The native kernel is left untouched until Phase 2.

constexpr int kKLdsRegionFloats = 8192; // golden dump per-slot stride (bf16 elems)

using lds_ptr_t = __attribute__((address_space(3))) void*;

__device__ __forceinline__ __amdgpu_buffer_rsrc_t
k_lds_make_srd(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(base), 0, 0xFFFFFFFF, 0x00027000);
}

// One async DRAM->LDS dword copy (lowers to `buffer_load_dword ... offen lds`).
// `lds_dst` is the wave-uniform LDS base (becomes m0); the hardware writes each
// active lane's loaded dword to (m0 + lane*4) automatically. `voffset` is the
// per-lane DRAM byte offset.
__device__ __forceinline__ void
k_lds_async_load(__amdgpu_buffer_rsrc_t k_srd, lds_ptr_t lds_dst, int voffset) {
    __builtin_amdgcn_raw_ptr_buffer_load_lds(
        k_srd, lds_dst, /*size=*/4, voffset, /*soffset=*/0, /*offset=*/0, /*aux=*/0);
}

// K: row-major [seqlen_k, D=64] bf16. stride_k in bf16 elements.
// out: device buffer >= kKLdsRegionFloats bf16 elements (the dumped K region).
__global__ void test_k_lds_kernel(const uint16_t* __restrict__ K,
                                  uint16_t* __restrict__ out,
                                  int stride_k,
                                  int kv_offset,
                                  int seqlen_k) {
    __shared__ __attribute__((aligned(16))) uint16_t smem[kKLdsRegionFloats];

    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;
    const int warp_id = tid >> 6;

    // Pre-zero the whole K region so OOB rows / pad slots are deterministic 0
    // (CK zero-fills OOB rows; this matches the partial-tile golden).
    for (int e = tid; e < kKLdsRegionFloats; e += blockDim.x) smem[e] = 0;
    __syncthreads();

    __amdgpu_buffer_rsrc_t k_srd = k_lds_make_srd(K);
    const int stride_bytes = stride_k * 2;
    char* smem_bytes = reinterpret_cast<char*>(smem);

    const int d_in_chunk = (lane_id & 15) * 2;      // 0,2,...,30
    const int n_base     = (lane_id >> 4) * 4 + warp_id;  // CORRECT factor order

    // 2 hdim chunks x 4 issues. The LDS destination base per (chunk,warp,issue)
    // is the wave-uniform m0; the HW adds lane*4. Active lanes (row in range)
    // issue one dword load; OOB rows are skipped and stay pre-zeroed.
    #pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int k_col_offset = chunk * 32;
        const int chunk_base_bytes = chunk * 4608; // 2304 elems
        #pragma unroll
        for (int issue = 0; issue < 4; ++issue) {
            const int m0_bytes = chunk_base_bytes + warp_id * 0x110 + issue * 0x440;
            lds_ptr_t lds_dst =
                (lds_ptr_t)(smem_bytes + m0_bytes);

            const int j   = issue * 16 + n_base;
            const int row = kv_offset + j;
            if (row < seqlen_k) {
                const int voffset =
                    row * stride_bytes + (k_col_offset + d_in_chunk) * 2;
                k_lds_async_load(k_srd, lds_dst, voffset);
            }
        }
    }

    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    // Cooperative dump of the K region to DRAM.
    for (int e = tid; e < kKLdsRegionFloats; e += blockDim.x) out[e] = smem[e];
}
