#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"

typedef int v4i __attribute__((ext_vector_type(4)));

// ---- waitcnt / barrier helpers ----

__device__ inline void s_waitcnt_vmcnt_0() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

__device__ inline void s_waitcnt_lgkmcnt_0() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

__device__ inline void s_waitcnt_lgkmcnt_1() {
    asm volatile("s_waitcnt lgkmcnt(1)" ::: "memory");
}

__device__ inline void s_barrier() {
    asm volatile("s_barrier" ::: "memory");
}

// ---- ds_bpermute: read another lane's VGPR ----

__device__ inline float bpermute_f32(int src_lane, float val) {
    int src_byte = src_lane * 4;
    int ret;
    asm volatile("ds_bpermute_b32 %0, %1, %2\n"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(ret) : "v"(src_byte), "v"(val) : "memory");
    return __builtin_bit_cast(float, ret);
}

// ---- LDS K layout ----
//
// K tile (64 N x 32 K bf16) stored in padded layout:
//   shape:   (4, 8, 8, 8)  [kKPerBlock/kKPack, kNPerBlock/NPerRow, NPerRow, kKPack]
//   strides: (576, 72, 8, 1) [bf16 elements]
//
// Total per buffer: 2304 bf16 = 4608 bytes.

__device__ inline int k_lds_offset(int n, int k) {
    return (k >> 3) * 576 + (n >> 3) * 72 + (n & 7) * 8 + (k & 7);
}

// ---- K copy to LDS via VGPR staging ----
//
// Since gfx942 doesn't support buffer_load_dword with lds flag,
// we load K data to VGPRs first, then store to LDS.
//
// Thread mapping (256 threads load 64 N x 32 K bf16):
//   Each thread loads 4 DW = 8 bf16 (one ds_write_b128 worth).
//   256 threads x 8 bf16 = 2048 bf16. The tile has 2048 data elements
//   (64 x 32), so each thread handles exactly 8 elements.
//
// Thread-to-tile mapping:
//   thread_id = warp_id * 64 + lane_id  (0..255)
//   Each thread handles one contiguous block of 8 K-dimension elements
//   at a specific N position.
//
//   n_pos = thread_id / 4    (0..63, since 256/4 = 64 N positions)
//   k_base = (thread_id % 4) * 8  (0, 8, 16, 24 within the 32 K range)
//
// DRAM address: K[n_pos, k0_col_offset + k_base : +8]
// LDS address: k_lds_offset(n_pos, k_base) * 2 + lds_buf_offset

// Copy two k0 slices (64N x 32K bf16 each) of K to LDS.
// Loads for both slices are issued before waiting, to maximize overlap.
__device__ inline void copy_k_to_lds_2x(__amdgpu_buffer_rsrc_t k_srd,
                                          int stride_k,
                                          char* lds,
                                          int lds_buf0_offset,
                                          int lds_buf1_offset) {
    int tid = threadIdx.x;
    int n_pos = tid >> 2;       // 0..63
    int k_group = tid & 3;     // 0..3
    int k_base = k_group * 8;  // 0, 8, 16, 24
    int stride_bytes = stride_k * 2;

    // Issue both DRAM loads before waiting
    int dram_off_k0 = n_pos * stride_bytes + k_base * 2;
    v4i data_k0 = __builtin_amdgcn_raw_buffer_load_b128(k_srd, dram_off_k0, 0, 0);

    int dram_off_k1 = n_pos * stride_bytes + (32 + k_base) * 2;
    v4i data_k1 = __builtin_amdgcn_raw_buffer_load_b128(k_srd, dram_off_k1, 0, 0);

    // Wait for DRAM loads (includes any prior Q loads too)
    s_waitcnt_vmcnt_0();

    // Compute LDS byte offsets (same within-buffer pattern for both slices)
    int lds_elem_off = k_lds_offset(n_pos, k_base);
    int lds_byte_off_k0 = lds_buf0_offset + lds_elem_off * 2;
    int lds_byte_off_k1 = lds_buf1_offset + lds_elem_off * 2;

    // Store to LDS
    *reinterpret_cast<v4i*>(lds + lds_byte_off_k0) = data_k0;
    *reinterpret_cast<v4i*>(lds + lds_byte_off_k1) = data_k1;
}

// Copy two k0 slices with seqlen_k bounds check.
// Rows beyond seqlen_k are zero-filled.
__device__ inline void copy_k_to_lds_2x_guarded(__amdgpu_buffer_rsrc_t k_srd,
                                                  int stride_k,
                                                  char* lds,
                                                  int lds_buf0_offset,
                                                  int lds_buf1_offset,
                                                  int kv_offset,
                                                  int seqlen_k) {
    int tid = threadIdx.x;
    int n_pos = tid >> 2;       // 0..63
    int k_group = tid & 3;     // 0..3
    int k_base = k_group * 8;  // 0, 8, 16, 24
    int stride_bytes = stride_k * 2;

    int row = kv_offset + n_pos;

    v4i data_k0, data_k1;
    if (row < seqlen_k) {
        int dram_off_k0 = row * stride_bytes + k_base * 2;
        data_k0 = __builtin_amdgcn_raw_buffer_load_b128(k_srd, dram_off_k0, 0, 0);

        int dram_off_k1 = row * stride_bytes + (32 + k_base) * 2;
        data_k1 = __builtin_amdgcn_raw_buffer_load_b128(k_srd, dram_off_k1, 0, 0);
    } else {
        data_k0 = v4i{0, 0, 0, 0};
        data_k1 = v4i{0, 0, 0, 0};
    }

    s_waitcnt_vmcnt_0();

    int lds_elem_off = k_lds_offset(n_pos, k_base);
    int lds_byte_off_k0 = lds_buf0_offset + lds_elem_off * 2;
    int lds_byte_off_k1 = lds_buf1_offset + lds_elem_off * 2;

    *reinterpret_cast<v4i*>(lds + lds_byte_off_k0) = data_k0;
    *reinterpret_cast<v4i*>(lds + lds_byte_off_k1) = data_k1;
}

// Copy a single k0 slice of K (64N x 32K bf16) to one LDS buffer.
// kv_offset is the row offset into seqlen_k for the current tile.
__device__ inline void copy_k_slice_to_lds(__amdgpu_buffer_rsrc_t k_srd,
                                            int stride_k,
                                            char* lds,
                                            int lds_buf_offset,
                                            int kv_offset,
                                            int k_col_offset,
                                            int seqlen_k) {
    int tid = threadIdx.x;
    int n_pos = tid >> 2;       // 0..63
    int k_group = tid & 3;     // 0..3
    int k_base = k_group * 8;  // 0, 8, 16, 24
    int stride_bytes = stride_k * 2;

    int row = kv_offset + n_pos;
    int dram_off = row * stride_bytes + (k_col_offset + k_base) * 2;

    v4i data;
    if (row < seqlen_k) {
        data = __builtin_amdgcn_raw_buffer_load_b128(k_srd, dram_off, 0, 0);
    } else {
        data = v4i{0, 0, 0, 0};
    }

    s_waitcnt_vmcnt_0();

    int lds_elem_off = k_lds_offset(n_pos, k_base);
    int lds_byte_off = lds_buf_offset + lds_elem_off * 2;
    *reinterpret_cast<v4i*>(lds + lds_byte_off) = data;
}

// ---- V copy to LDS ----
//
// V tile (64 N x 64 D bf16) is processed as two k1 slices (64 x 32 each).
// Each slice uses the same padded LDS layout as K.
// V is loaded via buffer_load_b128 (same thread mapping as K copy).

__device__ inline void copy_v_to_lds_2x(__amdgpu_buffer_rsrc_t v_srd,
                                          int stride_v,
                                          char* lds,
                                          int lds_buf0_offset,
                                          int lds_buf1_offset,
                                          int kv_offset,
                                          int seqlen_k) {
    int tid = threadIdx.x;
    int n_pos = tid >> 2;       // 0..63
    int k_group = tid & 3;     // 0..3
    int k_base = k_group * 8;  // 0, 8, 16, 24
    int stride_bytes = stride_v * 2;

    int row = kv_offset + n_pos;

    // Issue both DRAM loads before waiting
    v4i data_k0, data_k1;
    if (row < seqlen_k) {
        int dram_off_k0 = row * stride_bytes + k_base * 2;
        data_k0 = __builtin_amdgcn_raw_buffer_load_b128(v_srd, dram_off_k0, 0, 0);

        int dram_off_k1 = row * stride_bytes + (32 + k_base) * 2;
        data_k1 = __builtin_amdgcn_raw_buffer_load_b128(v_srd, dram_off_k1, 0, 0);
    } else {
        data_k0 = v4i{0, 0, 0, 0};
        data_k1 = v4i{0, 0, 0, 0};
    }

    s_waitcnt_vmcnt_0();

    int lds_elem_off = k_lds_offset(n_pos, k_base);
    int lds_byte_off_k0 = lds_buf0_offset + lds_elem_off * 2;
    int lds_byte_off_k1 = lds_buf1_offset + lds_elem_off * 2;

    *reinterpret_cast<v4i*>(lds + lds_byte_off_k0) = data_k0;
    *reinterpret_cast<v4i*>(lds + lds_byte_off_k1) = data_k1;
}
