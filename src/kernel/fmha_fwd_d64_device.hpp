#pragma once
#include "runner/params.hpp"

typedef int v4i __attribute__((ext_vector_type(4)));

// Build a 128-bit buffer resource descriptor (SRD) for buffer_load/store.
// |base| is the already-offset pointer for this batch/head slice.
__device__ inline v4i make_buffer_resource(const void* base) {
    v4i srd;
    auto addr = reinterpret_cast<uintptr_t>(base);
    srd[0] = static_cast<int>(static_cast<uint32_t>(addr));
    srd[1] = static_cast<int>(static_cast<uint32_t>(addr >> 32));
    srd[2] = static_cast<int>(0xFFFFFFFF);  // num_records (unlimited)
    srd[3] = static_cast<int>(0x00027000);  // format, dst_sel defaults
    return srd;
}

// Q passthrough: load Q from DRAM, store verbatim to O.
//
// Thread mapping (MFMA B-input / "Q load" layout):
//   m_row   = warp_id * 32 + lane_id % 32
//   k_group = lane_id / 32          (0 or 1, selects which 32-column half)
//
// Each thread does 4 x buffer_load_dwordx4 = 16 DW = 32 bf16 elements,
// covering 8 bf16 contiguous along K per load.  The same addressing is
// reused for the O store so that the output is element-wise identical
// to the input.
template <bool HasMask, bool IsVarlen>
__device__ void fmha_fwd_d64_device(const FmhaFwdParams& params,
                                    int batch_idx,
                                    int head_idx,
                                    int m_tile_idx) {
    const int lane_id = threadIdx.x % kWarpSize;
    const int warp_id = threadIdx.x / kWarpSize;

    // Row within the M tile this thread services
    const int m_local = warp_id * 32 + (lane_id % 32);
    const int m_row   = m_tile_idx * kM0 + m_local;

    // Bounds check: skip if this thread's row is out of range
    if (m_row >= params.seqlen_q)
        return;

    // Column half (0 or 1): each half covers 32 contiguous bf16 along K
    const int k_group = lane_id / 32;

    // Compute batch+head base pointers for Q and O (bf16 element offsets)
    const __hip_bfloat16* q_base =
        params.q + static_cast<int64_t>(batch_idx) * params.batch_stride_q
                  + static_cast<int64_t>(head_idx)  * params.nhead_stride_q;
    __hip_bfloat16* o_base =
        params.o + static_cast<int64_t>(batch_idx) * params.batch_stride_o
                  + static_cast<int64_t>(head_idx)  * params.nhead_stride_o;

    // Build SRDs with batch+head baked into the base address
    v4i srd_q = make_buffer_resource(q_base);
    v4i srd_o = make_buffer_resource(o_base);

    // Per-thread byte offset: row * stride_bytes + column_group_bytes
    // stride_q/stride_o are in bf16 elements; * 2 for bytes
    const int row_byte_offset = m_row * (params.stride_q * 2);
    const int col_byte_offset = k_group * 64;  // 32 bf16 = 64 bytes
    const int voff_q = row_byte_offset + col_byte_offset;

    const int row_byte_offset_o = m_row * (params.stride_o * 2);
    const int voff_o = row_byte_offset_o + col_byte_offset;

    // Load 4 x dwordx4 from Q (each = 16 bytes = 8 bf16 along K)
    // Loads at byte offsets +0, +16, +32, +48 from thread's base
    v4i d0 = __builtin_amdgcn_raw_buffer_load_v4i32(srd_q, voff_q,      0, 0);
    v4i d1 = __builtin_amdgcn_raw_buffer_load_v4i32(srd_q, voff_q + 16, 0, 0);
    v4i d2 = __builtin_amdgcn_raw_buffer_load_v4i32(srd_q, voff_q + 32, 0, 0);
    v4i d3 = __builtin_amdgcn_raw_buffer_load_v4i32(srd_q, voff_q + 48, 0, 0);

    // Store 4 x dwordx4 to O (same layout)
    __builtin_amdgcn_raw_buffer_store_v4i32(d0, srd_o, voff_o,      0, 0);
    __builtin_amdgcn_raw_buffer_store_v4i32(d1, srd_o, voff_o + 16, 0, 0);
    __builtin_amdgcn_raw_buffer_store_v4i32(d2, srd_o, voff_o + 32, 0, 0);
    __builtin_amdgcn_raw_buffer_store_v4i32(d3, srd_o, voff_o + 48, 0, 0);
}
