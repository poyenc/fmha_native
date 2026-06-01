#include "runner/params.hpp"
#include "kernel/fmha_fwd_d64_device.hpp"

__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk0(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    const int m_tile_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<false, false>(params, lds, batch_idx, head_idx, m_tile_idx);
}

__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk1(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    // Causal load balance: tile i does i+1 KV-tiles, so launch the heavy
    // high-index tiles first (matches CK). Natural order strands all heavy
    // tiles in the final block-wave with no light work to overlap -> tail.
    const int m_tile_idx = gridDim.y - 1 - blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<true, false>(params, lds, batch_idx, head_idx, m_tile_idx);
}

__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk0_varlen(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    const int m_tile_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<false, true>(params, lds, batch_idx, head_idx, m_tile_idx);
}

__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk1_varlen(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    // Causal load balance: launch heavy high-index tiles first (see msk1).
    const int m_tile_idx = gridDim.y - 1 - blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<true, true>(params, lds, batch_idx, head_idx, m_tile_idx);
}
