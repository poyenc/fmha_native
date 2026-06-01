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
    const int m_tile_idx = blockIdx.y;
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
    const int m_tile_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<true, true>(params, lds, batch_idx, head_idx, m_tile_idx);
}
