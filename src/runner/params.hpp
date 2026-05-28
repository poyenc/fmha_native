#pragma once
#include <hip/hip_bf16.h>
#include <cstdint>

struct FmhaFwdParams {
    const __hip_bfloat16 *q, *k, *v;
    __hip_bfloat16* o;
    float* lse;                     // nullptr if not storing LSE
    int seqlen_q, seqlen_k;
    int nhead_q, nhead_k;
    float scale;                    // pre-multiplied by log2(e)
    // Strides in elements
    int stride_q, nhead_stride_q, batch_stride_q;
    int stride_k, nhead_stride_k, batch_stride_k;
    int stride_v, nhead_stride_v, batch_stride_v;
    int stride_o, nhead_stride_o, batch_stride_o;
    // Group mode (nullptr for batch mode)
    const int32_t* seqstart_q;
    const int32_t* seqstart_k;
};

// Tile constants (D64 bf16 specific)
constexpr int kM0 = 128;
constexpr int kN0 = 64;
constexpr int kK0 = 32;
constexpr int kN1 = 64;
constexpr int kK1 = 32;
constexpr int kBlockSize = 256;
constexpr int kNumWarps = 4;
constexpr int kWarpSize = 64;
constexpr int kHeadDim = 64;
constexpr int kKPack = 8;
constexpr int kPixelsPerRow = 64;
constexpr int kPaddedRowStride = kPixelsPerRow + kKPack; // 72 elements
constexpr int kSingleSmemElements = 2304; // per buffer, in bf16 elements
constexpr int kNumLdsBuffers = 3;
constexpr int kLdsBytes = kNumLdsBuffers * kSingleSmemElements * sizeof(__hip_bfloat16); // 13824
