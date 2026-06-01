// Kernel argument block (kernarg) for the fused FMHA forward shader.
//
// This is the single struct passed by value to every fused-kernel entry
// point (fmha_fwd_d64_bf16_msk{0,1}[_varlen] in src/fused/kernel.cpp).  The
// benchmark (src/bench_fmha_fwd.cpp) fills it from device pointers + strides
// computed in FmhaBuffers; the device-side reader is fmha_fwd_d64_device() in
// src/fused/pipeline.hpp.  It is shared by bench only — the tests drive the
// kernels through the same path but the CPU/GPU oracles use their own param
// structs (FmhaParams / GpuRefParams).
//
// Layout note: the field order here IS the kernarg layout the HSACO expects,
// so do not reorder fields without re-checking the kernel ABI.

#pragma once
#include <hip/hip_bf16.h>
#include <cstdint>

struct FmhaFwdParams {
    // Input tensors, row-major, BF16.  Logical layout per tensor is
    // [batch, nhead, seqlen, head_dim]; the actual element offset of any
    // (b, h, s, d) is b*batch_stride + h*nhead_stride + s*stride + d.
    const __hip_bfloat16 *q, *k, *v;
    // Output tensor O, same [batch, nhead_q, seqlen_q, head_dim] layout as Q.
    __hip_bfloat16* o;
    // Optional log-sum-exp output, one FP32 value per query row
    // ([batch, nhead_q, seqlen_q]).  nullptr disables LSE writes.
    float* lse;

    // Per-sequence lengths.  In batch mode these apply to every sequence; in
    // group/varlen mode they are upper bounds and the real per-batch lengths
    // come from seqstart_* (see below).
    int seqlen_q, seqlen_k;
    // Head counts.  nhead_q is the query head count; nhead_k is the KV head
    // count.  When nhead_q > nhead_k the kernel runs grouped-query attention
    // (each KV head is shared by nhead_q / nhead_k query heads).
    int nhead_q, nhead_k;

    // Softmax scale, PRE-MULTIPLIED by log2(e): scale == log2(e)/sqrt(head_dim),
    // NOT a plain 1/sqrt(head_dim).  The kernel's softmax is base-2 (it uses
    // exp2, not exp), so folding log2(e) into the scale converts the natural-e
    // softmax into the equivalent base-2 form.  See bench_fmha_fwd.cpp
    // (kLog2e/sqrtf(head_dim)) and op_softmax.hpp.
    float scale;

    // All strides below are in ELEMENTS (BF16 units), not bytes.  For each
    // tensor: stride_* is the per-token (seqlen) stride, nhead_stride_* is the
    // per-head stride, batch_stride_* is the per-batch stride.  Contiguous
    // packing makes stride == head_dim, nhead_stride == seqlen*head_dim, etc.
    int stride_q, nhead_stride_q, batch_stride_q;
    int stride_k, nhead_stride_k, batch_stride_k;
    int stride_v, nhead_stride_v, batch_stride_v;
    int stride_o, nhead_stride_o, batch_stride_o;

    // Group (variable-length) mode: cumulative token-offset tables of length
    // batch+1, so the b-th sequence spans tokens [seqstart[b], seqstart[b+1]).
    // When non-null the kernel ignores batch_stride_* (sequences are packed
    // back-to-back) and derives each per-batch length from the table.  Both
    // nullptr selects fixed-length batch mode.
    //
    // Note: the causal "mask_shift" (seqlen_k - seqlen_q) is NOT stored here;
    // the kernel computes it on the fly in pipeline.hpp.
    const int32_t* seqstart_q;
    const int32_t* seqstart_k;
};

// --- Compile-time tile / launch geometry (D64 BF16 kernel specific) ---
// These describe how the fused kernel partitions the problem and lays out LDS.
// The benchmark also reads kM0 (M-tile size) and kBlockSize to build the grid.
// Tile constants (D64 bf16 specific)
constexpr int kM0 = 128;          // query rows per M-tile (one threadblock's work in Q)
constexpr int kN0 = 64;           // key columns per K-tile (GEMM0 inner N)
constexpr int kK0 = 32;           // contraction depth per step of GEMM0 (Q.K^T)
constexpr int kN1 = 64;           // output columns per tile of GEMM1 (= head_dim)
constexpr int kK1 = 32;           // contraction depth per step of GEMM1 (P.V)
constexpr int kBlockSize = 256;   // threads per block (= kNumWarps * kWarpSize)
constexpr int kNumWarps = 4;      // warps (waves) per block
constexpr int kWarpSize = 64;     // lanes per wavefront (CDNA: 64, not 32)
constexpr int kHeadDim = 64;      // head dimension D this kernel is specialized for
constexpr int kKPack = 8;         // BF16 elements packed per vectorized LDS access
// LDS rows are padded to kPixelsPerRow + kKPack to avoid bank conflicts.
constexpr int kPixelsPerRow = 64;
constexpr int kPaddedRowStride = kPixelsPerRow + kKPack; // 72 elements
constexpr int kSingleSmemElements = 2304; // per LDS buffer, in bf16 elements
constexpr int kNumLdsBuffers = 3;         // triple-buffered Q/K/V staging in LDS
// Total LDS footprint in bytes (3 * 2304 * 2 = 13824).
constexpr int kLdsBytes = kNumLdsBuffers * kSingleSmemElements * sizeof(__hip_bfloat16); // 13824
