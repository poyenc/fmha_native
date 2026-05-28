// Naive FP32 GPU reference kernel for FMHA forward pass.
//
// Each thread computes one output row (one query position) using a
// straightforward 3-pass algorithm: find max, compute softmax, and
// accumulate P*V.  BF16 inputs are widened to FP32 for arithmetic;
// the softmax P values are truncated back to BF16 to match the
// precision pipeline of the optimized HSACO shader.
//
// The kernel supports arbitrary per-dimension strides and variable-
// length batching via cumulative offset tables, matching the HSACO
// shader's addressing model.
//
// This is intentionally slow (one thread per row, no tiling, no shared
// memory) — it exists purely for correctness testing.

#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Per-dimension stride-based parameter block for the GPU reference kernel.
// Strides are in BF16 elements (not bytes).
struct GpuRefParams {
    const uint16_t* d_Q;
    const uint16_t* d_K;
    const uint16_t* d_V;
    uint16_t* d_O;
    float* d_LSE;          // nullable — only written when LSE is enabled

    int batch, q_heads, kv_heads, gqa;
    int seq_len, kv_seq_len, head_dim;

    // Element strides (not bytes) for each tensor dimension
    int stride_q_seq, stride_q_head, stride_q_batch;
    int stride_k_seq, stride_k_head, stride_k_batch;
    int stride_v_seq, stride_v_head, stride_v_batch;
    int stride_o_seq, stride_o_head, stride_o_batch;

    bool mask;             // causal mask (right-aligned)
    float scalar;          // 1/sqrt(head_dim)

    // Varlen: cumulative sequence offset tables (device ptrs, null for fixed-length)
    const uint32_t* d_seqstart_q;
    const uint32_t* d_seqstart_k;
};

// Launch the naive GPU reference kernel.
// Processes batch * q_heads * seq_len rows, one thread per row.
void gpu_ref_fmha_fwd(const GpuRefParams& p, hipStream_t stream = nullptr);
