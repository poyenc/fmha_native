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

// ===== Split-K naive GPU oracles (Stage-1 trusted vs CPU at small S) ========
//
// These two kernels mirror, line-for-line, the CPU oracles cpu_ref_split (in
// runner/cpu_ref.cpp) and cpu_ref_combine (in components_ref/cpu_ref_combine).
// They exist because the CPU oracles are minutes-slow at S=40000; one fp32
// thread-per-row GPU pass makes large-S verification fast.  They are
// correctness-only oracles and are trusted ONLY because they match the CPU
// refs at small S (the Stage-1 trust test in tests/test_fmha_gpu_ref.cpp).
//
// They are PURELY ADDITIVE: gpu_ref_fmha_fwd / GpuRefParams / the existing
// kernel are untouched.  Both reuse GpuRefParams for the Q/K/V/strides/mask/
// scalar/varlen plumbing (d_O / d_LSE there are ignored — partials go to the
// dedicated fp32 buffers below).
// ============================================================================

// Naive GPU partial attention over KV range [kv_start, kv_end). One thread/row.
//
// Mirrors cpu_ref_split.  For each output row (flat thread id decoded as the
// existing gpu_ref kernel does: [batch, q_heads, seq] row order, hkv = hq/gqa,
// varlen via d_seqstart_q/k) it:
//   - scores S = (Q . K^T) * scalar over j in [kv_start, kv_end) clamped to
//     [0, Skv_b), with the SAME right-aligned causal mask (j > i+(Skv_b-Sq_b)),
//   - finds the local (range) max of the ALREADY-SCALED scores,
//   - softmaxes over the range (P truncated to bf16 then widened, matching the
//     production P.V GEMM precision and cpu_ref_split),
//   - writes fp32 O_g normalized by THIS range's local_sum, and
//   - writes the natural-log per-range LSE.
//
// LSE convention (★ the double-scale anti-bug): LSE_g = logf(local_sum) + local_max, where
// local_max is the ALREADY-SCALED row max.  There is NO extra `* scalar` here —
// the scale is baked into local_max up front (scores are scaled before the max).
// This matches gpu_ref.cpp:117 (`logf(sum_exp)+max_s`), cpu_ref_split, and
// op_epilog.hpp.  An empty / fully-masked range -> LSE_g = -INFINITY, O_g = 0.
//
// Output layout (the combine oracle reads this same layout):
//   d_o_g  : fp32 [total_rows * head_dim], total_rows = batch*q_heads*seq_len,
//            row index == the flat thread id (same [b,q_heads,seq] order as the
//            existing kernel decodes), inner dimension head_dim in natural order.
//            For a varlen row beyond this batch's true length nothing is written
//            (the thread early-exits, exactly like gpu_ref_fmha_kernel).
//   d_lse_g: fp32 [total_rows], one LSE per row, same row index as d_o_g.
void gpu_ref_split(const GpuRefParams& p, int kv_start, int kv_end,
                   float* d_o_g, float* d_lse_g, hipStream_t stream = nullptr);

// Naive GPU combine of G per-range partials -> final fp32 O. One thread/row,
// looping g = 0..G-1 and accumulating over head_dim.  Mirrors cpu_ref_combine
// (natural-e reweight by per-range LSE with the max-subtract stability trick;
// a -inf range contributes 0; all-inf -> zero output).
//
// Layout (must match gpu_ref_split's output above):
//   d_o_part : fp32 [G * rows * Dlog], SPLIT-MAJOR — plane g (offset
//              g*rows*Dlog) is the o_part produced by gpu_ref_split for range g.
//              Row r of plane g is at d_o_part[(g*rows + r)*Dlog + d].
//   d_lse_part: fp32 [G * rows], lse for (range g, row r) at d_lse_part[g*rows+r].
//   d_o_final : fp32 [rows * Dlog], combined output, row r at d_o_final[r*Dlog+d].
// rows is the number of output rows being combined; Dlog the logical head dim.
void gpu_ref_combine(int G, int rows, int Dlog,
                     const float* d_o_part, const float* d_lse_part,
                     float* d_o_final, hipStream_t stream = nullptr);
