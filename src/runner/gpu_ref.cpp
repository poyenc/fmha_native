// Naive FP32 GPU reference kernel — see gpu_ref.hpp for API documentation.

#include "gpu_ref.hpp"
#include "bf16_utils.hpp"
#include <hip/hip_runtime.h>
#include <cmath>

// One thread = one output row (one query position).  The grid is FLAT 1D:
// total_rows threads laid out as [batch, q_heads, seq_len], unlike the fused
// kernel's 3D grid (q_heads, m_tiles, batch) which assigns a 128-row M-tile to
// each threadblock.  This naive form trades all of that tiling for readability.
__global__ void gpu_ref_fmha_kernel(GpuRefParams p) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rows = p.batch * p.q_heads * p.seq_len;
    if (tid >= total_rows) return;

    // Decode the flat thread id into (batch, query-head, query-row).
    const int b  = tid / (p.q_heads * p.seq_len);
    const int hq = (tid / p.seq_len) % p.q_heads;
    const int i  = tid % p.seq_len;
    // GQA: several query heads share one KV head.
    const int hkv = hq / p.gqa;

    const int D = p.head_dim;

    // Default (fixed-length batch) addressing: query row i within this batch's
    // full sequence; keys start at 0.  Overridden below in varlen mode.
    int i_q = i;
    int i_k = 0;
    int Sq_b = p.seq_len;
    int Skv_b = p.kv_seq_len;

    // Varlen/group mode: the cumulative offset tables give this batch's slice
    // [seqstart[b], seqstart[b+1]) inside the packed Q/K/V tensors.  Threads
    // whose row index exceeds this batch's true length exit early.
    if (p.d_seqstart_q) {
        uint32_t qstart = p.d_seqstart_q[b];
        Sq_b = p.d_seqstart_q[b + 1] - qstart;
        if (i >= Sq_b) return;
        i_q = qstart + i;

        uint32_t kstart = p.d_seqstart_k[b];
        Skv_b = p.d_seqstart_k[b + 1] - kstart;
        i_k = kstart;
    }

    const uint16_t* Q_row  = p.d_Q + (size_t)b * p.stride_q_batch + (size_t)hq  * p.stride_q_head + (size_t)i_q * p.stride_q_seq;
    const uint16_t* K_base = p.d_K + (size_t)b * p.stride_k_batch + (size_t)hkv * p.stride_k_head + (size_t)i_k * p.stride_k_seq;
    const uint16_t* V_base = p.d_V + (size_t)b * p.stride_v_batch + (size_t)hkv * p.stride_v_head + (size_t)i_k * p.stride_v_seq;
    uint16_t*       O_row  = p.d_O + (size_t)b * p.stride_o_batch + (size_t)hq  * p.stride_o_head + (size_t)i_q * p.stride_o_seq;

    // Pass 1: scan all keys to find the row max (for stable softmax).
    float max_s = -INFINITY;
    for (int j = 0; j < Skv_b; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        if (s > max_s) max_s = s;
    }

    // Pass 2: re-scan to accumulate the softmax denominator sum(exp(s - max)).
    // Scores are recomputed rather than cached to keep per-thread state tiny.
    float sum_exp = 0;
    for (int j = 0; j < Skv_b; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        float e = (s == -INFINITY) ? 0.0f : expf(s - max_s);
        sum_exp += e;
    }

    float inv_sum = (sum_exp > 0) ? 1.0f / sum_exp : 0.0f;

    // O_acc holds this row's output; D <= 256 is guaranteed by the dispatch.
    float O_acc[256];
    for (int d = 0; d < D; d++) O_acc[d] = 0;

    // Pass 3: recompute probabilities and accumulate O += P * V.
    for (int j = 0; j < Skv_b; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        float p_val = ((s == -INFINITY) ? 0.0f : expf(s - max_s)) * inv_sum;

        // Truncate P to BF16 and widen back, matching the optimized kernel's
        // P.V GEMM input precision (and the CPU reference).
        uint16_t p_bf16 = float_to_bf16(p_val);
        float p_trunc = bf16_to_float(p_bf16);

        for (int d = 0; d < D; d++) {
            float v = bf16_to_float(V_base[(size_t)j * p.stride_v_seq + d]);
            O_acc[d] += p_trunc * v;
        }
    }

    for (int d = 0; d < D; d++) {
        O_row[d] = float_to_bf16(O_acc[d]);
    }

    // Optional log-sum-exp: log(sum exp(s - max)) + max recovers log(sum exp(s)).
    if (p.d_LSE) {
        float lse_val = (sum_exp > 0) ? logf(sum_exp) + max_s : -INFINITY;
        // LSE is packed [batch, q_heads, seqlen] with no head_dim; recover the
        // per-row strides by dividing the Q element strides by the per-token
        // stride (Q's stride_q_seq == head_dim for contiguous tensors).
        size_t lse_idx = (size_t)b * (p.stride_q_batch / p.stride_q_seq)
                       + (size_t)hq * (p.stride_q_head / p.stride_q_seq)
                       + i_q;
        p.d_LSE[lse_idx] = lse_val;
    }
}

void gpu_ref_fmha_fwd(const GpuRefParams& p, hipStream_t stream) {
    const int total_rows = p.batch * p.q_heads * p.seq_len;
    const int block = 256;
    const int grid = (total_rows + block - 1) / block;
    hipLaunchKernelGGL(gpu_ref_fmha_kernel, dim3(grid), dim3(block), 0, stream, p);
}

// =============================================================================
// Split-K naive GPU oracles.  See gpu_ref.hpp for the API contract and layouts.
// Both are line-for-line ports of the CPU oracles (cpu_ref_split /
// cpu_ref_combine); only the KV-range bounds and the fp32 partial outputs differ
// from gpu_ref_fmha_kernel above.
// =============================================================================

// One thread = one output row.  Partial attention over KV range
// [kv_start, kv_end) (clamped to this row's valid [0, Skv_b)).  Mirrors
// gpu_ref_fmha_kernel's structure exactly, except: (a) j loops over the clamped
// sub-range, (b) the output O_g is fp32 normalized by THIS range's local sum,
// and (c) the LSE is always written (per-range) in natural-log form.
__global__ void gpu_ref_split_kernel(GpuRefParams p, int kv_start, int kv_end,
                                     float* d_o_g, float* d_lse_g) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rows = p.batch * p.q_heads * p.seq_len;
    if (tid >= total_rows) return;

    // Decode the flat thread id into (batch, query-head, query-row) — identical
    // to the existing kernel, so d_o_g/d_lse_g use the same flat row order.
    const int b  = tid / (p.q_heads * p.seq_len);
    const int hq = (tid / p.seq_len) % p.q_heads;
    const int i  = tid % p.seq_len;
    // GQA: several query heads share one KV head.
    const int hkv = hq / p.gqa;

    const int D = p.head_dim;

    // Default (fixed-length batch) addressing; overridden below in varlen mode.
    int i_q = i;
    int i_k = 0;
    int Sq_b = p.seq_len;
    int Skv_b = p.kv_seq_len;

    // Varlen/group mode: cumulative offset tables give this batch's slice.
    // Threads whose row index exceeds this batch's true length exit early —
    // exactly like gpu_ref_fmha_kernel, so those rows in d_o_g are left as the
    // caller initialized them (the caller zeroes the buffer).
    if (p.d_seqstart_q) {
        uint32_t qstart = p.d_seqstart_q[b];
        Sq_b = p.d_seqstart_q[b + 1] - qstart;
        if (i >= Sq_b) return;
        i_q = qstart + i;

        uint32_t kstart = p.d_seqstart_k[b];
        Skv_b = p.d_seqstart_k[b + 1] - kstart;
        i_k = kstart;
    }

    const uint16_t* Q_row  = p.d_Q + (size_t)b * p.stride_q_batch + (size_t)hq  * p.stride_q_head + (size_t)i_q * p.stride_q_seq;
    const uint16_t* K_base = p.d_K + (size_t)b * p.stride_k_batch + (size_t)hkv * p.stride_k_head + (size_t)i_k * p.stride_k_seq;
    const uint16_t* V_base = p.d_V + (size_t)b * p.stride_v_batch + (size_t)hkv * p.stride_v_head + (size_t)i_k * p.stride_v_seq;

    // fp32 partial output + LSE destinations for this row (flat row == tid).
    float* o_g   = d_o_g + (size_t)tid * D;
    float* lse_g = d_lse_g + tid;

    // Clamp the requested range to this row's valid KV span (mirrors
    // cpu_ref_split's `if (kv_start<0) kv_start=0; if (kv_end>Skv_b) ...`).
    int js = kv_start < 0 ? 0 : kv_start;
    int je = kv_end > Skv_b ? Skv_b : kv_end;

    // Empty range: nothing to attend to -> zero output, -inf LSE.
    if (js >= je) {
        for (int d = 0; d < D; d++) o_g[d] = 0.0f;
        *lse_g = -INFINITY;
        return;
    }

    // Pass 1: scan the sub-range to find the (range) max of the SCALED scores.
    float max_s = -INFINITY;
    for (int j = js; j < je; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        if (s > max_s) max_s = s;
    }

    // Entire sub-range masked out (causal): zero output, -inf LSE — matches the
    // empty-range convention and cpu_ref_split's local_max==-INF branch.
    if (max_s == -INFINITY) {
        for (int d = 0; d < D; d++) o_g[d] = 0.0f;
        *lse_g = -INFINITY;
        return;
    }

    // Pass 2: re-scan to accumulate the softmax denominator over the sub-range.
    float sum_exp = 0;
    for (int j = js; j < je; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        float e = (s == -INFINITY) ? 0.0f : expf(s - max_s);
        sum_exp += e;
    }

    float inv_sum = (sum_exp > 0) ? 1.0f / sum_exp : 0.0f;

    float O_acc[256];
    for (int d = 0; d < D; d++) O_acc[d] = 0;

    // Pass 3: recompute probabilities and accumulate O += P * V over the range.
    for (int j = js; j < je; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        float p_val = ((s == -INFINITY) ? 0.0f : expf(s - max_s)) * inv_sum;

        // Truncate P to BF16 and widen back, matching the production P.V GEMM
        // input precision and cpu_ref_split.
        uint16_t p_bf16 = float_to_bf16(p_val);
        float p_trunc = bf16_to_float(p_bf16);

        for (int d = 0; d < D; d++) {
            float v = bf16_to_float(V_base[(size_t)j * p.stride_v_seq + d]);
            O_acc[d] += p_trunc * v;
        }
    }

    // fp32 output, normalized by THIS range's sum (P already divided by it).
    for (int d = 0; d < D; d++) o_g[d] = O_acc[d];

    // ★ Per-range natural-log LSE.  local_max (== max_s) is ALREADY SCALED, so
    // there is NO extra `* scalar` here — matches gpu_ref.cpp:117 & cpu_ref_split
    // (logf(sum)+max).  Writing `scalar*max_s` would double-apply the scale.
    *lse_g = logf(sum_exp) + max_s;
}

void gpu_ref_split(const GpuRefParams& p, int kv_start, int kv_end,
                   float* d_o_g, float* d_lse_g, hipStream_t stream) {
    const int total_rows = p.batch * p.q_heads * p.seq_len;
    const int block = 256;
    const int grid = (total_rows + block - 1) / block;
    hipLaunchKernelGGL(gpu_ref_split_kernel, dim3(grid), dim3(block), 0, stream,
                       p, kv_start, kv_end, d_o_g, d_lse_g);
}

// One thread = one output row.  Combine G per-range partials by reweighting
// with each range's relative softmax mass.  Line-for-line port of
// cpu_ref_combine (natural-e domain, max-subtract for stability, double denom).
__global__ void gpu_ref_combine_kernel(int G, int rows, int Dlog,
                                       const float* d_o_part,
                                       const float* d_lse_part,
                                       float* d_o_final) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float* o_final = d_o_final + (size_t)row * Dlog;

    // Step 1: global max M = max_g lse_part[g] for this row.  Subtracting M
    // before every exp() keeps the largest exponent at exp(0)=1 (no overflow).
    float M = -INFINITY;
    for (int g = 0; g < G; ++g) {
        float l = d_lse_part[(size_t)g * rows + row];
        if (l > M) M = l;
    }

    // Step 2: zero the output (also the correct answer for the all-empty case).
    for (int d = 0; d < Dlog; ++d) o_final[d] = 0.0f;

    // Every range -inf -> no mass to combine, leave zeros (and never exp(-inf)).
    if (M == -INFINITY) return;

    // Step 3: unnormalized weights w_g = exp(lse_g - M); denom in double.
    // A -inf range maps to weight 0 explicitly (documents intent).
    double denom = 0.0;
    for (int g = 0; g < G; ++g) {
        float l = d_lse_part[(size_t)g * rows + row];
        float wg = (l == -INFINITY) ? 0.0f : expf(l - M);
        denom += wg;
    }

    // Step 4: normalize and accumulate the convex combination.  denom > 0 always
    // holds here (M finite => at least one weight is exp(0)=1) but guard anyway.
    float inv = (denom > 0.0) ? (float)(1.0 / denom) : 0.0f;
    for (int g = 0; g < G; ++g) {
        float l = d_lse_part[(size_t)g * rows + row];
        float wg = ((l == -INFINITY) ? 0.0f : expf(l - M)) * inv;
        if (wg == 0.0f) continue;                 // skip empty/masked ranges
        const float* o_g = d_o_part + ((size_t)g * rows + row) * Dlog;
        for (int d = 0; d < Dlog; ++d) o_final[d] += wg * o_g[d];
    }
}

void gpu_ref_combine(int G, int rows, int Dlog,
                     const float* d_o_part, const float* d_lse_part,
                     float* d_o_final, hipStream_t stream) {
    const int block = 256;
    const int grid = (rows + block - 1) / block;
    hipLaunchKernelGGL(gpu_ref_combine_kernel, dim3(grid), dim3(block), 0, stream,
                       G, rows, Dlog, d_o_part, d_lse_part, d_o_final);
}
