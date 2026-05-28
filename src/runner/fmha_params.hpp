// Workload configuration for FMHA forward pass.
//
// FmhaParams holds the problem dimensions, feature flags, and tiling
// parameters shared by the benchmark, GPU reference kernel, HSACO shader
// dispatch, and CPU reference verification.  CLI parsing is intentionally
// excluded — that responsibility belongs to the benchmark binary.

#pragma once
#include <cmath>
#include <vector>

struct FmhaParams {
    // --- Problem dimensions ---
    int batch      = 2;       // number of sequences (or varlen batch count)
    int q_heads    = 16;      // query head count
    int kv_heads   = 0;       // key/value head count (0 = derive from q_heads/gqa)
    int gqa        = 1;       // grouped-query-attention ratio (q_heads / kv_heads)
    int seq_len    = 4096;    // query sequence length (max across varlen batches)
    int kv_seq_len = 0;       // key/value sequence length (0 = seq_len)
    int head_dim   = 128;     // logical head dimension (data elements per row)
    int hdim_padded = 0;      // padded head dimension for dispatch (0 = head_dim).
                              // When > head_dim, the kernel operates on wider rows
                              // with trailing zeros — numerically equivalent to true
                              // head_dim but dispatched with a larger-D HSACO.

    // --- Tiling / launch geometry ---
    int sub_Q      = 256;     // sub-Q tile size (query rows per threadgroup)
    int wv_tg      = 8;       // waves per threadgroup (4 → 256 threads, 8 → 512)

    // --- Feature flags ---
    int mask       = 0;       // causal mask enable (right-aligned FA-2 convention)
    int opt        = 0;       // mask optimization strategy (0=none, 1=reverse, 2=remap, 3=both)
    int lse        = 0;       // log-sum-exp output enable

    // --- Variable-length batching ---
    std::vector<int> varlen_seqs;  // per-batch sequence lengths; when non-empty,
                                   // batch is inferred from size, seq_len from max

    // Softmax temperature: 1/sqrt(head_dim).
    float scalar() const { return 1.0f / sqrtf((float)head_dim); }

    // Head dimension seen by the kernel (padded or logical).
    int hdim_dispatch() const { return hdim_padded > 0 ? hdim_padded : head_dim; }
};
