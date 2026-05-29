#pragma once
#include <cstdint>

// CPU reference for GEMM0 (S = Q * K^T), Phase 1 Kernel 2.
//
// Produces the expected S_acc thread-buffer image: out[tid*32 + r] for
// tid in [0,256), r in [0,32), in the verified TransposedC distribution:
//   m_row(seqq) = (lane%32) + 32*warp       (warp=tid/64, lane=tid%64)
//   n_col(seqk) = (r/8)*16 + (lane/32)*8 + (r%8)
// Value = RAW sum_d Q[m_row,d] * K[n_col,d] (bf16 inputs, fp32 accum, NO scale).
// S = 0 where m_row >= seqlen_q (Q padding) or n_col >= seqlen_k (K padding).
//
// Q, K are raw bf16 (uint16) row-major [seqlen, D=64], row strides in elements.

constexpr int kQKOutElems = 256 * 32;

inline int qk_ref_m_row(int tid) {
    int warp = tid / 64, lane = tid % 64;
    return (lane % 32) + 32 * warp;
}
inline int qk_ref_n_col(int tid, int r) {
    int lane = tid % 64;
    return (r / 8) * 16 + (lane / 32) * 8 + (r % 8);
}

// Fill `out` (kQKOutElems floats) with expected S_acc values.
void ref_qk_gemm(const uint16_t* Q, int stride_q, int seqlen_q,
                 const uint16_t* K, int stride_k, int seqlen_k,
                 float* out);
