#include "components_ref/cpu_ref_qk_gemm.hpp"
#include "runner/bf16_utils.hpp"

// Brute-force S = Q*K^T in the GPU's thread-buffer layout. For each (tid,r)
// slot we recover the (m_row, n_col) it represents, then dot the 64-elem head.
// Inputs are read as bf16 then widened to fp32 (bf16_to_float) so rounding
// matches the GPU's bf16 operands; the accumulation is fp32, unscaled (raw).
void cpu_ref_qk_gemm(const uint16_t* Q, int stride_q, int seqlen_q,
                 const uint16_t* K, int stride_k, int seqlen_k,
                 float* out) {
    for (int tid = 0; tid < 256; ++tid) {
        int m = qk_ref_m_row(tid);
        for (int r = 0; r < 32; ++r) {
            int n = qk_ref_n_col(tid, r);
            float acc = 0.0f;
            if (m < seqlen_q && n < seqlen_k) {
                for (int d = 0; d < 64; ++d) {
                    float qv = bf16_to_float(Q[m * stride_q + d]);
                    float kv = bf16_to_float(K[n * stride_k + d]);
                    acc += qv * kv;
                }
            }
            out[tid * 32 + r] = acc;
        }
    }
}
