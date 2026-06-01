#include "components_ref/ref_pv_gemm.hpp"
#include "runner/bf16_utils.hpp"

// Extract P[m, n] from the P thread-buffer (TransposedC layout).
static float get_p(const float* p_acc, int m, int n) {
    int warp = m / 32;
    int lane_base = m % 32;
    int k_sub, r_group, r_within;
    if (n % 16 < 8) {
        k_sub = 0; r_group = n / 16; r_within = n % 8;
    } else {
        k_sub = 1; r_group = n / 16; r_within = (n - 8) % 8;
    }
    int r = r_group * 8 + r_within;
    int tid = warp * 64 + k_sub * 32 + lane_base;
    return p_acc[tid * 32 + r];
}

void ref_pv_gemm(const float* p_acc,
                 const uint16_t* V, int stride_v, int seqlen_k,
                 float* out) {
    for (int tid = 0; tid < 256; ++tid) {
        int m = pv_ref_m_row(tid);
        for (int r = 0; r < 32; ++r) {
            int d = pv_ref_d_col(tid, r);
            float acc = 0.0f;
            for (int n = 0; n < 64; ++n) {
                float pv = get_p(p_acc, m, n);
                float vv = (n < seqlen_k) ? bf16_to_float(V[n * stride_v + d]) : 0.0f;
                acc += pv * vv;
            }
            out[tid * 32 + r] = acc;
        }
    }
}
