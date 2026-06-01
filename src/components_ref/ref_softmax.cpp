#include "components_ref/ref_softmax.hpp"
#include "runner/bf16_utils.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

// Map a (tid, reg r) slot back to the N-column it represents in the
// TransposedC distribution. Mirrors the GPU's n_col formula so masking and the
// max/sum reductions operate on the same logical columns.
static int softmax_n_col(int tid, int r) {
    int lane = tid % 64;
    int k_sub = lane >> 5;
    return (r / 8) * 16 + k_sub * 8 + (r % 8);
}

void ref_softmax(const float* s_acc,
                 int seqlen_k, int kv_offset,
                 float scale_s,
                 float* p_out,
                 float* rmax_out,
                 float* rsum_out) {
    const float log2e = 1.4426950408889634f;
    const float scale_s_log2e = scale_s * log2e;

    // Step 1: Apply mask + compute per-thread local max over 32 regs
    float local_max[256];
    float s_masked[256 * 32];

    for (int tid = 0; tid < 256; ++tid) {
        float lm = -INFINITY;
        for (int r = 0; r < 32; ++r) {
            int n_col = softmax_n_col(tid, r);
            float s = s_acc[tid * 32 + r];
            if (n_col >= (seqlen_k - kv_offset))
                s = -INFINITY;
            s_masked[tid * 32 + r] = s;
            lm = std::fmax(lm, s);
        }
        local_max[tid] = lm;
    }

    // Step 2: Cross-half merge for rmax
    for (int tid = 0; tid < 256; ++tid) {
        int lane = tid % 64;
        int partner_tid = (tid & ~63) | ((lane ^ 32) & 63);
        rmax_out[tid] = std::fmax(local_max[tid], local_max[partner_tid]);
    }

    // Step 3: Compute P_fp32 = exp2(scale_s_log2e * (S_masked - rmax))
    //         and accumulate rowsum (from fp32, before bf16 cast)
    float local_sum[256];
    for (int tid = 0; tid < 256; ++tid) {
        float rmax = rmax_out[tid];
        float ls = 0.0f;
        for (int r = 0; r < 32; ++r) {
            float s = s_masked[tid * 32 + r];
            float p;
            if (s == -INFINITY) {
                p = 0.0f;
            } else {
                float arg = scale_s_log2e * (s - rmax);
                p = exp2f(arg);
            }
            // Store fp32 P for rowsum accumulation
            p_out[tid * 32 + r] = p;
            ls += p;
        }
        local_sum[tid] = ls;
    }

    // Step 4: Cross-half merge for rowsum
    for (int tid = 0; tid < 256; ++tid) {
        int lane = tid % 64;
        int partner_tid = (tid & ~63) | ((lane ^ 32) & 63);
        rsum_out[tid] = local_sum[tid] + local_sum[partner_tid];
    }

    // Step 5: Truncate P to bf16 (in-place, promoted back to fp32)
    for (int tid = 0; tid < 256; ++tid) {
        for (int r = 0; r < 32; ++r) {
            float p = p_out[tid * 32 + r];
            uint16_t bf16 = float_to_bf16(p);
            p_out[tid * 32 + r] = bf16_to_float(bf16);
        }
    }
}
