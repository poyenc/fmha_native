#include "refs/ref_epilog.hpp"
#include "runner/bf16_utils.hpp"
#include <cstring>

void ref_epilog(const float* o_acc,
                const float* rsum,
                int seqlen_q,
                float* o_final,
                float* o_dram) {
    // Step 1: Normalize O_acc by RSUM
    for (int tid = 0; tid < 256; ++tid) {
        float rs = rsum[tid];
        float inv_rs = (rs != 0.0f) ? (1.0f / rs) : 0.0f;
        for (int r = 0; r < 32; ++r) {
            o_final[tid * 32 + r] = o_acc[tid * 32 + r] * inv_rs;
        }
    }

    // Step 2: Store to DRAM with bf16 truncation
    std::memset(o_dram, 0, seqlen_q * 64 * sizeof(float));

    for (int tid = 0; tid < 256; ++tid) {
        int warp = tid / 64;
        int lane = tid % 64;
        int k_sub = lane / 32;
        int m_row = (lane % 32) + 32 * warp;

        if (m_row >= seqlen_q) continue;

        for (int r = 0; r < 32; ++r) {
            int d_nom = (r / 8) * 16 + k_sub * 8 + (r % 8);
            int d_col = epilog_swz(d_nom);

            float val = o_final[tid * 32 + r];
            uint16_t bf16 = float_to_bf16(val);
            o_dram[m_row * 64 + d_col] = bf16_to_float(bf16);
        }
    }
}
