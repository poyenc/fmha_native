#include "components_ref/ref_row_max.hpp"
#include <cmath>
#include <algorithm>

void ref_row_max(const float* s_acc, float* rmax) {
    // For each m_row (0..127), find max over all 64 N-columns from s_acc.
    // m_row = (lane%32) + 32*warp.
    // Lanes with same m_row: same (lane%32, warp) but k_sub=0 and k_sub=1.
    //
    // Strategy: for each m_row, gather all S_acc values from the two threads
    // that own this row (k_sub=0 and k_sub=1), take max.

    // Build per-row max
    float row_max_vals[128];
    for (int m = 0; m < 128; ++m) row_max_vals[m] = -INFINITY;

    for (int tid = 0; tid < 256; ++tid) {
        int warp = tid / 64;
        int lane = tid % 64;
        int m_row = (lane % 32) + 32 * warp;
        for (int r = 0; r < 32; ++r) {
            float v = s_acc[tid * 32 + r];
            row_max_vals[m_row] = std::fmax(row_max_vals[m_row], v);
        }
    }

    // Each thread gets the max for its m_row.
    for (int tid = 0; tid < 256; ++tid) {
        int warp = tid / 64;
        int lane = tid % 64;
        int m_row = (lane % 32) + 32 * warp;
        rmax[tid] = row_max_vals[m_row];
    }
}
