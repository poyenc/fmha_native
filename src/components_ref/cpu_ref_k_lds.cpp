// CPU oracle for the K-LDS staging component — see cpu_ref_k_lds.hpp for the
// golden offset formula. Builds the expected LDS byte image for the GPU's
// swizzled K copy; the test byte-compares the GPU's dumped LDS against it.
#include "components_ref/cpu_ref_k_lds.hpp"
#include <cstring>

void cpu_ref_k_lds(const uint16_t* K,
               int stride_k,
               int kv_offset,
               int seqlen_k,
               uint16_t* expected_lds) {
    // Zero the whole region first (matches CK OOB zero-fill + untouched pad slots).
    std::memset(expected_lds, 0, kKLdsRegionElems * sizeof(uint16_t));

    // 64-row x 64-col K tile staged into LDS via the verified layout.
    for (int j = 0; j < 64; ++j) {
        int row = kv_offset + j;
        if (row >= seqlen_k) continue; // OOB rows stay zero
        for (int d = 0; d < 64; ++d) {
            int e = k_lds_offset_elems(j, d);
            expected_lds[e] = K[row * stride_k + d];
        }
    }
}
