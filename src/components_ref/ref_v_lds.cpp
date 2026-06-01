#include "components_ref/ref_v_lds.hpp"
#include <cstring>

void ref_v_lds(const uint16_t* V,
               int stride_v,
               int kv_offset,
               int seqlen_k,
               uint16_t* expected_lds) {
    // Zero the entire LDS region (OOB/padding stays 0)
    std::memset(expected_lds, 0, kVLdsRegionElems * sizeof(uint16_t));

    // Stage ONE 32-row k1 slice (n = 0..31 within the tile).
    // Golden V_LDS dump captures the first k1 slice at buffer_base = 2304.
    for (int n = 0; n < 32; ++n) {
        int row = kv_offset + n;
        if (row >= seqlen_k) continue;

        for (int d = 0; d < 64; ++d) {
            int off = kVLdsBufBase + v_lds_offset_elems(n, d);
            expected_lds[off] = V[row * stride_v + d];
        }
    }
}
