#pragma once
#include <cstdint>

// ===== REFERENCE ORACLE for the epilogue component (src/components/epilog.hpp)
// CPU truth for the final output: divide O_acc by RSUM, truncate to bf16, and
// scatter into row-major DRAM at the swz()-permuted column. Emits both the
// pre-store normalized fp32 (o_final) and the bf16-promoted DRAM image
// (o_dram) so the test can check each against its golden slot. Test-only.
// ============================================================================
//
// CPU reference for Default2D epilogue, Phase 1 Kernel 7.
//
// Given O_acc (256×32 fp32) and RSUM (256 fp32):
//   O_final[tid*32+r] = O_acc[tid*32+r] / RSUM[tid]
// Then bf16 truncation and DRAM store at row-major O[m_row, d_col]:
//   m_row = (lane%32) + 32*warp
//   d_col = swz((r/8)*16 + (lane/32)*8 + (r%8))
//
// o_dram format: fp32 (bf16-promoted), row-major [seqlen_q, D=64].
// Only valid M-rows (m < seqlen_q) are stored.

inline int epilog_swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

// Compute O_final (normalized, fp32) and DRAM output (bf16-promoted fp32).
// o_final: [256*32] fp32 (for golden slot 6 comparison)
// o_dram:  [seqlen_q*64] fp32 bf16-promoted (for o_dram.bin comparison)
void ref_epilog(const float* o_acc,     // [256*32]
                const float* rsum,      // [256]
                int seqlen_q,
                float* o_final,         // [256*32]
                float* o_dram);         // [seqlen_q*64]
