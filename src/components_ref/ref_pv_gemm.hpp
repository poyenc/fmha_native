#pragma once
#include <cstdint>

// ===== REFERENCE ORACLE for the GEMM1 component (src/components/pv_gemm.hpp) ==
// CPU truth for O_acc = P*V in the GPU's SwizzleA'd TransposedC output layout.
// The tricky part is reading P: P is stored in the SAME TransposedC layout, so
// to fetch logical element P[m,n] the .cpp must invert that mapping (see get_p).
// O_acc here is RAW (not yet divided by rsum); the epilogue does that. Test-only.
// ============================================================================
//
// CPU reference for GEMM1 (O = P * V), Phase 1 Kernel 6.
//
// Produces the expected O_acc thread-buffer image: out[tid*32 + r] for
// tid in [0,256), r in [0,32), in the TransposedC+SwizzleA distribution:
//   m_row(seqq)  = (lane%32) + 32*warp
//   d_col(hdim)  = swz((r/8)*16 + (lane/32)*8 + (r%8))
//   where swz swaps bits 2,3: swz(x) = (x & ~0xC) | ((x>>2&1)<<3) | ((x>>3&1)<<2)
//
// Value = sum_{n=0}^{63} P_bf16[m_row, n] * V_bf16[n, d_col]
// P is bf16 (32 regs, TransposedC layout). V is bf16 [seqlen_k, D=64].
// O_acc is the raw MFMA accumulator, NOT normalized by rsum.

constexpr int kPVOutElems = 256 * 32;

inline int pv_swz(int x) {
    int b2 = (x >> 2) & 1;
    int b3 = (x >> 3) & 1;
    return (x & ~0xC) | (b2 << 3) | (b3 << 2);
}

inline int pv_ref_m_row(int tid) {
    int warp = tid / 64, lane = tid % 64;
    return (lane % 32) + 32 * warp;
}

inline int pv_ref_d_col(int tid, int r) {
    int lane = tid % 64;
    int k_sub = lane / 32;
    int d_nom = (r / 8) * 16 + k_sub * 8 + (r % 8);
    return pv_swz(d_nom);
}

// P is in TransposedC layout: p[tid*32+r] where
//   m = (lane%32)+32*warp, n = (r/8)*16+(lane/32)*8+(r%8).
// These are bf16-promoted fp32 values.
// V is raw bf16 row-major [seqlen_k, D=64].
void ref_pv_gemm(const float* p_acc,     // [256*32] bf16-promoted fp32
                 const uint16_t* V, int stride_v, int seqlen_k,
                 float* out);            // [256*32] fp32 O_acc
