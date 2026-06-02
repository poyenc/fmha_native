#pragma once
#include <cstdint>

// ===== REFERENCE ORACLE for the softmax component (src/components/softmax.hpp)
// CPU truth for P/rmax/rsum. Unlike the full-kernel CPU references (which use
// natural expf), THIS oracle mirrors the GPU exactly: it uses exp2f with the
// same log2(e)-folded scale, so the comparison can be (near) bit-exact. The
// cross-half merge is done by directly reading the partner thread (tid^32)
// instead of a shuffle. Test-only oracle.
// ============================================================================
//
// CPU reference for softmax (scale + mask + exp2 + rowsum + bf16 cast),
// Phase 1 Kernel 4.
//
// Given S_acc (256×32 fp32, raw unscaled Q·Kᵀ) and tile parameters,
// produces:
//   P      (256×32 fp32, bf16-promoted): unnormalized softmax output
//   rmax   (256 fp32): row max of MASKED+SCALED S_acc
//   rsum   (256 fp32): row sum of P_fp32 (pre-bf16-cast)
//
// Pipeline:
//   1. Mask: S[n_col >= seqlen_k] = -INFINITY
//   2. Rmax: intra-lane max + cross-half merge (K3 algorithm)
//   3. Exp2: P_fp32 = exp2(scale_s_log2e * (S_masked - rmax))
//   4. Rsum: intra-lane sum + cross-half merge
//   5. Cast: P_bf16 = bf16_trunc(P_fp32)  (truncation, not RNE)
//
// S_acc layout: m=(lane%32)+32*warp, n=(r/8)*16+(lane/32)*8+(r%8).
// scale_s_log2e = scale_s * log2(e) = 0.125 * 1.4426950408... = 0.1803368801...
// CK FAST_EXP2 path: exp2 used directly (no exp).

constexpr int kSoftmaxOutRegs = 32;   // P regs per thread
constexpr int kSoftmaxNThreads = 256;

void cpu_ref_softmax(const float* s_acc,
                 int seqlen_k, int kv_offset,
                 float scale_s,
                 float* p_out,      // [256*32] bf16-promoted fp32
                 float* rmax_out,   // [256]
                 float* rsum_out);  // [256]
