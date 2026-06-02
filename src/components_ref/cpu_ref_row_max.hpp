#pragma once
#include <cstdint>

// ===== REFERENCE ORACLE for the row_max component (src/components/row_max.hpp)
// Computes the per-row max the GPU's intra-lane + ds_bpermute reduction should
// produce. The CPU version skips the shuffle entirely: it just gathers BOTH
// k_sub halves of each M-row (the two threads tid and tid^32 that share a row)
// and takes the plain max over all 64 columns. Test-only oracle.
// ============================================================================
//
// CPU reference for row-max reduction over S_acc, Phase 1 Kernel 3.
//
// Computes the max over the N-dimension of S_acc for each thread's M-row.
// S_acc is in the verified TransposedC distribution:
//   m_row = (lane%32) + 32*warp       (warp=tid/64, lane=tid%64)
//   n_col = (r/8)*16 + (lane/32)*8 + (r%8)
//
// Each thread has ONE m_row and 32 registers spanning 32 of the 64 N-columns.
// Lanes with the same (lane%32, warp) but different k_sub (lane/32) hold the
// complementary N-columns. The row max must combine both halves.
//
// Input:  s_acc[256*32] -- raw S_acc values (no scale), thread-buffer order
// Output: rmax[256]     -- one scalar per thread (max over all 64 N-cols)
//
// After reduction, both k_sub halves (tid and tid^32) hold the same rmax.

constexpr int kRowMaxOutElems = 256;

// Fill `rmax` (256 floats) with expected row-max values.
void cpu_ref_row_max(const float* s_acc, float* rmax);
