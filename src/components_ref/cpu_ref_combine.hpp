#pragma once

// ===== REFERENCE ORACLE for the split-K combine =============================
// This is the GROUND-TRUTH oracle that every later split-K combine gate trusts,
// so its math is written to be exactly right and hand-verifiable. Test-only.
// ============================================================================
//
// Background: split-K (a.k.a. "flash-decoding" style splitting) cuts the KV
// sequence into G disjoint ranges and runs an independent FMHA pass over each
// range. Each pass g produces, for one query row:
//   o_part[g][d]  : the *normalized* fp32 partial output of range g, where
//                   "normalized" means it was already divided by that range's
//                   local softmax denominator (rowsum). d in [0, Dlog).
//   lse_part[g]   : the natural-log LSE (log-sum-exp) of range g, i.e.
//                   lse = ln(Σ_{j in range g} exp(s_j - m_g)) + m_g, where
//                   m_g is range g's row max. lse_part[g] == -INFINITY marks a
//                   range that was empty or fully masked (it contributes 0).
//
// Combine math. Stitching the per-range partials back into the global softmax
// output is a *reweighting* by each range's relative softmax mass:
//   M        = max_g lse_part[g]                       (global max, stable)
//   L*       = ln( Σ_g exp(lse_part[g] - M) ) + M      (global LSE)
//   w_g      = exp( lse_part[g] - L* )                 (range g's weight)
//   o_final[d] = Σ_g w_g * o_part[g][d]
//
// Because each o_part[g] is already locally normalized, the global output is a
// convex combination of the partials. The weights w_g sum to 1 *by
// construction* (Σ_g exp(lse_g - L*) = exp(-L*) * Σ_g exp(lse_g) = 1, since
// exp(L*) = Σ_g exp(lse_g)). So this oracle is a PURE REWEIGHT: it never
// rescales the total mass, it only redistributes it across ranges.
//
// Edge cases:
//   - Any range with lse_part[g] == -INFINITY contributes exactly 0 (w_g = 0).
//   - If ALL ranges are -INFINITY (every range empty/masked), o_final = 0.
//
// Numerical stability: the implementation subtracts the max M before every
// exp() so the largest exponent is exactly 0 (no overflow), and accumulates
// the denominator in double precision.
//
// Inputs:
//   G       : number of split-K ranges (partials) for this row.
//   Dlog    : logical head dimension (e.g. 64); length of each partial.
//   o_part  : [G*Dlog] normalized fp32 partials, row-major (range-major).
//   lse_part: [G]      natural-log LSE per range (-INFINITY if empty/masked).
// Output:
//   o_final : [Dlog]   combined fp32 output for this row.
void cpu_ref_combine(int G, int Dlog,
                 const float* o_part,   // [G*Dlog]
                 const float* lse_part, // [G]
                 float* o_final);       // [Dlog]
