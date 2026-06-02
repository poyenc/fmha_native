#include "components_ref/cpu_ref_combine.hpp"
#include <cmath>
#include <vector>

// CPU oracle for the split-K combine. See cpu_ref_combine.hpp for the full
// derivation. Summary: o_final[d] = Σ_g w_g * o_part[g][d], with normalized
// weights w_g = exp(lse_part[g] - L*) that sum to 1 by construction. This file
// computes the same thing in the natural-e domain using the numerically stable
// max-subtract trick.
void cpu_ref_combine(int G, int Dlog, const float* o_part,
                 const float* lse_part, float* o_final) {
    // --- Step 1: global max M = max_g lse_part[g]. We subtract M before every
    // exp() so the largest exponent is exactly exp(0)=1; this prevents overflow
    // for large LSE values (e.g. lse=50) while keeping equal-LSE cases exact.
    float M = -INFINITY;
    for (int g = 0; g < G; ++g) if (lse_part[g] > M) M = lse_part[g];

    // --- Step 2: zero the output up front. This is also the correct answer for
    // the "all ranges empty/masked" case handled by the early return below.
    for (int d = 0; d < Dlog; ++d) o_final[d] = 0.0f;

    // If every range is -inf, M stays -INFINITY: there is no mass to combine,
    // so o_final stays all zeros (and we never feed -inf into exp()).
    if (M == -INFINITY) return;             // all empty/masked -> zeros

    // --- Step 3: unnormalized weights w[g] = exp(lse_g - M). Accumulate the
    // denominator (= Σ_g exp(lse_g - M) = exp(L* - M)) in double precision to
    // limit rounding error when G is large or the spread is wide. A -inf range
    // maps to weight 0 explicitly (expf(-inf) would also give 0, but being
    // explicit documents intent and avoids touching that range below).
    double denom = 0.0;
    std::vector<float> w(G);
    for (int g = 0; g < G; ++g) {
        w[g] = (lse_part[g] == -INFINITY) ? 0.0f : expf(lse_part[g] - M);
        denom += w[g];
    }

    // --- Step 4: normalize (divide by denom so Σ_g w_g = 1) and accumulate the
    // convex combination. denom > 0 always holds here because M is finite (at
    // least one range has lse == M, giving w = exp(0) = 1), but we guard anyway.
    float inv = (denom > 0.0) ? (float)(1.0/denom) : 0.0f;
    for (int g = 0; g < G; ++g) {
        float wg = w[g] * inv;              // normalized weight, Σ=1
        if (wg == 0.0f) continue;           // skip empty/masked ranges
        for (int d = 0; d < Dlog; ++d) o_final[d] += wg * o_part[g*Dlog + d];
    }
}
