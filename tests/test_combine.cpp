// =============================================================================
// UNIT TEST: split-K COMBINE kernel (fmha_fwd_d64_bf16_combine) — TDD RED step.
//
// THIS TEST IS INTENTIONALLY UN-LINKABLE RIGHT NOW.  The combine __global__ does
// not exist yet (it arrives in Task 4).  The test below references it, so the
// link step fails with an "undefined reference to fmha_fwd_d64_bf16_combine"
// error.  That undefined-symbol error IS the RED state we want: the test must
// COMPILE cleanly (the param struct lives in runner/params.hpp), and only fail
// to LINK.  When Task 4 lands the kernel, this same file turns GREEN unchanged.
//
// It validates the combine in two complementary ways:
//
//   Source A (synthetic): hand-built adversarial o_part / lse_part patterns fed
//     straight into the GPU combine, diffed against cpu_ref_combine per row.  G
//     sweeps over both powers-of-two AND non-powers-of-two, including G=1 (the
//     identity case: output must equal o_part[0]).
//
//   Source B (real-partial, the G-INVARIANCE test): build G real FMHA partials
//     by calling cpu_ref_split over G disjoint KV sub-ranges, combine them on
//     the GPU, and assert the result equals the FULL-range attention
//     (cpu_ref_split over [0, Skv)).  Because the global softmax is invariant to
//     HOW the KV axis is partitioned, the combined output must match the
//     un-split reference for EVERY G — that is the invariant under test.
//
// ---------------------------------------------------------------------------
// COMBINE LAUNCH CONTRACT (defined HERE; Task 4 must implement to match)
// ---------------------------------------------------------------------------
//   * Grid  : dim3(nhead_q, m_tiles, batch)  — one block per (b, h, m_tile),
//             EXACTLY mirroring the forward kernel's grid (kernel.cpp launches
//             grid(Hq, m_tiles, B)).  m_tile size is kM0 (=128) query rows.
//   * Block : kBlockSize (=256) threads, the same as the forward/epilog path.
//   * Reads : scratch_o / scratch_lse, split-major (see FmhaFwdCombineParams in
//             runner/params.hpp).  For output row R = blockIdx.y*kM0 + (row in
//             tile), and head h=blockIdx.x, batch b=blockIdx.z, the kernel must
//             gather the G planes scratch_o[g][b][h][R][0..63] and the G scalars
//             scratch_lse[g][b][h][R], run the combine math (== cpu_ref_combine),
//             and store the 64 fp32 results truncated to BF16 into O at the
//             epilogue's row/col mapping:
//               O index = b*batch_stride_o + h*nhead_stride_o
//                       + R*stride_o + d                  (d in [0,64))
//   * LSE   : if params.lse != nullptr, also write the global LSE per row.  The
//             synthetic + invariance tests below do NOT exercise the LSE path
//             (params.lse == nullptr); a dedicated LSE gate is out of scope for
//             the RED step.
//
// To keep Source A independent of any internal tiling cleverness, it uses ONLY
// b=0, h=0, m_tile=0 and small row counts (< kM0), so "output row R" is just the
// scratch row index r in [0, num_rows).  Source B uses the real grid mapping.
//
// NOTE on softmax domain: cpu_ref_split / cpu_ref_combine work in the natural-e
// domain (expf, plain 1/sqrt(d)); the eventual GPU combine math is exact fp32
// reweighting, domain-agnostic.  The 2e-3 Source-B tolerance (see below) is for
// the per-range BF16 P-truncation drift flagged upstream in Task 1, NOT for the
// combine arithmetic, which is exact.
// =============================================================================
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "runner/params.hpp"        // FmhaFwdCombineParams (owned here since T3)
#include "runner/fmha_params.hpp"   // FmhaParams (workload config for Source B)
#include "runner/buffers.hpp"       // FmhaBuffers + fill_random (Source B inputs)
#include "runner/cpu_ref.hpp"       // cpu_ref_split (real-partial oracle, host)
#include "components_ref/cpu_ref_combine.hpp"  // ground-truth combine oracle

// ---------------------------------------------------------------------------
// THE undefined symbol that makes this the RED test.  Declared with the param
// struct that already exists in runner/params.hpp; defined by Task 4.
// ---------------------------------------------------------------------------
extern __global__ void fmha_fwd_d64_bf16_combine(FmhaFwdCombineParams);

namespace {

constexpr int kD = 64;  // head_dim this kernel is specialized for

// ---------------------------------------------------------------------------
// Compare helper: max-abs over a row block, with a few mismatches printed.
// ---------------------------------------------------------------------------
void compare_max_abs(const std::vector<float>& got, const std::vector<float>& exp,
                     const char* label, float tol) {
    ASSERT_EQ(got.size(), exp.size());
    float max_abs = 0.0f;
    int mism = 0, shown = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float ae = std::fabs(got[i] - exp[i]);
        if (ae > max_abs) max_abs = ae;
        if (ae > tol) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [%s] mismatch i=%zu got=%.6f exp=%.6f abs=%.6g\n",
                        label, i, got[i], exp[i], ae);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] max_abs=%.6g mism=%d/%zu (tol=%.6g)\n",
            label, max_abs, mism, got.size(), tol);
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values exceed tol " << tol;
}

// ---------------------------------------------------------------------------
// GPU combine driver.
//
// Uploads host-side scratch (split-major G partials + per-row LSE), launches the
// combine kernel over a grid of (nhead_q, m_tiles, batch), and copies the final
// BF16 O back as fp32.  The O buffer is sized for the WHOLE (B,Hq,Sq) problem so
// the contract's O index math is exercised; callers extract the rows they care
// about.
//
// Strides are contiguous BHSD: stride_o = D, nhead_stride_o = Sq*D,
// batch_stride_o = Hq*Sq*D — matching FmhaFwdParams in the full-kernel tests.
// ---------------------------------------------------------------------------
struct CombineDims {
    int G;        // number of split planes
    int B;        // batch
    int Hq;       // query heads
    int Sq;       // query rows
};

std::vector<float> run_combine(const CombineDims& dim,
                               const std::vector<float>& scratch_o,   // [G*B*Hq*Sq*D]
                               const std::vector<float>& scratch_lse) // [G*B*Hq*Sq]
{
    const int G = dim.G, B = dim.B, Hq = dim.Hq, Sq = dim.Sq;
    const size_t n_o     = (size_t)B * Hq * Sq * kD;       // final O elements
    const size_t n_sc_o  = (size_t)G * n_o;                // scratch O elements
    const size_t n_sc_lse = (size_t)G * B * Hq * Sq;       // scratch LSE elements

    EXPECT_EQ(scratch_o.size(),   n_sc_o);
    EXPECT_EQ(scratch_lse.size(), n_sc_lse);

    void *dScO = nullptr, *dScLse = nullptr, *dO = nullptr;
    EXPECT_EQ(hipMalloc(&dScO,   n_sc_o   * sizeof(float)),    hipSuccess);
    EXPECT_EQ(hipMalloc(&dScLse, n_sc_lse * sizeof(float)),    hipSuccess);
    EXPECT_EQ(hipMalloc(&dO,     n_o      * sizeof(uint16_t)), hipSuccess);
    EXPECT_EQ(hipMemcpy(dScO,   scratch_o.data(),   n_sc_o   * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    EXPECT_EQ(hipMemcpy(dScLse, scratch_lse.data(), n_sc_lse * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    EXPECT_EQ(hipMemset(dO, 0, n_o * sizeof(uint16_t)), hipSuccess);

    FmhaFwdCombineParams cp{};
    cp.scratch_o   = reinterpret_cast<const float*>(dScO);
    cp.scratch_lse = reinterpret_cast<const float*>(dScLse);
    cp.o           = reinterpret_cast<__hip_bfloat16*>(dO);
    cp.lse         = nullptr;            // LSE path not exercised in the RED step
    cp.num_splits  = G;
    cp.seqlen_q    = Sq;
    cp.nhead_q     = Hq;
    cp.stride_o        = kD;
    cp.nhead_stride_o  = Sq * kD;
    cp.batch_stride_o  = Hq * Sq * kD;
    cp.scale       = 1.0f;              // only used for global LSE (unused here)

    // Grid mirrors the forward kernel: (Hq, m_tiles, B), kBlockSize threads.
    const int m_tiles = (Sq + kM0 - 1) / kM0;
    dim3 grid(Hq, m_tiles, B);
    dim3 block(kBlockSize);
    hipLaunchKernelGGL(fmha_fwd_d64_bf16_combine, grid, block, 0, nullptr, cp);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint16_t> o_bf16(n_o);
    EXPECT_EQ(hipMemcpy(o_bf16.data(), dO, n_o * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), hipSuccess);

    std::vector<float> o_f32(n_o);
    for (size_t i = 0; i < n_o; ++i) o_f32[i] = bf16_to_float(o_bf16[i]);

    hipFree(dScO); hipFree(dScLse); hipFree(dO);
    return o_f32;
}

// Index helpers for the split-major scratch layout (see FmhaFwdCombineParams).
inline size_t scratch_o_idx(int g, int b, int h, int row, int d,
                            int B, int Hq, int Sq) {
    return ((((size_t)g * B + b) * Hq + h) * Sq + row) * kD + d;
}
inline size_t scratch_lse_idx(int g, int b, int h, int row,
                              int B, int Hq, int Sq) {
    return (((size_t)g * B + b) * Hq + h) * Sq + row;
}

// The G set under test: powers AND non-powers of two, plus the G=1 identity.
const std::vector<int> kGSet = {1, 2, 3, 4, 5, 7, 8, 16};

} // namespace

// ===========================================================================
// Source A — synthetic adversarial partials.  b=0, h=0, m_tile=0; scratch row r
// maps directly to output row r in [0, num_rows).  num_rows < kM0 keeps it a
// single m-tile and keeps the row mapping trivial.
// ===========================================================================
TEST(CombineSynthetic, AdversarialPatterns) {
    const int B = 1, Hq = 1, Sq = 96;   // 96 < kM0 (128): one m-tile
    const int num_rows = Sq;

    for (int G : kGSet) {
        const size_t n_o    = (size_t)B * Hq * Sq * kD;
        std::vector<float> scratch_o((size_t)G * n_o, 0.0f);
        std::vector<float> scratch_lse((size_t)G * B * Hq * Sq, 0.0f);

        // Deterministic but adversarial per-(row,g) fill.  We rotate the LSE
        // pattern by row so different rows stress different combine regimes:
        //   pattern 0: all-equal LSE        (uniform weights)
        //   pattern 1: one dominant plane   (near-degenerate weights)
        //   pattern 2: monotone ramp        (graded weights)
        //   pattern 3: random ± LSE/o_part  (general case)
        //   pattern 4: some -inf planes     (empty/masked ranges drop out)
        unsigned rng = 0x1234567u ^ (unsigned)G;
        auto next = [&]() {                 // cheap LCG in [0,1)
            rng = rng * 1664525u + 1013904223u;
            return (rng >> 8) * (1.0f / 16777216.0f);
        };

        for (int r = 0; r < num_rows; ++r) {
            int pattern = r % 5;
            for (int g = 0; g < G; ++g) {
                float lse;
                switch (pattern) {
                    case 0: lse = 2.5f; break;                       // all-equal
                    case 1: lse = (g == 0) ? 30.0f : -5.0f; break;   // one-dominant
                    case 2: lse = -3.0f + 1.5f * g; break;           // monotone ramp
                    case 3: lse = (next() - 0.5f) * 20.0f; break;    // random ±
                    default:                                         // some -inf
                        lse = (g % 2 == 0) ? -INFINITY : (next() * 6.0f - 3.0f);
                        break;
                }
                // Guarantee at least one finite plane per row so the combine has
                // mass (pattern 4 with all-even-g -inf at G=1 would be all -inf:
                // force g==0 finite there).
                if (pattern == 4 && g == 0 && std::isinf(lse))
                    lse = 1.0f;
                scratch_lse[scratch_lse_idx(g, 0, 0, r, B, Hq, Sq)] = lse;

                for (int d = 0; d < kD; ++d) {
                    // Distinct, well-separated per-(g,d) partials so a wrong
                    // weight assignment is visible in the combined output.
                    float val = (next() - 0.5f) * 4.0f + 0.1f * g + 0.01f * d;
                    scratch_o[scratch_o_idx(g, 0, 0, r, d, B, Hq, Sq)] = val;
                }
            }
        }

        // GPU combine over the whole (B=1,Hq=1,Sq) problem.
        std::vector<float> gpu_o = run_combine({G, B, Hq, Sq}, scratch_o, scratch_lse);

        // CPU oracle per row: gather the G planes (range-major) for this row.
        std::vector<float> exp_o(n_o, 0.0f);
        std::vector<float> o_part((size_t)G * kD), lse_part(G);
        for (int r = 0; r < num_rows; ++r) {
            for (int g = 0; g < G; ++g) {
                lse_part[g] = scratch_lse[scratch_lse_idx(g, 0, 0, r, B, Hq, Sq)];
                for (int d = 0; d < kD; ++d)
                    o_part[g * kD + d] =
                        scratch_o[scratch_o_idx(g, 0, 0, r, d, B, Hq, Sq)];
            }
            cpu_ref_combine(G, kD, o_part.data(), lse_part.data(),
                            exp_o.data() + (size_t)r * kD);
        }

        std::string label = "synthetic/G=" + std::to_string(G);
        // fp32 combine is exact; allow only BF16 store rounding (< 1e-3).
        compare_max_abs(gpu_o, exp_o, label.c_str(), 1e-3f);

        // G=1 identity: output row must equal o_part[0] (modulo bf16 store).
        if (G == 1) {
            std::vector<float> ident(n_o);
            for (int r = 0; r < num_rows; ++r)
                for (int d = 0; d < kD; ++d)
                    ident[(size_t)r * kD + d] =
                        bf16_to_float(float_to_bf16(
                            scratch_o[scratch_o_idx(0, 0, 0, r, d, B, Hq, Sq)]));
            compare_max_abs(gpu_o, ident, "synthetic/G=1/identity", 1e-6f);
        }
    }
}

// ===========================================================================
// Source B — real partials from cpu_ref_split, the G-INVARIANCE test.
//
// Build a small real FMHA problem, fill_random on the HOST (cpu_ref_split reads
// host Q/K/V — no device copy needed for the oracle).  For each G, split the KV
// axis into G tile-aligned sub-ranges, compute a partial per range, combine on
// the GPU, and assert the result == the full-range attention.  S=2048 keeps the
// host oracle fast while giving enough KV tiles to make ragged/empty splits
// meaningful for non-power-of-two G.
// ===========================================================================
TEST(CombineRealPartial, GInvariance) {
    FmhaParams p{};
    p.batch    = 1;
    p.q_heads  = 2;
    p.kv_heads = 2;       // no GQA (gqa=1) keeps the oracle mapping simple
    p.gqa      = 1;
    p.seq_len  = 2048;    // Sq
    p.kv_seq_len = 2048;  // Skv (host oracle is fast at this size)
    p.head_dim = kD;
    p.mask     = 0;       // dense (no causal); invariance must hold regardless

    const int B  = p.batch;
    const int Hq = p.q_heads;
    const int Sq = p.seq_len;
    const int Skv = p.kv_seq_len;

    FmhaBuffers bufs(p);
    bufs.fill_random(42);   // host Q/K/V; cpu_ref_split reads these on the host

    // KV is tiled in kN0 (=64) columns.  Split-K partitions WHOLE tiles across
    // G ranges; T = ceil(num_tiles / G) tiles per range, so the last range may
    // be ragged and (for large G) some trailing ranges are empty (-inf LSE).
    const int kv_tile = kN0;                       // 64
    const int num_tiles = (Skv + kv_tile - 1) / kv_tile;

    // FULL-range reference (the invariant target): cpu_ref_split over [0,Skv).
    // Computed once per (h,row); independent of G.
    std::vector<float> full_o((size_t)Hq * Sq * kD);
    for (int h = 0; h < Hq; ++h)
        for (int row = 0; row < Sq; ++row)
            cpu_ref_split(p, bufs, 0, h, row, 0, Skv,
                          full_o.data() + ((size_t)h * Sq + row) * kD, nullptr);

    for (int G : kGSet) {
        const int T = (num_tiles + G - 1) / G;     // tiles per split (ceil)

        const size_t n_o = (size_t)B * Hq * Sq * kD;
        std::vector<float> scratch_o((size_t)G * n_o, 0.0f);
        std::vector<float> scratch_lse((size_t)G * B * Hq * Sq, 0.0f);

        // Build the G partials.  For split g, KV sub-range is
        //   [g*T*kv_tile, (g+1)*T*kv_tile)  clamped to [0,Skv).
        // cpu_ref_split returns o_row (normalized by THIS range's sum) + the
        // range's natural-log LSE; an empty range yields o=0, lse=-inf, which
        // the combine drops (weight 0) — exactly what we want for trailing
        // empty splits at large G.
        for (int h = 0; h < Hq; ++h) {
            for (int row = 0; row < Sq; ++row) {
                for (int g = 0; g < G; ++g) {
                    int kv0 = g * T * kv_tile;
                    int kv1 = (g + 1) * T * kv_tile;
                    if (kv0 > Skv) kv0 = Skv;       // fully-empty trailing split
                    if (kv1 > Skv) kv1 = Skv;
                    float lse_g = -INFINITY;
                    cpu_ref_split(p, bufs, 0, h, row, kv0, kv1,
                                  scratch_o.data() +
                                      scratch_o_idx(g, 0, h, row, 0, B, Hq, Sq),
                                  &lse_g);
                    scratch_lse[scratch_lse_idx(g, 0, h, row, B, Hq, Sq)] = lse_g;
                }
            }
        }

        std::vector<float> gpu_o = run_combine({G, B, Hq, Sq}, scratch_o, scratch_lse);

        std::string label = "real/G=" + std::to_string(G);
        // 2e-3 absorbs per-range BF16 P-truncation drift (T1 note); the fp32
        // combine reweighting itself is exact.
        compare_max_abs(gpu_o, full_o, label.c_str(), 2e-3f);
    }
}

int main(int argc, char** argv) {
    // Mirror the component tests' arg handling: strip any --golden* flags so the
    // gate's shared invocation (which passes them to all standalone bins) does
    // not confuse GoogleTest.  This suite has no golden dependency.
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--golden", 8) == 0) continue;
        rest.push_back(argv[i]);
    }
    int rc = static_cast<int>(rest.size());
    ::testing::InitGoogleTest(&rc, rest.data());
    return RUN_ALL_TESTS();
}
