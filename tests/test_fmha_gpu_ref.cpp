// =============================================================================
// TEST: GPU reference vs CPU reference (oracle-vs-oracle, no production kernel).
//
// New here? The full-kernel test (test_fmha_fwd_d64.cpp) checks the production
// kernel against the CPU reference, and separately uses the GPU reference to
// verify LSE. This file exists to keep those two ORACLES honest with each
// other: it runs gpu_ref_fmha_fwd (a naive fp32 attention kernel) and diffs it
// against cpu_ref_verify across every kAllFull config. If both oracles agree,
// we trust them as ground truth for the production kernel. Both follow the same
// bf16-truncation pipeline, so the only slack is fp32 accumulation order.
// =============================================================================
//
// GPU reference kernel correctness test.
//
// Verifies that the naive FP32 GPU reference kernel (gpu_ref_fmha_fwd)
// matches the CPU reference (cpu_ref_verify) across all D64 test configs.
// Both implementations follow the same BF16 truncation pipeline, so
// results should match within tight FP32 accumulation-order tolerance.

#include "test_configs.hpp"
#include "test_params.hpp"
#include "runner/buffers.hpp"
#include "runner/gpu_ref.hpp"
#include "runner/cpu_ref.hpp"
#include "runner/fmha_params.hpp"
#include "runner/bf16_utils.hpp"
#include "components_ref/cpu_ref_combine.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

class FmhaGpuRefTest : public ::testing::TestWithParam<TestCase> {};

TEST_P(FmhaGpuRefTest, MatchesCpuRef) {
    FmhaParams p = make_params(GetParam());
    FmhaBuffers bufs(p);
    bufs.fill_random(42);
    bufs.copy_to_device();

    const int Dpad = p.hdim_dispatch();
    const bool varlen_mode = !p.varlen_seqs.empty();

    GpuRefParams gp{};
    gp.d_Q = reinterpret_cast<const uint16_t*>(bufs.d_Q);
    gp.d_K = reinterpret_cast<const uint16_t*>(bufs.d_K);
    gp.d_V = reinterpret_cast<const uint16_t*>(bufs.d_V);
    gp.d_O = reinterpret_cast<uint16_t*>(bufs.d_O);
    gp.d_LSE = p.lse ? reinterpret_cast<float*>(bufs.d_LSE) : nullptr;
    gp.batch = p.batch; gp.q_heads = p.q_heads; gp.kv_heads = p.kv_heads; gp.gqa = p.gqa;
    gp.seq_len = p.seq_len; gp.kv_seq_len = p.kv_seq_len; gp.head_dim = p.head_dim;
    gp.stride_q_seq = Dpad; gp.stride_k_seq = Dpad;
    gp.stride_v_seq = Dpad; gp.stride_o_seq = Dpad;
    gp.stride_q_head = bufs.stride_q_head / 2; gp.stride_q_batch = bufs.stride_q_batch / 2;
    gp.stride_k_head = bufs.stride_k_head / 2; gp.stride_k_batch = bufs.stride_k_batch / 2;
    gp.stride_v_head = bufs.stride_k_head / 2; gp.stride_v_batch = bufs.stride_k_batch / 2;
    gp.stride_o_head = bufs.stride_q_head / 2; gp.stride_o_batch = bufs.stride_q_batch / 2;
    gp.mask = p.mask; gp.scalar = p.scalar();
    gp.d_seqstart_q = varlen_mode ? reinterpret_cast<const uint32_t*>(bufs.d_qseq) : nullptr;
    gp.d_seqstart_k = varlen_mode ? reinterpret_cast<const uint32_t*>(bufs.d_kseq) : nullptr;

    gpu_ref_fmha_fwd(gp);
    hipDeviceSynchronize();

    bufs.copy_from_device();
    CpuRefResult r = cpu_ref_verify(p, bufs);
    EXPECT_TRUE(r.pass) << "max_abs=" << r.max_abs << " max_rel=" << r.max_rel
                        << " min_cos=" << r.min_cos << " mismatch=" << r.mismatch;
}

INSTANTIATE_TEST_SUITE_P(Full, FmhaGpuRefTest,
    ::testing::ValuesIn(kAllFull), test_name);

// =============================================================================
// STAGE-1 TRUST: split-K GPU oracles vs CPU oracles at small S.
//
// gpu_ref_split and gpu_ref_combine become trustworthy ONLY by matching the CPU
// ground truth (cpu_ref_split / cpu_ref_combine) at small S, where the CPU refs
// are fast enough to run exhaustively.  Both sides are fp32 naive with identical
// bf16 truncation, so the tolerance is tight (1e-4).  The final case below also
// pins the LSE *value* against a HAND-computed logf(local_sum)+scaled_max on a
// 2-key input, so a convention error that fooled BOTH refs would still be caught
// (the A1 double-scale bug guard).
// =============================================================================

namespace {

// Build a GpuRefParams from FmhaBuffers/FmhaParams — identical stride derivation
// to the FmhaGpuRefTest body above (V reuses K strides; O reuses Q strides).
GpuRefParams make_gpu_ref_params(const FmhaParams& p, const FmhaBuffers& bufs) {
    const int Dpad = p.hdim_dispatch();
    const bool varlen_mode = !p.varlen_seqs.empty();
    GpuRefParams gp{};
    gp.d_Q = reinterpret_cast<const uint16_t*>(bufs.d_Q);
    gp.d_K = reinterpret_cast<const uint16_t*>(bufs.d_K);
    gp.d_V = reinterpret_cast<const uint16_t*>(bufs.d_V);
    gp.d_O = reinterpret_cast<uint16_t*>(bufs.d_O);    // unused by split kernel
    gp.d_LSE = nullptr;                                // unused by split kernel
    gp.batch = p.batch; gp.q_heads = p.q_heads; gp.kv_heads = p.kv_heads; gp.gqa = p.gqa;
    gp.seq_len = p.seq_len; gp.kv_seq_len = p.kv_seq_len; gp.head_dim = p.head_dim;
    gp.stride_q_seq = Dpad; gp.stride_k_seq = Dpad;
    gp.stride_v_seq = Dpad; gp.stride_o_seq = Dpad;
    gp.stride_q_head = bufs.stride_q_head / 2; gp.stride_q_batch = bufs.stride_q_batch / 2;
    gp.stride_k_head = bufs.stride_k_head / 2; gp.stride_k_batch = bufs.stride_k_batch / 2;
    gp.stride_v_head = bufs.stride_k_head / 2; gp.stride_v_batch = bufs.stride_k_batch / 2;
    gp.stride_o_head = bufs.stride_q_head / 2; gp.stride_o_batch = bufs.stride_q_batch / 2;
    gp.mask = p.mask; gp.scalar = p.scalar();
    gp.d_seqstart_q = varlen_mode ? reinterpret_cast<const uint32_t*>(bufs.d_qseq) : nullptr;
    gp.d_seqstart_k = varlen_mode ? reinterpret_cast<const uint32_t*>(bufs.d_kseq) : nullptr;
    return gp;
}

}  // namespace

// gpu_ref_split per-range vs cpu_ref_split, swept over S, G, and mask.
struct SplitTrustCase { int S; int G; int mask; };

class GpuRefSplitTrust : public ::testing::TestWithParam<SplitTrustCase> {};

TEST_P(GpuRefSplitTrust, MatchesCpuRefSplit) {
    const SplitTrustCase c = GetParam();

    // Small fixed-length problem: 1 batch, 2 q-heads, S query rows, D=64.
    TestCase tc{"split", 1, 2, c.S, 64, 0, 0, c.mask, 0, 0};
    FmhaParams p = make_params(tc);
    FmhaBuffers bufs(p);
    bufs.fill_random(123);
    bufs.copy_to_device();

    const int Dlog   = p.head_dim;
    const int Sq     = p.seq_len;
    const int Skv    = p.kv_seq_len;
    const int total_rows = p.batch * p.q_heads * Sq;

    // Even split: T = ceil(Skv / G); range g is [g*T, min((g+1)*T, Skv)).
    const int G = c.G;
    const int T = (Skv + G - 1) / G;

    GpuRefParams gp = make_gpu_ref_params(p, bufs);

    // Device + host scratch for one range's partials.
    float *d_o_g = nullptr, *d_lse_g = nullptr;
    ASSERT_EQ(hipMalloc(&d_o_g,   (size_t)total_rows * Dlog * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_lse_g, (size_t)total_rows * sizeof(float)), hipSuccess);
    std::vector<float> h_o_g((size_t)total_rows * Dlog);
    std::vector<float> h_lse_g(total_rows);

    double max_abs_o = 0.0, max_abs_lse = 0.0;
    std::vector<float> cpu_o(Dlog);

    for (int g = 0; g < G; ++g) {
        const int kv_start = g * T;
        const int kv_end   = std::min((g + 1) * T, Skv);

        gpu_ref_split(gp, kv_start, kv_end, d_o_g, d_lse_g);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(h_o_g.data(), d_o_g,
                            (size_t)total_rows * Dlog * sizeof(float),
                            hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(h_lse_g.data(), d_lse_g,
                            (size_t)total_rows * sizeof(float),
                            hipMemcpyDeviceToHost), hipSuccess);

        // Compare every row against cpu_ref_split for the same range.
        for (int b = 0; b < p.batch; ++b)
        for (int hq = 0; hq < p.q_heads; ++hq)
        for (int i = 0; i < Sq; ++i) {
            float cpu_lse = 0.0f;
            cpu_ref_split(p, bufs, b, hq, i, kv_start, kv_end, cpu_o.data(), &cpu_lse);

            const int row = (b * p.q_heads + hq) * Sq + i;  // flat thread-id order
            const float gpu_lse = h_lse_g[row];

            // Both empty/masked -> both -inf: agree, nothing finite to diff.
            const bool gpu_inf = std::isinf(gpu_lse) && gpu_lse < 0;
            const bool cpu_inf = std::isinf(cpu_lse) && cpu_lse < 0;
            ASSERT_EQ(gpu_inf, cpu_inf)
                << "LSE -inf mismatch S=" << c.S << " G=" << c.G << " mask=" << c.mask
                << " g=" << g << " row=" << row
                << " gpu=" << gpu_lse << " cpu=" << cpu_lse;
            if (gpu_inf) {
                // -inf range: O must be zero on both sides.
                for (int d = 0; d < Dlog; ++d) {
                    max_abs_o = std::max(max_abs_o, (double)std::fabs(h_o_g[(size_t)row * Dlog + d]));
                    max_abs_o = std::max(max_abs_o, (double)std::fabs(cpu_o[d]));
                }
                continue;
            }
            max_abs_lse = std::max(max_abs_lse, (double)std::fabs(gpu_lse - cpu_lse));
            for (int d = 0; d < Dlog; ++d)
                max_abs_o = std::max(max_abs_o,
                    (double)std::fabs(h_o_g[(size_t)row * Dlog + d] - cpu_o[d]));
        }
    }

    hipFree(d_o_g);
    hipFree(d_lse_g);

    EXPECT_LT(max_abs_o, 1e-4) << "O_g vs cpu_ref_split S=" << c.S
                               << " G=" << c.G << " mask=" << c.mask;
    EXPECT_LT(max_abs_lse, 1e-4) << "LSE_g vs cpu_ref_split S=" << c.S
                                 << " G=" << c.G << " mask=" << c.mask;
}

INSTANTIATE_TEST_SUITE_P(Stage1, GpuRefSplitTrust,
    ::testing::Values(
        SplitTrustCase{512, 1, 0},  SplitTrustCase{512, 1, 1},
        SplitTrustCase{512, 3, 0},  SplitTrustCase{512, 3, 1},
        SplitTrustCase{512, 8, 0},  SplitTrustCase{512, 8, 1},
        SplitTrustCase{2048, 1, 0}, SplitTrustCase{2048, 1, 1},
        SplitTrustCase{2048, 3, 0}, SplitTrustCase{2048, 3, 1},
        SplitTrustCase{2048, 8, 0}, SplitTrustCase{2048, 8, 1}),
    [](const testing::TestParamInfo<SplitTrustCase>& info) {
        return "S" + std::to_string(info.param.S) +
               "_G" + std::to_string(info.param.G) +
               "_mask" + std::to_string(info.param.mask);
    });

// gpu_ref_combine vs cpu_ref_combine on synthetic partials (some -inf planes).
TEST(GpuRefCombineTrust, MatchesCpuRefCombine) {
    const int G = 4;
    const int rows = 5;
    const int Dlog = 64;

    // Synthetic per-(range,row) LSE and partials.  Include -inf planes and an
    // all-inf row (row 4) so the empty-range / all-empty branches are exercised.
    std::vector<float> h_o_part((size_t)G * rows * Dlog);
    std::vector<float> h_lse_part((size_t)G * rows);
    for (int g = 0; g < G; ++g)
    for (int r = 0; r < rows; ++r) {
        // Row 4: every range -inf (all empty).  Range 1 of row 2: -inf.
        bool is_inf = (r == 4) || (g == 1 && r == 2);
        h_lse_part[(size_t)g * rows + r] =
            is_inf ? -INFINITY : (0.5f * g - 0.25f * r + 1.0f);
        for (int d = 0; d < Dlog; ++d)
            h_o_part[((size_t)g * rows + r) * Dlog + d] =
                is_inf ? 0.0f : (0.01f * (d + 1) + 0.1f * g - 0.05f * r);
    }

    float *d_o_part = nullptr, *d_lse_part = nullptr, *d_o_final = nullptr;
    ASSERT_EQ(hipMalloc(&d_o_part,   h_o_part.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_lse_part, h_lse_part.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_o_final,  (size_t)rows * Dlog * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_o_part, h_o_part.data(), h_o_part.size() * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_lse_part, h_lse_part.data(), h_lse_part.size() * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);

    gpu_ref_combine(G, rows, Dlog, d_o_part, d_lse_part, d_o_final);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<float> h_o_final((size_t)rows * Dlog);
    ASSERT_EQ(hipMemcpy(h_o_final.data(), d_o_final,
                        (size_t)rows * Dlog * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(d_o_part); hipFree(d_lse_part); hipFree(d_o_final);

    // cpu_ref_combine works on ONE row (o_part [G*Dlog], lse_part [G]); the GPU
    // layout is split-major [G*rows*Dlog], so gather per-row planes for the CPU.
    std::vector<float> cpu_o(Dlog), cpu_o_part((size_t)G * Dlog), cpu_lse(G);
    double max_abs = 0.0;
    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < G; ++g) {
            cpu_lse[g] = h_lse_part[(size_t)g * rows + r];
            for (int d = 0; d < Dlog; ++d)
                cpu_o_part[(size_t)g * Dlog + d] =
                    h_o_part[((size_t)g * rows + r) * Dlog + d];
        }
        cpu_ref_combine(G, Dlog, cpu_o_part.data(), cpu_lse.data(), cpu_o.data());
        for (int d = 0; d < Dlog; ++d)
            max_abs = std::max(max_abs,
                (double)std::fabs(h_o_final[(size_t)r * Dlog + d] - cpu_o[d]));
    }
    EXPECT_LT(max_abs, 1e-4) << "gpu_ref_combine vs cpu_ref_combine";
}

// ★ LSE-domain assertion (A1 carryover).  A tiny 2-key, D=2 single-range case
// where the LSE is HAND-computed in fp32 with the exact bf16 truncation.  We
// pin gpu_ref_split's LSE to BOTH cpu_ref_split AND the hand value: if a wrong
// convention (e.g. the `scalar*local_max` double-scale) fooled both refs, the
// hand value would still disagree and fail the test.
TEST(GpuRefSplitTrust, LseHandComputed) {
    // 1 batch, 1 q-head, 1 query row, 2 keys, head_dim=2 (within Dpad path).
    TestCase tc{"lse2", 1, 1, 1, 2, 0, /*kv_seq=*/2, /*mask=*/0, 0, 0};
    FmhaParams p = make_params(tc);
    FmhaBuffers bufs(p);

    const int Dpad = p.hdim_dispatch();   // == head_dim == 2 here
    const int Dlog = p.head_dim;          // 2

    // Hand-pick small bf16-representable values (whole/half numbers are exact in
    // bf16) so Q.K is exact and reproducible.
    //   Q   = [1.0, 2.0]
    //   K0  = [1.0, 0.0]   -> dot = 1.0
    //   K1  = [0.5, 1.0]   -> dot = 0.5 + 2.0 = 2.5
    //   V0  = [3.0, 4.0]   V1 = [5.0, 6.0]
    auto setq = [&](std::vector<uint16_t>& v, size_t off, float a, float b) {
        v[off + 0] = float_to_bf16(a);
        v[off + 1] = float_to_bf16(b);
    };
    // Q row 0 at offset 0; K/V rows 0,1 at offsets 0 and Dpad.
    setq(bufs.h_Q, 0, 1.0f, 2.0f);
    setq(bufs.h_K, 0,           1.0f, 0.0f);
    setq(bufs.h_K, (size_t)Dpad, 0.5f, 1.0f);
    setq(bufs.h_V, 0,           3.0f, 4.0f);
    setq(bufs.h_V, (size_t)Dpad, 5.0f, 6.0f);
    bufs.copy_to_device();

    const float scalar = p.scalar();   // 1/sqrt(2)

    // --- Hand-computed expected LSE over the FULL [0,2) range ---
    // Scores are scaled UP FRONT, then max/sum/lse are over the scaled scores;
    // LSE = logf(local_sum) + local_max, with local_max the SCALED max — NO
    // extra *scalar (that would be the A1 double-scale bug).
    const float q0 = bf16_to_float(float_to_bf16(1.0f));
    const float q1 = bf16_to_float(float_to_bf16(2.0f));
    const float dot0 = q0 * bf16_to_float(float_to_bf16(1.0f))
                     + q1 * bf16_to_float(float_to_bf16(0.0f));   // 1.0
    const float dot1 = q0 * bf16_to_float(float_to_bf16(0.5f))
                     + q1 * bf16_to_float(float_to_bf16(1.0f));   // 2.5
    const float s0 = dot0 * scalar;
    const float s1 = dot1 * scalar;
    const float local_max = std::max(s0, s1);
    const float local_sum = std::exp(s0 - local_max) + std::exp(s1 - local_max);
    const float hand_lse  = std::log(local_sum) + local_max;      // correct convention
    // The WRONG (A1) convention would be logf(sum)+scalar*local_max; record it
    // only to make the test's intent explicit (we assert we are NOT this).
    const float wrong_lse = std::log(local_sum) + scalar * local_max;

    const int total_rows = p.batch * p.q_heads * p.seq_len;   // 1
    GpuRefParams gp = make_gpu_ref_params(p, bufs);

    float *d_o_g = nullptr, *d_lse_g = nullptr;
    ASSERT_EQ(hipMalloc(&d_o_g,   (size_t)total_rows * Dlog * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_lse_g, (size_t)total_rows * sizeof(float)), hipSuccess);
    gpu_ref_split(gp, 0, 2, d_o_g, d_lse_g);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    float gpu_lse = 0.0f;
    ASSERT_EQ(hipMemcpy(&gpu_lse, d_lse_g, sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    hipFree(d_o_g); hipFree(d_lse_g);

    // CPU oracle LSE for the same range.
    std::vector<float> cpu_o(Dlog);
    float cpu_lse = 0.0f;
    cpu_ref_split(p, bufs, 0, 0, 0, 0, 2, cpu_o.data(), &cpu_lse);

    printf("[LseHandComputed] hand=%.8f  gpu=%.8f  cpu=%.8f  (wrong_A1=%.8f)\n",
           hand_lse, gpu_lse, cpu_lse, wrong_lse);

    // (a) gpu vs cpu agree to 1e-4.
    EXPECT_NEAR(gpu_lse, cpu_lse, 1e-4) << "gpu vs cpu LSE";
    // (b) BOTH match the hand-computed correct convention — catches a shared bug.
    EXPECT_NEAR(gpu_lse, hand_lse, 1e-4) << "gpu LSE vs hand-computed";
    EXPECT_NEAR(cpu_lse, hand_lse, 1e-4) << "cpu LSE vs hand-computed";
    // Sanity: the correct and A1-wrong conventions are actually distinguishable
    // here (so passing (b) is meaningful, not a degenerate equality).
    EXPECT_GT(std::fabs(hand_lse - wrong_lse), 1e-3) << "hand vs A1-wrong should differ";
}
