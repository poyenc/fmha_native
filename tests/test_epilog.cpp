#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "components_ref/ref_epilog.hpp"
#include "components_ref/ref_qk_gemm.hpp"
#include "components_ref/ref_softmax.hpp"
#include "components_ref/ref_pv_gemm.hpp"
#include "kernels/epilog.hpp"

static std::string g_golden_full;
static std::string g_golden_partial;

namespace {

struct EpilogResult {
    std::vector<float> o_final;  // 256*32 fp32
    std::vector<float> o_dram;   // seqlen_q*64 fp32 (bf16-promoted)
};

void run_kernel(const std::vector<float>& o_acc,
                const std::vector<float>& rsum,
                int seqlen_q, int stride_o,
                EpilogResult& res) {
    const int n_final = 256 * 32;
    const int n_dram = seqlen_q * 64;

    void *dOacc = nullptr, *dRsum = nullptr, *dOdram = nullptr, *dOfinal = nullptr;
    ASSERT_EQ(hipMalloc(&dOacc, o_acc.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRsum, rsum.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOdram, n_dram * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOfinal, n_final * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dOacc, o_acc.data(), o_acc.size() * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dRsum, rsum.data(), rsum.size() * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dOdram, 0, n_dram * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMemset(dOfinal, 0, n_final * sizeof(float)), hipSuccess);

    hipLaunchKernelGGL(test_epilog_kernel, dim3(1), dim3(256), 0, nullptr,
                       (const float*)dOacc, (const float*)dRsum,
                       (uint16_t*)dOdram, (float*)dOfinal,
                       seqlen_q, stride_o);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    res.o_final.resize(n_final);
    ASSERT_EQ(hipMemcpy(res.o_final.data(), dOfinal, n_final * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);

    // Read DRAM output as bf16, widen to fp32
    std::vector<uint16_t> dram_bf16(n_dram);
    ASSERT_EQ(hipMemcpy(dram_bf16.data(), dOdram, n_dram * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), hipSuccess);
    res.o_dram.resize(n_dram);
    for (int i = 0; i < n_dram; ++i)
        res.o_dram[i] = bf16_to_float(dram_bf16[i]);

    hipFree(dOacc); hipFree(dRsum); hipFree(dOdram); hipFree(dOfinal);
}

bool load_golden_slot(const std::string& dir, int slot, int regs,
                      std::vector<float>& out) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_reg.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    const int NT = 256, MR = 64;
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> all(bytes / 4);
    size_t got = fread(all.data(), 4, all.size(), f);
    fclose(f);
    if (got < static_cast<size_t>((slot * NT + NT - 1) * MR + regs)) return false;
    out.resize(NT * regs);
    for (int tid = 0; tid < NT; ++tid)
        for (int r = 0; r < regs; ++r)
            out[tid * regs + r] = all[(slot * NT + tid) * MR + r];
    return true;
}

bool load_golden_odram(const std::string& dir, std::vector<float>& out) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "o_dram.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(bytes / 4);
    size_t got = fread(out.data(), 4, out.size(), f);
    fclose(f);
    return got == out.size();
}

void compare_exact(const std::vector<float>& got, const std::vector<float>& exp,
                   const char* label) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [%s] mismatch i=%zu got=%.10f exp=%.10f\n",
                        label, i, got[i], exp[i]);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] mism=%d/%zu\n", label, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values differ";
}

void compare_tolerance(const std::vector<float>& got, const std::vector<float>& exp,
                       const char* label, float rtol = 1e-5f) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0;
    float maxrel = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float relerr = (std::fabs(exp[i]) > 1e-8f)
                     ? std::fabs(got[i] - exp[i]) / std::fabs(exp[i])
                     : std::fabs(got[i] - exp[i]);
        if (relerr > maxrel) maxrel = relerr;
        if (relerr > rtol) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [%s] mismatch i=%zu got=%.10f exp=%.10f rel=%.6g\n",
                        label, i, got[i], exp[i], relerr);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] maxrel=%.6g mism=%d/%zu\n", label, maxrel, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values exceed tolerance";
}

// Golden input formulas (match CK dump manifest)
std::vector<uint16_t> make_Q(int seqlen, int D) {
    std::vector<uint16_t> v(static_cast<size_t>(seqlen) * D);
    for (int i = 0; i < seqlen * D; ++i) v[i] = float_to_bf16((i % 256) / 256.0f);
    return v;
}
std::vector<uint16_t> make_K(int seqlen, int D) {
    std::vector<uint16_t> v(static_cast<size_t>(seqlen) * D);
    for (int i = 0; i < seqlen * D; ++i) v[i] = float_to_bf16(((i + 64) % 256) / 256.0f);
    return v;
}
std::vector<uint16_t> make_V(int seqlen, int D) {
    std::vector<uint16_t> v(static_cast<size_t>(seqlen) * D);
    for (int i = 0; i < seqlen * D; ++i) v[i] = float_to_bf16((i % 256) / 256.0f + 1.0f);
    return v;
}

// Compute O_acc + RSUM from scratch using verified refs (self-contained, no golden).
struct EpilogInput {
    std::vector<float> o_acc;  // 256*32
    std::vector<float> rsum;   // 256
};

EpilogInput compute_epilog_input(int sq, int sk, int D, float scale_s) {
    auto hQ = make_Q(sq, D);
    auto hK = make_K(sk, D);
    auto hV = make_V(sk, D);

    // GEMM0: S_acc
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(), D, sq, hK.data(), D, sk, s_acc.data());

    // Softmax: P + RSUM
    std::vector<float> p(256 * 32), rmax(256), rsum(256);
    ref_softmax(s_acc.data(), sk, 0, scale_s, p.data(), rmax.data(), rsum.data());

    // GEMM1: O_acc
    std::vector<float> o_acc(kPVOutElems);
    ref_pv_gemm(p.data(), hV.data(), D, sk, o_acc.data());

    return {std::move(o_acc), std::move(rsum)};
}

// Compute RSUM from golden S_ACC using ref_softmax (for golden tests only).
std::vector<float> compute_rsum_from_sacc(const std::vector<float>& sacc,
                                          int seqlen_k, float scale_s) {
    std::vector<float> p(256 * 32), rmax(256), rsum(256);
    ref_softmax(sacc.data(), seqlen_k, 0, scale_s, p.data(), rmax.data(), rsum.data());
    return rsum;
}

} // namespace

TEST(EpilogFullTile, MatchesCpuRef) {
    const int sq = 64, sk = 64, D = 64;
    const float scale_s = 0.125f;
    auto [o_acc, rsum] = compute_epilog_input(sq, sk, D, scale_s);

    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);

    std::vector<float> exp_final(256 * 32), exp_dram(sq * D);
    ref_epilog(o_acc.data(), rsum.data(), sq, exp_final.data(), exp_dram.data());

    compare_tolerance(res.o_final, exp_final, "full/cpuref/o_final");
    compare_exact(res.o_dram, exp_dram, "full/cpuref/o_dram");
}

TEST(EpilogFullTile, MatchesGolden) {
    if (g_golden_full.empty()) GTEST_SKIP() << "no --golden-full dir";
    const int sq = 64, sk = 64, D = 64;
    const float scale_s = 0.125f;

    std::vector<float> o_acc;
    ASSERT_TRUE(load_golden_slot(g_golden_full, 5, 32, o_acc));
    std::vector<float> sacc;
    ASSERT_TRUE(load_golden_slot(g_golden_full, 1, 32, sacc));
    auto rsum = compute_rsum_from_sacc(sacc, sk, scale_s);

    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);

    // O_final vs golden slot 6
    std::vector<float> golden_ofinal;
    ASSERT_TRUE(load_golden_slot(g_golden_full, 6, 32, golden_ofinal));
    compare_tolerance(res.o_final, golden_ofinal, "full/golden/o_final");

    // o_dram vs golden o_dram.bin
    std::vector<float> golden_odram;
    ASSERT_TRUE(load_golden_odram(g_golden_full, golden_odram));
    compare_exact(res.o_dram, golden_odram, "full/golden/o_dram");
}

TEST(EpilogPartialTile, MatchesCpuRef) {
    const int sq = 17, sk = 33, D = 64;
    const float scale_s = 0.125f;
    auto [o_acc, rsum] = compute_epilog_input(sq, sk, D, scale_s);

    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);

    std::vector<float> exp_final(256 * 32), exp_dram(sq * D);
    ref_epilog(o_acc.data(), rsum.data(), sq, exp_final.data(), exp_dram.data());

    compare_tolerance(res.o_final, exp_final, "partial/cpuref/o_final");
    compare_exact(res.o_dram, exp_dram, "partial/cpuref/o_dram");
}

TEST(EpilogPartialTile, MatchesGolden) {
    if (g_golden_partial.empty()) GTEST_SKIP() << "no --golden-partial dir";
    const int sq = 17, sk = 33, D = 64;
    const float scale_s = 0.125f;

    std::vector<float> o_acc;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, 5, 32, o_acc));
    std::vector<float> sacc;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, 1, 32, sacc));
    auto rsum = compute_rsum_from_sacc(sacc, sk, scale_s);

    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);

    std::vector<float> golden_ofinal;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, 6, 32, golden_ofinal));
    compare_tolerance(res.o_final, golden_ofinal, "partial/golden/o_final");

    std::vector<float> golden_odram;
    ASSERT_TRUE(load_golden_odram(g_golden_partial, golden_odram));
    compare_exact(res.o_dram, golden_odram, "partial/golden/o_dram");
}

// ---- Edge case tests (CPU ref only) ----

TEST(EpilogEdge, MinTile) {
    const int sq=1, sk=1, D=64; const float scale_s=0.125f;
    auto [o_acc, rsum] = compute_epilog_input(sq, sk, D, scale_s);
    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);
    std::vector<float> exp_final(256*32), exp_dram(sq*D);
    ref_epilog(o_acc.data(), rsum.data(), sq, exp_final.data(), exp_dram.data());
    compare_tolerance(res.o_final, exp_final, "edge/min/o_final");
    compare_exact(res.o_dram, exp_dram, "edge/min/o_dram");
}

TEST(EpilogEdge, FullM0) {
    const int sq=128, sk=64, D=64; const float scale_s=0.125f;
    auto [o_acc, rsum] = compute_epilog_input(sq, sk, D, scale_s);
    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);
    std::vector<float> exp_final(256*32), exp_dram(sq*D);
    ref_epilog(o_acc.data(), rsum.data(), sq, exp_final.data(), exp_dram.data());
    compare_tolerance(res.o_final, exp_final, "edge/fullM0/o_final");
    compare_exact(res.o_dram, exp_dram, "edge/fullM0/o_dram");
}

TEST(EpilogEdge, SingleKCol) {
    const int sq=64, sk=1, D=64; const float scale_s=0.125f;
    auto [o_acc, rsum] = compute_epilog_input(sq, sk, D, scale_s);
    EpilogResult res;
    run_kernel(o_acc, rsum, sq, D, res);
    std::vector<float> exp_final(256*32), exp_dram(sq*D);
    ref_epilog(o_acc.data(), rsum.data(), sq, exp_final.data(), exp_dram.data());
    compare_tolerance(res.o_final, exp_final, "edge/singleK/o_final");
    compare_exact(res.o_dram, exp_dram, "edge/singleK/o_dram");
}

int main(int argc, char** argv) {
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--golden-full=", 14) == 0)
            g_golden_full = argv[i] + 14;
        else if (std::strncmp(argv[i], "--golden-partial=", 17) == 0)
            g_golden_partial = argv[i] + 17;
        else if (std::strncmp(argv[i], "--golden=", 9) == 0)
            { g_golden_full = argv[i] + 9; g_golden_partial = argv[i] + 9; }
        else
            rest.push_back(argv[i]);
    }
    int rc = static_cast<int>(rest.size());
    ::testing::InitGoogleTest(&rc, rest.data());
    return RUN_ALL_TESTS();
}
