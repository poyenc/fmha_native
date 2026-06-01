#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "components_ref/ref_pv_gemm.hpp"
#include "components_ref/ref_qk_gemm.hpp"
#include "components_ref/ref_softmax.hpp"
#include "components/pv_gemm.hpp"

static std::string g_golden_full;
static std::string g_golden_partial;

namespace {

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

// Compute P from S_acc using ref_softmax (returns bf16-promoted fp32)
std::vector<float> compute_p(int sq, int sk, int D) {
    auto hQ = make_Q(sq, D);
    auto hK = make_K(sk, D);
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(), D, sq, hK.data(), D, sk, s_acc.data());
    std::vector<float> p(256 * 32), rmax(256), rsum(256);
    ref_softmax(s_acc.data(), sk, 0, 0.125f, p.data(), rmax.data(), rsum.data());
    return p;
}

void run_kernel(const std::vector<float>& p,
                const std::vector<uint16_t>& hV,
                int stride_v, int seqlen_k,
                std::vector<float>& hOut) {
    hOut.resize(kPVOutElems);
    void *dP = nullptr, *dV = nullptr, *dOut = nullptr;
    ASSERT_EQ(hipMalloc(&dP, p.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dV, hV.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dOut, kPVOutElems * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dP, p.data(), p.size() * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dV, hV.data(), hV.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dOut, 0, kPVOutElems * sizeof(float)), hipSuccess);

    hipLaunchKernelGGL(test_pv_gemm_kernel, dim3(1), dim3(256), 0, nullptr,
                       (const float*)dP, (const uint16_t*)dV, (float*)dOut,
                       stride_v, seqlen_k);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hOut.data(), dOut, kPVOutElems * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(dP); hipFree(dV); hipFree(dOut);
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

void compare(const std::vector<float>& got, const std::vector<float>& exp,
             const char* label, float rtol = 2e-3f, float atol = 1e-2f) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0; float maxabs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float abserr = std::fabs(got[i] - exp[i]);
        float relerr = (std::fabs(exp[i]) > 1e-6f) ? abserr / std::fabs(exp[i]) : abserr;
        if (abserr > maxabs) maxabs = abserr;
        if (abserr > atol && relerr > rtol) {
            ++mism;
            if (shown < 10) {
                int tid = i / 32, r = i % 32;
                fprintf(stderr, "  [%s] mismatch tid=%d r=%d got=%.6f exp=%.6f abs=%.6f\n",
                        label, tid, r, got[i], exp[i], abserr);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] maxabs=%.6g mism=%d/%zu\n", label, maxabs, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values exceed tolerance";
}

void compare_exact(const std::vector<float>& got, const std::vector<float>& exp,
                   const char* label) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            ++mism;
            if (shown < 10) {
                int tid = i / 32, r = i % 32;
                fprintf(stderr, "  [%s] mismatch tid=%d r=%d got=%.6f exp=%.6f\n",
                        label, tid, r, got[i], exp[i]);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] mism=%d/%zu\n", label, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values differ";
}

} // namespace

TEST(PvGemmFullTile, MatchesCpuRef) {
    const int sq = 64, sk = 64, D = 64;
    auto p = compute_p(sq, sk, D);
    auto hV = make_V(sk, D);
    std::vector<float> got;
    run_kernel(p, hV, D, sk, got);
    std::vector<float> exp(kPVOutElems);
    ref_pv_gemm(p.data(), hV.data(), D, sk, exp.data());
    compare(got, exp, "full/cpuref");
}

TEST(PvGemmFullTile, MatchesGolden) {
    if (g_golden_full.empty()) GTEST_SKIP() << "no --golden-full dir";
    const int sk = 64, D = 64;
    std::vector<float> golden_p;
    ASSERT_TRUE(load_golden_slot(g_golden_full, 3, 32, golden_p));
    auto hV = make_V(sk, D);
    std::vector<float> got;
    run_kernel(golden_p, hV, D, sk, got);
    std::vector<float> golden_oacc;
    ASSERT_TRUE(load_golden_slot(g_golden_full, 5, 32, golden_oacc));
    compare_exact(got, golden_oacc, "full/golden");
}

TEST(PvGemmPartialTile, MatchesCpuRef) {
    const int sq = 17, sk = 33, D = 64;
    auto p = compute_p(sq, sk, D);
    auto hV = make_V(sk, D);
    std::vector<float> got;
    run_kernel(p, hV, D, sk, got);
    std::vector<float> exp(kPVOutElems);
    ref_pv_gemm(p.data(), hV.data(), D, sk, exp.data());
    compare(got, exp, "partial/cpuref");
}

TEST(PvGemmPartialTile, MatchesGolden) {
    if (g_golden_partial.empty()) GTEST_SKIP() << "no --golden-partial dir";
    const int sk = 33, D = 64;
    std::vector<float> golden_p;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, 3, 32, golden_p));
    auto hV = make_V(sk, D);
    std::vector<float> got;
    run_kernel(golden_p, hV, D, sk, got);
    std::vector<float> golden_oacc;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, 5, 32, golden_oacc));
    compare_exact(got, golden_oacc, "partial/golden");
}

// ---- Edge case tests (CPU ref only) ----

TEST(PvGemmEdge, MinTile) {
    const int sq=1, sk=1, D=64;
    auto p=compute_p(sq,sk,D); auto hV=make_V(sk,D);
    std::vector<float> got;
    run_kernel(p,hV,D,sk,got);
    std::vector<float> exp(kPVOutElems);
    ref_pv_gemm(p.data(),hV.data(),D,sk,exp.data());
    compare(got,exp,"edge/min");
}

TEST(PvGemmEdge, FullM0) {
    const int sq=128, sk=64, D=64;
    auto p=compute_p(sq,sk,D); auto hV=make_V(sk,D);
    std::vector<float> got;
    run_kernel(p,hV,D,sk,got);
    std::vector<float> exp(kPVOutElems);
    ref_pv_gemm(p.data(),hV.data(),D,sk,exp.data());
    compare(got,exp,"edge/fullM0");
}

TEST(PvGemmEdge, SingleKCol) {
    const int sq=64, sk=1, D=64;
    auto p=compute_p(sq,sk,D); auto hV=make_V(sk,D);
    std::vector<float> got;
    run_kernel(p,hV,D,sk,got);
    std::vector<float> exp(kPVOutElems);
    ref_pv_gemm(p.data(),hV.data(),D,sk,exp.data());
    compare(got,exp,"edge/singleK");
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
