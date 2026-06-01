#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "components_ref/ref_softmax.hpp"
#include "components_ref/ref_qk_gemm.hpp"
#include "components/softmax.hpp"

static std::string g_golden_full;
static std::string g_golden_partial;

namespace {

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

struct SoftmaxResult {
    std::vector<float> p;      // 256*32
    std::vector<float> rmax;   // 256
    std::vector<float> rsum;   // 256
};

void run_kernel(const std::vector<float>& s_acc,
                int seqlen_k, int kv_offset, float scale_s,
                SoftmaxResult& res) {
    const int p_size = 256 * 32;
    const int scalar_size = 256;

    void *dSacc = nullptr, *dP = nullptr, *dRmax = nullptr, *dRsum = nullptr;
    ASSERT_EQ(hipMalloc(&dSacc, s_acc.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dP, p_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRmax, scalar_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRsum, scalar_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dSacc, s_acc.data(), s_acc.size() * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dP, 0, p_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemset(dRmax, 0, scalar_size * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemset(dRsum, 0, scalar_size * sizeof(float)), hipSuccess);

    hipLaunchKernelGGL(test_softmax_kernel, dim3(1), dim3(256), 0, nullptr,
                       (const float*)dSacc, (float*)dP, (float*)dRmax, (float*)dRsum,
                       seqlen_k, kv_offset, scale_s);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    res.p.resize(p_size);
    res.rmax.resize(scalar_size);
    res.rsum.resize(scalar_size);
    ASSERT_EQ(hipMemcpy(res.p.data(), dP, p_size * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(res.rmax.data(), dRmax, scalar_size * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(res.rsum.data(), dRsum, scalar_size * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(dSacc); hipFree(dP); hipFree(dRmax); hipFree(dRsum);
}

// Load golden dump_reg slot.
bool load_golden_slot(const std::string& dir, int slot, int regs_per_thread,
                      std::vector<float>& out) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_reg.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    const int NT = 256, MR = 64;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> all(bytes / 4);
    size_t got = fread(all.data(), 4, all.size(), f);
    fclose(f);
    if (got < static_cast<size_t>((slot * NT + NT - 1) * MR + regs_per_thread)) {
        fprintf(stderr, "  golden too small\n");
        return false;
    }
    out.resize(NT * regs_per_thread);
    for (int tid = 0; tid < NT; ++tid)
        for (int r = 0; r < regs_per_thread; ++r)
            out[tid * regs_per_thread + r] = all[(slot * NT + tid) * MR + r];
    return true;
}

void compare_exact(const std::vector<float>& got, const std::vector<float>& exp,
                   const char* label) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            ++mism;
            if (shown < 10) {
                int tid = i / (got.size() == 256 ? 1 : 32);
                int r = i % (got.size() == 256 ? 1 : 32);
                fprintf(stderr, "  [%s] mismatch tid=%d r=%d got=%.6f exp=%.6f\n",
                        label, tid, r, got[i], exp[i]);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] mism=%d/%zu\n", label, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values differ";
}

void compare_tolerance(const std::vector<float>& got, const std::vector<float>& exp,
                       const char* label, float atol = 1e-5f) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0;
    float maxabs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::fabs(got[i] - exp[i]);
        if (diff > maxabs) maxabs = diff;
        if (diff > atol) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [%s] mismatch i=%zu got=%.6f exp=%.6f diff=%.6g\n",
                        label, i, got[i], exp[i], diff);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] maxabs=%.6g mism=%d/%zu\n", label, maxabs, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values exceed tolerance";
}

} // namespace

// ---- Full tile tests (sq=64, sk=64) ----

TEST(SoftmaxFullTile, MatchesCpuRef) {
    const int sq = 64, sk = 64, D = 64;
    const float scale_s = 0.125f;
    auto hQ = make_Q(sq, D);
    auto hK = make_K(sk, D);
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(), D, sq, hK.data(), D, sk, s_acc.data());

    SoftmaxResult res;
    run_kernel(s_acc, sk, /*kv_offset=*/0, scale_s, res);

    std::vector<float> exp_p(256 * 32), exp_rmax(256), exp_rsum(256);
    ref_softmax(s_acc.data(), sk, 0, scale_s, exp_p.data(), exp_rmax.data(), exp_rsum.data());

    compare_exact(res.p, exp_p, "full/cpuref/P");
    compare_exact(res.rmax, exp_rmax, "full/cpuref/rmax");
    compare_tolerance(res.rsum, exp_rsum, "full/cpuref/rsum");
}

TEST(SoftmaxFullTile, MatchesGolden) {
    if (g_golden_full.empty()) GTEST_SKIP() << "no --golden-full dir";
    const int sk = 64;
    const float scale_s = 0.125f;

    // Use golden S_ACC as input
    std::vector<float> golden_sacc;
    ASSERT_TRUE(load_golden_slot(g_golden_full, /*slot=*/1, /*regs=*/32, golden_sacc));

    SoftmaxResult res;
    run_kernel(golden_sacc, sk, 0, scale_s, res);

    // Compare P against golden slot 3
    std::vector<float> golden_p;
    ASSERT_TRUE(load_golden_slot(g_golden_full, /*slot=*/3, /*regs=*/32, golden_p));
    compare_exact(res.p, golden_p, "full/golden/P");

    // Compare rmax against golden slot 2
    std::vector<float> golden_rmax;
    ASSERT_TRUE(load_golden_slot(g_golden_full, /*slot=*/2, /*regs=*/1, golden_rmax));
    compare_exact(res.rmax, golden_rmax, "full/golden/rmax");
}

// ---- Partial tile tests (sq=17, sk=33) ----

TEST(SoftmaxPartialTile, MatchesCpuRef) {
    const int sq = 17, sk = 33, D = 64;
    const float scale_s = 0.125f;
    auto hQ = make_Q(sq, D);
    auto hK = make_K(sk, D);
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(), D, sq, hK.data(), D, sk, s_acc.data());

    SoftmaxResult res;
    run_kernel(s_acc, sk, 0, scale_s, res);

    std::vector<float> exp_p(256 * 32), exp_rmax(256), exp_rsum(256);
    ref_softmax(s_acc.data(), sk, 0, scale_s, exp_p.data(), exp_rmax.data(), exp_rsum.data());

    compare_exact(res.p, exp_p, "partial/cpuref/P");
    compare_exact(res.rmax, exp_rmax, "partial/cpuref/rmax");
    compare_tolerance(res.rsum, exp_rsum, "partial/cpuref/rsum");
}

TEST(SoftmaxPartialTile, MatchesGolden) {
    if (g_golden_partial.empty()) GTEST_SKIP() << "no --golden-partial dir";
    const int sk = 33;
    const float scale_s = 0.125f;

    std::vector<float> golden_sacc;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, /*slot=*/1, /*regs=*/32, golden_sacc));

    SoftmaxResult res;
    run_kernel(golden_sacc, sk, 0, scale_s, res);

    std::vector<float> golden_p;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, /*slot=*/3, /*regs=*/32, golden_p));
    compare_exact(res.p, golden_p, "partial/golden/P");

    std::vector<float> golden_rmax;
    ASSERT_TRUE(load_golden_slot(g_golden_partial, /*slot=*/2, /*regs=*/1, golden_rmax));
    compare_exact(res.rmax, golden_rmax, "partial/golden/rmax");
}

// ---- Edge case tests (CPU ref only) ----

TEST(SoftmaxEdge, MinTile) {
    const int sq=1, sk=1, D=64; const float scale_s=0.125f;
    auto hQ=make_Q(sq,D); auto hK=make_K(sk,D);
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(),D,sq,hK.data(),D,sk,s_acc.data());
    SoftmaxResult res;
    run_kernel(s_acc, sk, 0, scale_s, res);
    std::vector<float> exp_p(256*32), exp_rmax(256), exp_rsum(256);
    ref_softmax(s_acc.data(), sk, 0, scale_s, exp_p.data(), exp_rmax.data(), exp_rsum.data());
    compare_exact(res.p, exp_p, "edge/min/P");
    compare_exact(res.rmax, exp_rmax, "edge/min/rmax");
}

TEST(SoftmaxEdge, FullM0) {
    const int sq=128, sk=64, D=64; const float scale_s=0.125f;
    auto hQ=make_Q(sq,D); auto hK=make_K(sk,D);
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(),D,sq,hK.data(),D,sk,s_acc.data());
    SoftmaxResult res;
    run_kernel(s_acc, sk, 0, scale_s, res);
    std::vector<float> exp_p(256*32), exp_rmax(256), exp_rsum(256);
    ref_softmax(s_acc.data(), sk, 0, scale_s, exp_p.data(), exp_rmax.data(), exp_rsum.data());
    compare_exact(res.p, exp_p, "edge/fullM0/P");
    compare_exact(res.rmax, exp_rmax, "edge/fullM0/rmax");
}

TEST(SoftmaxEdge, SingleKCol) {
    const int sq=64, sk=1, D=64; const float scale_s=0.125f;
    auto hQ=make_Q(sq,D); auto hK=make_K(sk,D);
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(),D,sq,hK.data(),D,sk,s_acc.data());
    SoftmaxResult res;
    run_kernel(s_acc, sk, 0, scale_s, res);
    std::vector<float> exp_p(256*32), exp_rmax(256), exp_rsum(256);
    ref_softmax(s_acc.data(), sk, 0, scale_s, exp_p.data(), exp_rmax.data(), exp_rsum.data());
    compare_exact(res.p, exp_p, "edge/singleK/P");
    compare_exact(res.rmax, exp_rmax, "edge/singleK/rmax");
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
