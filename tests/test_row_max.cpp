#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "refs/ref_row_max.hpp"
#include "refs/ref_qk_gemm.hpp"
#include "kernels/row_max.hpp"

static std::string g_golden_full;
static std::string g_golden_partial;

namespace {

// Golden input formulas (match CK dump manifest):
//   q[i] = (i % 256) / 256
//   k[i] = ((i + 64) % 256) / 256
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

// Compute S_acc on CPU using the verified GEMM0 reference.
std::vector<float> compute_s_acc(const std::vector<uint16_t>& hQ, int sq,
                                 const std::vector<uint16_t>& hK, int sk, int D) {
    std::vector<float> s_acc(kQKOutElems);
    ref_qk_gemm(hQ.data(), D, sq, hK.data(), D, sk, s_acc.data());
    return s_acc;
}

// Run the row_max GPU kernel. Input: S_acc (256*32 fp32). Output: rmax (256 fp32).
void run_kernel(const std::vector<float>& s_acc, std::vector<float>& hRmax) {
    hRmax.resize(kRowMaxOutElems);
    void *dSacc = nullptr, *dRmax = nullptr;
    ASSERT_EQ(hipMalloc(&dSacc, s_acc.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRmax, kRowMaxOutElems * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(dSacc, s_acc.data(), s_acc.size() * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dRmax, 0, kRowMaxOutElems * sizeof(float)), hipSuccess);

    hipLaunchKernelGGL(test_row_max_kernel, dim3(1), dim3(256), 0, nullptr,
                       reinterpret_cast<const float*>(dSacc),
                       reinterpret_cast<float*>(dRmax));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hRmax.data(), dRmax, kRowMaxOutElems * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(dSacc);
    hipFree(dRmax);
}

// Load golden RMAX (dump_reg slot 2, 1 f32 reg per thread).
bool load_golden_rmax(const std::string& dir, std::vector<float>& rmax) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_reg.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    const int NT = 256, MR = 64, SLOT = 2;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> all(bytes / 4);
    size_t got = fread(all.data(), 4, all.size(), f);
    fclose(f);
    if (got < static_cast<size_t>((SLOT * NT + NT - 1) * MR + 1)) {
        fprintf(stderr, "  golden too small\n");
        return false;
    }
    rmax.resize(NT);
    for (int tid = 0; tid < NT; ++tid)
        rmax[tid] = all[(SLOT * NT + tid) * MR + 0];
    return true;
}

// Load golden S_ACC (dump_reg slot 1, 32 f32 regs per thread).
bool load_golden_sacc(const std::string& dir, std::vector<float>& sacc) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_reg.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    const int NT = 256, MR = 64, SLOT = 1;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> all(bytes / 4);
    size_t got = fread(all.data(), 4, all.size(), f);
    fclose(f);
    if (got < static_cast<size_t>((SLOT * NT + NT - 1) * MR + 32)) {
        fprintf(stderr, "  golden too small\n");
        return false;
    }
    sacc.resize(NT * 32);
    for (int tid = 0; tid < NT; ++tid)
        for (int r = 0; r < 32; ++r)
            sacc[tid * 32 + r] = all[(SLOT * NT + tid) * MR + r];
    return true;
}

// Exact fp32 compare for row_max (scalar fmaxf, no FMA rounding).
void compare_exact(const std::vector<float>& got, const std::vector<float>& exp,
                   const char* label) {
    ASSERT_EQ(got.size(), exp.size());
    int mism = 0, shown = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != exp[i]) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [%s] mismatch tid=%zu got=%.6f exp=%.6f\n",
                        label, i, got[i], exp[i]);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] mism=%d/%zu\n", label, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values differ";
}

} // namespace

// ---- Full tile tests (sq=64, sk=64) ----

TEST(RowMaxFullTile, MatchesCpuRef) {
    const int sq = 64, sk = 64, D = 64;
    auto hQ = make_Q(sq, D);
    auto hK = make_K(sk, D);
    auto s_acc = compute_s_acc(hQ, sq, hK, sk, D);
    std::vector<float> got;
    run_kernel(s_acc, got);

    std::vector<float> exp(kRowMaxOutElems);
    ref_row_max(s_acc.data(), exp.data());
    compare_exact(got, exp, "full/cpuref");
}

TEST(RowMaxFullTile, MatchesGolden) {
    if (g_golden_full.empty()) GTEST_SKIP() << "no --golden-full dir";
    // Use golden S_ACC as input (not CPU-computed S_ACC)
    std::vector<float> golden_sacc;
    ASSERT_TRUE(load_golden_sacc(g_golden_full, golden_sacc));
    std::vector<float> got;
    run_kernel(golden_sacc, got);

    std::vector<float> golden_rmax;
    ASSERT_TRUE(load_golden_rmax(g_golden_full, golden_rmax));
    compare_exact(got, golden_rmax, "full/golden");
}

// ---- Partial tile tests (sq=17, sk=33) ----

TEST(RowMaxPartialTile, MatchesCpuRef) {
    const int sq = 17, sk = 33, D = 64;
    auto hQ = make_Q(sq, D);
    auto hK = make_K(sk, D);
    auto s_acc = compute_s_acc(hQ, sq, hK, sk, D);
    std::vector<float> got;
    run_kernel(s_acc, got);

    std::vector<float> exp(kRowMaxOutElems);
    ref_row_max(s_acc.data(), exp.data());
    compare_exact(got, exp, "partial/cpuref");
}

TEST(RowMaxPartialTile, MatchesGolden) {
    if (g_golden_partial.empty()) GTEST_SKIP() << "no --golden-partial dir";
    std::vector<float> golden_sacc;
    ASSERT_TRUE(load_golden_sacc(g_golden_partial, golden_sacc));
    std::vector<float> got;
    run_kernel(golden_sacc, got);

    std::vector<float> golden_rmax;
    ASSERT_TRUE(load_golden_rmax(g_golden_partial, golden_rmax));
    compare_exact(got, golden_rmax, "partial/golden");
}

// ---- Edge case tests (CPU ref only) ----

TEST(RowMaxEdge, MinTile) {
    const int sq=1, sk=1, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto s_acc=compute_s_acc(hQ,sq,hK,sk,D);
    std::vector<float> got;
    run_kernel(s_acc, got);
    std::vector<float> exp(kRowMaxOutElems);
    ref_row_max(s_acc.data(), exp.data());
    compare_exact(got, exp, "edge/min");
}

TEST(RowMaxEdge, FullM0) {
    const int sq=128, sk=64, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto s_acc=compute_s_acc(hQ,sq,hK,sk,D);
    std::vector<float> got;
    run_kernel(s_acc, got);
    std::vector<float> exp(kRowMaxOutElems);
    ref_row_max(s_acc.data(), exp.data());
    compare_exact(got, exp, "edge/fullM0");
}

TEST(RowMaxEdge, SingleKCol) {
    const int sq=64, sk=1, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto s_acc=compute_s_acc(hQ,sq,hK,sk,D);
    std::vector<float> got;
    run_kernel(s_acc, got);
    std::vector<float> exp(kRowMaxOutElems);
    ref_row_max(s_acc.data(), exp.data());
    compare_exact(got, exp, "edge/singleK");
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
