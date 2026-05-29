#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "refs/ref_qk_gemm.hpp"
#include "kernels/qk_gemm.hpp"

static std::string g_golden_dir;

namespace {

// Golden input formulas (match the CK dump manifest):
//   q[i] = (i % 256) / 256       i = linear index into [seqlen, D] contiguous
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

std::vector<float> run_kernel(const std::vector<uint16_t>& hQ, int sq,
                              const std::vector<uint16_t>& hK, int sk, int D) {
    void *dQ=nullptr, *dK=nullptr, *dOut=nullptr;
    EXPECT_EQ(hipMalloc(&dQ, hQ.size()*2), hipSuccess);
    EXPECT_EQ(hipMalloc(&dK, hK.size()*2), hipSuccess);
    EXPECT_EQ(hipMalloc(&dOut, kQKOutElems*sizeof(float)), hipSuccess);
    EXPECT_EQ(hipMemcpy(dQ, hQ.data(), hQ.size()*2, hipMemcpyHostToDevice), hipSuccess);
    EXPECT_EQ(hipMemcpy(dK, hK.data(), hK.size()*2, hipMemcpyHostToDevice), hipSuccess);
    EXPECT_EQ(hipMemset(dOut, 0, kQKOutElems*sizeof(float)), hipSuccess);

    hipLaunchKernelGGL(test_qk_gemm_kernel, dim3(1), dim3(256), 0, nullptr,
                       reinterpret_cast<const uint16_t*>(dQ),
                       reinterpret_cast<const uint16_t*>(dK),
                       reinterpret_cast<float*>(dOut),
                       /*stride_q=*/D, /*stride_k=*/D, sq, sk);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<float> hOut(kQKOutElems);
    EXPECT_EQ(hipMemcpy(hOut.data(), dOut, kQKOutElems*sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    hipFree(dQ); hipFree(dK); hipFree(dOut);
    return hOut;
}

// Load golden dump_reg slot 1 (S_ACC): reg[(slot*256+tid)*64 + r], MAX_REGS=64.
bool load_golden_sacc(const std::string& dir, std::vector<float>& sacc /*256*32*/) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_reg.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  cannot open %s\n", path.c_str()); return false; }
    const int NT=256, MR=64, SLOT=1;
    std::vector<float> all;
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    all.resize(bytes/4);
    size_t got = fread(all.data(), 4, all.size(), f);
    fclose(f);
    if (got < static_cast<size_t>((SLOT*NT+255)*MR + 64)) { fprintf(stderr,"  golden too small\n"); return false; }
    sacc.resize(NT*32);
    for (int tid=0; tid<NT; ++tid)
        for (int r=0; r<32; ++r)
            sacc[tid*32+r] = all[(SLOT*NT+tid)*MR + r];
    return true;
}

// Compare kernel vs reference values. Values are fp32 sums of bf16 products;
// MFMA accumulation order may differ from the reference by ULPs, so use a small
// relative tolerance. Structural (layout) errors produce large diffs.
void compare(const std::vector<float>& got, const std::vector<float>& exp,
             const char* label, float rtol = 2e-3f, float atol = 1e-2f) {
    ASSERT_EQ(got.size(), exp.size());
    int mism=0, shown=0; float maxabs=0, maxrel=0;
    for (size_t i=0;i<got.size();++i) {
        float a=got[i], e=exp[i];
        float abserr = std::fabs(a-e);
        float relerr = (std::fabs(e)>1e-6f) ? abserr/std::fabs(e) : abserr;
        if (abserr>maxabs) maxabs=abserr;
        if (relerr>maxrel) maxrel=relerr;
        if (abserr > atol && relerr > rtol) {
            ++mism;
            if (shown<10) {
                int tid=i/32, r=i%32;
                fprintf(stderr, "  [%s] mismatch tid=%d r=%d got=%.6f exp=%.6f abs=%.6f rel=%.6f\n",
                        label, tid, r, a, e, abserr, relerr);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [%s] maxabs=%.6g maxrel=%.6g mism=%d/%zu\n",
            label, maxabs, maxrel, mism, got.size());
    EXPECT_EQ(mism, 0) << label << ": " << mism << " values exceed tolerance";
}

} // namespace

TEST(QkGemmFullTile, MatchesCpuRef) {
    const int sq=64, sk=64, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto got=run_kernel(hQ,sq,hK,sk,D);
    std::vector<float> exp(kQKOutElems);
    ref_qk_gemm(hQ.data(),D,sq,hK.data(),D,sk,exp.data());
    compare(got, exp, "full/cpuref");
}

TEST(QkGemmFullTile, MatchesGolden) {
    if (g_golden_dir.empty()) GTEST_SKIP() << "no --golden dir";
    const int sq=64, sk=64, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto got=run_kernel(hQ,sq,hK,sk,D);
    std::vector<float> golden;
    ASSERT_TRUE(load_golden_sacc(g_golden_dir, golden));
    compare(got, golden, "full/golden");
}

TEST(QkGemmPartialTile, MatchesCpuRef) {
    const int sq=17, sk=33, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto got=run_kernel(hQ,sq,hK,sk,D);
    std::vector<float> exp(kQKOutElems);
    ref_qk_gemm(hQ.data(),D,sq,hK.data(),D,sk,exp.data());
    compare(got, exp, "partial/cpuref");
}

TEST(QkGemmPartialTile, MatchesGolden) {
    if (g_golden_dir.empty()) GTEST_SKIP() << "no --golden dir";
    const int sq=17, sk=33, D=64;
    auto hQ=make_Q(sq,D), hK=make_K(sk,D);
    auto got=run_kernel(hQ,sq,hK,sk,D);
    std::vector<float> golden;
    ASSERT_TRUE(load_golden_sacc(g_golden_dir, golden));
    compare(got, golden, "partial/golden");
}

int main(int argc, char** argv) {
    std::vector<char*> rest; rest.push_back(argv[0]);
    for (int i=1;i<argc;++i) {
        if (std::strncmp(argv[i], "--golden=", 9)==0) g_golden_dir = argv[i]+9;
        else rest.push_back(argv[i]);
    }
    int rc=static_cast<int>(rest.size());
    ::testing::InitGoogleTest(&rc, rest.data());
    return RUN_ALL_TESTS();
}
