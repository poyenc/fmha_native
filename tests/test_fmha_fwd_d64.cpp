#include <gtest/gtest.h>
#include "runner/gpu_ref.hpp"
#include "runner/cpu_ref.hpp"
#include "runner/buffers.hpp"
#include "runner/params.hpp"
#include "runner/bf16_utils.hpp"
#include "test_configs.hpp"
#include "test_params.hpp"
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdio>

// Kernel declaration (defined in fmha_fwd_d64_kernel.cpp)
__global__ void fmha_fwd_d64_bf16_msk0(FmhaFwdParams params);

// ---------- Q passthrough smoke test ----------

TEST(FmhaFwdD64Smoke, QPassthrough) {
    constexpr int B = 1, H = 1, S = 128, D = 64;

    // Allocate device buffers
    const size_t nelems = static_cast<size_t>(B) * H * S * D;
    const size_t nbytes = nelems * sizeof(__hip_bfloat16);
    void *d_Q = nullptr, *d_O = nullptr;
    ASSERT_EQ(hipMalloc(&d_Q, nbytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_O, nbytes), hipSuccess);

    // Fill Q with known pattern on host, upload
    std::vector<uint16_t> h_Q(nelems);
    srand(123);
    for (size_t i = 0; i < nelems; i++)
        h_Q[i] = float_to_bf16(((int)(i % 1000) - 500) / 5000.0f);

    ASSERT_EQ(hipMemcpy(d_Q, h_Q.data(), nbytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(d_O, 0, nbytes), hipSuccess);

    // Build FmhaFwdParams
    FmhaFwdParams params{};
    params.q = reinterpret_cast<const __hip_bfloat16*>(d_Q);
    params.o = reinterpret_cast<__hip_bfloat16*>(d_O);
    params.k = nullptr;
    params.v = nullptr;
    params.lse = nullptr;
    params.seqlen_q = S;
    params.seqlen_k = S;
    params.nhead_q = H;
    params.nhead_k = H;
    params.scale = 1.0f;
    // Strides in bf16 elements (row-major: [B, H, S, D])
    params.stride_q       = D;
    params.nhead_stride_q = S * D;
    params.batch_stride_q = H * S * D;
    params.stride_o       = D;
    params.nhead_stride_o = S * D;
    params.batch_stride_o = H * S * D;
    params.seqstart_q = nullptr;
    params.seqstart_k = nullptr;

    // Launch: grid = (nhead, m_tiles, batch), block = (256)
    const int m_tiles = (S + kM0 - 1) / kM0;
    dim3 grid(H, m_tiles, B);
    dim3 block(kBlockSize);
    hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0, grid, block, 0, nullptr, params);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Read back O
    std::vector<uint16_t> h_O(nelems);
    ASSERT_EQ(hipMemcpy(h_O.data(), d_O, nbytes, hipMemcpyDeviceToHost), hipSuccess);

    // Verify O == Q element-wise
    int mismatches = 0;
    for (size_t i = 0; i < nelems; i++) {
        if (h_O[i] != h_Q[i]) {
            if (mismatches < 10) {
                int row = static_cast<int>(i / D);
                int col = static_cast<int>(i % D);
                fprintf(stderr, "  mismatch [%d,%d]: Q=0x%04x O=0x%04x\n",
                        row, col, h_Q[i], h_O[i]);
            }
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << mismatches << " / " << nelems << " elements differ";

    hipFree(d_Q);
    hipFree(d_O);
}

// ---------- Parameterized full test (stub) ----------

class FmhaFwdD64Test : public ::testing::TestWithParam<TestCase> {};

TEST_P(FmhaFwdD64Test, MatchesGpuRef) {
    GTEST_SKIP() << "Kernel not implemented yet";
}

INSTANTIATE_TEST_SUITE_P(Full, FmhaFwdD64Test,
    ::testing::ValuesIn(kAllFull),
    [](const auto& info) { return info.param.name; });
