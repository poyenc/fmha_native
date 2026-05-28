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
#include <cmath>
#include <vector>

// Kernel declaration (defined in fmha_fwd_d64_kernel.cpp)
__global__ void fmha_fwd_d64_bf16_msk0(FmhaFwdParams params);

// ---------- GEMM0 smoke test ----------
// Verifies that S_acc = Q * K^T * scale is computed correctly.
// The kernel currently stores S_acc (as bf16) to O for verification.

TEST(FmhaFwdD64Smoke, Gemm0) {
    constexpr int B = 1, H = 1, S_q = 128, S_k = 64, D = 64;

    // Allocate device buffers
    const size_t nelems_q = static_cast<size_t>(B) * H * S_q * D;
    const size_t nelems_k = static_cast<size_t>(B) * H * S_k * D;
    const size_t nelems_o = static_cast<size_t>(B) * H * S_q * S_k;
    const size_t nbytes_q = nelems_q * sizeof(__hip_bfloat16);
    const size_t nbytes_k = nelems_k * sizeof(__hip_bfloat16);
    const size_t nbytes_o = nelems_o * sizeof(__hip_bfloat16);

    void *d_Q = nullptr, *d_K = nullptr, *d_O = nullptr;
    ASSERT_EQ(hipMalloc(&d_Q, nbytes_q), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_K, nbytes_k), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_O, nbytes_o), hipSuccess);

    // Fill Q and K with known data on host
    std::vector<uint16_t> h_Q(nelems_q);
    std::vector<uint16_t> h_K(nelems_k);
    srand(42);
    for (size_t i = 0; i < nelems_q; i++)
        h_Q[i] = float_to_bf16(((int)(i % 1000) - 500) / 5000.0f);
    for (size_t i = 0; i < nelems_k; i++)
        h_K[i] = float_to_bf16(((int)(i % 997) - 498) / 5000.0f);

    ASSERT_EQ(hipMemcpy(d_Q, h_Q.data(), nbytes_q, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_K, h_K.data(), nbytes_k, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(d_O, 0, nbytes_o), hipSuccess);

    // Build FmhaFwdParams
    FmhaFwdParams params{};
    params.q = reinterpret_cast<const __hip_bfloat16*>(d_Q);
    params.k = reinterpret_cast<const __hip_bfloat16*>(d_K);
    params.v = nullptr;
    params.o = reinterpret_cast<__hip_bfloat16*>(d_O);
    params.lse = nullptr;
    params.seqlen_q = S_q;
    params.seqlen_k = S_k;
    params.nhead_q = H;
    params.nhead_k = H;
    params.scale = 1.0f / sqrtf(static_cast<float>(D));
    // Q strides: [B, H, S_q, D]
    params.stride_q       = D;
    params.nhead_stride_q = S_q * D;
    params.batch_stride_q = H * S_q * D;
    // K strides: [B, H, S_k, D]
    params.stride_k       = D;
    params.nhead_stride_k = S_k * D;
    params.batch_stride_k = H * S_k * D;
    // O strides: [B, H, S_q, S_k] (storing S matrix, not attention output)
    params.stride_o       = S_k;
    params.nhead_stride_o = S_q * S_k;
    params.batch_stride_o = H * S_q * S_k;
    params.seqstart_q = nullptr;
    params.seqstart_k = nullptr;

    // Launch kernel
    const int m_tiles = (S_q + kM0 - 1) / kM0;
    dim3 grid(H, m_tiles, B);
    dim3 block(kBlockSize);
    hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0, grid, block, 0, nullptr, params);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Read back O (which contains S_acc as bf16)
    std::vector<uint16_t> h_O(nelems_o);
    ASSERT_EQ(hipMemcpy(h_O.data(), d_O, nbytes_o, hipMemcpyDeviceToHost), hipSuccess);

    // CPU reference: S = Q * K^T * scale
    // Q: [S_q, D], K: [S_k, D], S: [S_q, S_k]
    std::vector<float> ref_S(static_cast<size_t>(S_q) * S_k, 0.0f);
    for (int i = 0; i < S_q; i++) {
        for (int j = 0; j < S_k; j++) {
            float dot = 0;
            for (int d = 0; d < D; d++) {
                float q_val = bf16_to_float(h_Q[i * D + d]);
                float k_val = bf16_to_float(h_K[j * D + d]);
                dot += q_val * k_val;
            }
            ref_S[i * S_k + j] = dot * params.scale;
        }
    }

    // Compare: convert ref to bf16 for comparison
    int mismatches = 0;
    float max_abs_err = 0;
    for (int i = 0; i < S_q; i++) {
        for (int j = 0; j < S_k; j++) {
            float ref_val = ref_S[i * S_k + j];
            float kern_val = bf16_to_float(h_O[i * S_k + j]);
            float ref_bf16_val = bf16_to_float(float_to_bf16(ref_val));
            float abs_err = fabsf(kern_val - ref_bf16_val);
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            // Allow tolerance for accumulated bf16 rounding
            if (abs_err > 0.01f) {
                if (mismatches < 10) {
                    fprintf(stderr,
                        "  mismatch [%d,%d]: kernel=%+.5f ref=%+.5f ref_bf16=%+.5f err=%.5f\n",
                        i, j, kern_val, ref_val, ref_bf16_val, abs_err);
                }
                mismatches++;
            }
        }
    }

    fprintf(stderr, "GEMM0 max abs err: %.6f, mismatches (>0.01): %d / %d\n",
            max_abs_err, mismatches, S_q * S_k);

    EXPECT_EQ(mismatches, 0) << mismatches << " / " << (S_q * S_k) << " elements exceed tolerance";

    hipFree(d_Q);
    hipFree(d_K);
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
