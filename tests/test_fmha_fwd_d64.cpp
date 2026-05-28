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

// Kernel declarations (defined in fmha_fwd_d64_kernel.cpp)
__global__ void fmha_fwd_d64_bf16_msk0(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk1(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk0_varlen(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk1_varlen(FmhaFwdParams params);

// ---------- Softmax smoke test (disabled) ----------
// This test verified the intermediate softmax P output that was stored
// to O in an earlier kernel version. The kernel now computes full FMHA
// (O = softmax(Q*K^T) * V). Use the parameterized test below instead.

TEST(FmhaFwdD64Smoke, DISABLED_Softmax) {
    constexpr int B = 1, H = 1, S_q = 128, S_k = 64, D = 64;
    constexpr float kLog2e = 1.4426950408889634f;

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
    // scale includes log2(e) for exp2-based softmax
    params.scale = kLog2e / sqrtf(static_cast<float>(D));
    // Q strides: [B, H, S_q, D]
    params.stride_q       = D;
    params.nhead_stride_q = S_q * D;
    params.batch_stride_q = H * S_q * D;
    // K strides: [B, H, S_k, D]
    params.stride_k       = D;
    params.nhead_stride_k = S_k * D;
    params.batch_stride_k = H * S_k * D;
    // O strides: [B, H, S_q, S_k] (storing P matrix for verification)
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

    // Read back O (which contains softmax P as bf16)
    std::vector<uint16_t> h_O(nelems_o);
    ASSERT_EQ(hipMemcpy(h_O.data(), d_O, nbytes_o, hipMemcpyDeviceToHost), hipSuccess);

    // CPU reference: P = softmax(Q * K^T / sqrt(d))
    // 1. Compute S = Q * K^T / sqrt(d) using bf16 inputs, fp32 accumulation
    const float nat_scale = 1.0f / sqrtf(static_cast<float>(D));
    std::vector<float> ref_S(static_cast<size_t>(S_q) * S_k, 0.0f);
    for (int i = 0; i < S_q; i++) {
        for (int j = 0; j < S_k; j++) {
            float dot = 0;
            for (int d = 0; d < D; d++) {
                float q_val = bf16_to_float(h_Q[i * D + d]);
                float k_val = bf16_to_float(h_K[j * D + d]);
                dot += q_val * k_val;
            }
            ref_S[i * S_k + j] = dot * nat_scale;
        }
    }

    // 2. Softmax per row
    std::vector<float> ref_P(static_cast<size_t>(S_q) * S_k);
    for (int i = 0; i < S_q; i++) {
        float row_max = -INFINITY;
        for (int j = 0; j < S_k; j++)
            row_max = fmaxf(row_max, ref_S[i * S_k + j]);

        float row_sum = 0;
        for (int j = 0; j < S_k; j++) {
            ref_P[i * S_k + j] = expf(ref_S[i * S_k + j] - row_max);
            row_sum += ref_P[i * S_k + j];
        }

        float inv_sum = 1.0f / row_sum;
        for (int j = 0; j < S_k; j++)
            ref_P[i * S_k + j] *= inv_sum;
    }

    // Compare kernel output against CPU reference (both as bf16)
    int mismatches = 0;
    float max_abs_err = 0;
    float max_rel_err = 0;
    for (int i = 0; i < S_q; i++) {
        for (int j = 0; j < S_k; j++) {
            float ref_val = ref_P[i * S_k + j];
            float kern_val = bf16_to_float(h_O[i * S_k + j]);
            float ref_bf16_val = bf16_to_float(float_to_bf16(ref_val));
            float abs_err = fabsf(kern_val - ref_bf16_val);
            float rel_err = (ref_bf16_val != 0) ? abs_err / fabsf(ref_bf16_val) : abs_err;
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            // Softmax values are in [0,1]; allow larger tolerance for exp/sum
            if (abs_err > 0.02f) {
                if (mismatches < 10) {
                    fprintf(stderr,
                        "  mismatch [%d,%d]: kernel=%+.6f ref=%+.6f ref_bf16=%+.6f err=%.6f\n",
                        i, j, kern_val, ref_val, ref_bf16_val, abs_err);
                }
                mismatches++;
            }
        }
    }

    fprintf(stderr, "Softmax max abs err: %.6f, max rel err: %.6f, mismatches (>0.02): %d / %d\n",
            max_abs_err, max_rel_err, mismatches, S_q * S_k);

    // Verify row sums are close to 1.0
    float max_rowsum_err = 0;
    for (int i = 0; i < S_q; i++) {
        float rowsum = 0;
        for (int j = 0; j < S_k; j++)
            rowsum += bf16_to_float(h_O[i * S_k + j]);
        float err = fabsf(rowsum - 1.0f);
        if (err > max_rowsum_err) max_rowsum_err = err;
    }
    fprintf(stderr, "Softmax max row-sum deviation from 1.0: %.6f\n", max_rowsum_err);

    EXPECT_EQ(mismatches, 0) << mismatches << " / " << (S_q * S_k) << " elements exceed tolerance";
    EXPECT_LT(max_rowsum_err, 0.05f) << "Row sums deviate too much from 1.0";

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_O);
}

// ---------- Parameterized full test ----------

class FmhaFwdD64Test : public ::testing::TestWithParam<TestCase> {};

TEST_P(FmhaFwdD64Test, MatchesGpuRef) {
    const auto& tc = GetParam();

    FmhaParams p = make_params(tc);
    const bool varlen = !tc.varlen_seqs.empty();
    FmhaBuffers bufs(p);
    bufs.fill_random(42);
    bufs.copy_to_device();

    constexpr float kLog2e = 1.4426950408889634f;

    // Allocate separate LSE buffer for kernel when LSE is enabled
    void* d_LSE_kern = nullptr;
    if (tc.lse) {
        ASSERT_EQ(hipMalloc(&d_LSE_kern, bufs.sz_LSE * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMemset(d_LSE_kern, 0, bufs.sz_LSE * sizeof(float)), hipSuccess);
    }

    // Build FmhaFwdParams for our kernel
    FmhaFwdParams kparams{};
    kparams.q = reinterpret_cast<const __hip_bfloat16*>(bufs.d_Q);
    kparams.k = reinterpret_cast<const __hip_bfloat16*>(bufs.d_K);
    kparams.v = reinterpret_cast<const __hip_bfloat16*>(bufs.d_V);
    kparams.o = reinterpret_cast<__hip_bfloat16*>(bufs.d_O);
    kparams.lse = tc.lse ? reinterpret_cast<float*>(d_LSE_kern) : nullptr;
    kparams.seqlen_q = p.seq_len;
    kparams.seqlen_k = p.kv_seq_len;
    kparams.nhead_q = p.q_heads;
    kparams.nhead_k = p.kv_heads;
    kparams.scale = kLog2e / sqrtf(static_cast<float>(p.head_dim));

    // Strides in elements: [B, H, S, D]
    const int D = p.hdim_dispatch();
    kparams.stride_q       = D;
    kparams.stride_k       = D;
    kparams.stride_v       = D;
    kparams.stride_o       = D;

    if (varlen) {
        kparams.nhead_stride_q = bufs.total_seqlen * D;
        kparams.nhead_stride_k = bufs.total_seqlen * D;
        kparams.nhead_stride_v = bufs.total_seqlen * D;
        kparams.nhead_stride_o = bufs.total_seqlen * D;
        kparams.batch_stride_q = 0;
        kparams.batch_stride_k = 0;
        kparams.batch_stride_v = 0;
        kparams.batch_stride_o = 0;
        kparams.seqstart_q = reinterpret_cast<const int32_t*>(bufs.d_qseq);
        kparams.seqstart_k = reinterpret_cast<const int32_t*>(bufs.d_kseq);
    } else {
        kparams.nhead_stride_q = p.seq_len * D;
        kparams.nhead_stride_k = p.kv_seq_len * D;
        kparams.nhead_stride_v = p.kv_seq_len * D;
        kparams.nhead_stride_o = p.seq_len * D;
        kparams.batch_stride_q = p.q_heads * p.seq_len * D;
        kparams.batch_stride_k = p.kv_heads * p.kv_seq_len * D;
        kparams.batch_stride_v = p.kv_heads * p.kv_seq_len * D;
        kparams.batch_stride_o = p.q_heads * p.seq_len * D;
        kparams.seqstart_q = nullptr;
        kparams.seqstart_k = nullptr;
    }

    // Launch kernel
    const int m_tiles = (p.seq_len + kM0 - 1) / kM0;
    dim3 grid(p.q_heads, m_tiles, p.batch);
    dim3 block(kBlockSize);
    if (varlen) {
        if (tc.mask) {
            hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk1_varlen, grid, block, 0, nullptr, kparams);
        } else {
            hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0_varlen, grid, block, 0, nullptr, kparams);
        }
    } else {
        if (tc.mask) {
            hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk1, grid, block, 0, nullptr, kparams);
        } else {
            hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0, grid, block, 0, nullptr, kparams);
        }
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Copy output back
    bufs.copy_from_device();

    // Verify against CPU reference
    CpuRefResult result = cpu_ref_verify(p, bufs);
    EXPECT_TRUE(result.pass)
        << "max_abs=" << result.max_abs
        << " mismatches=" << result.mismatch << "/" << result.total_elems
        << " nonfinite=" << result.nonfinite
        << " cos_fail=" << result.cos_fail << "/" << result.cos_rows;

    // Verify LSE against GPU reference
    if (tc.lse) {
        // Run GPU ref with LSE enabled (writes to bufs.d_LSE)
        GpuRefParams gp{};
        gp.d_Q = reinterpret_cast<const uint16_t*>(bufs.d_Q);
        gp.d_K = reinterpret_cast<const uint16_t*>(bufs.d_K);
        gp.d_V = reinterpret_cast<const uint16_t*>(bufs.d_V);
        gp.d_O = reinterpret_cast<uint16_t*>(bufs.d_O);
        gp.d_LSE = reinterpret_cast<float*>(bufs.d_LSE);
        gp.batch = p.batch; gp.q_heads = p.q_heads; gp.kv_heads = p.kv_heads; gp.gqa = p.gqa;
        gp.seq_len = p.seq_len; gp.kv_seq_len = p.kv_seq_len; gp.head_dim = p.head_dim;
        gp.stride_q_seq = D; gp.stride_k_seq = D;
        gp.stride_v_seq = D; gp.stride_o_seq = D;
        gp.stride_q_head = bufs.stride_q_head / 2; gp.stride_q_batch = bufs.stride_q_batch / 2;
        gp.stride_k_head = bufs.stride_k_head / 2; gp.stride_k_batch = bufs.stride_k_batch / 2;
        gp.stride_v_head = bufs.stride_k_head / 2; gp.stride_v_batch = bufs.stride_k_batch / 2;
        gp.stride_o_head = bufs.stride_q_head / 2; gp.stride_o_batch = bufs.stride_q_batch / 2;
        gp.mask = p.mask; gp.scalar = p.scalar();
        gp.d_seqstart_q = varlen ? reinterpret_cast<const uint32_t*>(bufs.d_qseq) : nullptr;
        gp.d_seqstart_k = varlen ? reinterpret_cast<const uint32_t*>(bufs.d_kseq) : nullptr;

        gpu_ref_fmha_fwd(gp);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        // Copy both LSE buffers to host
        std::vector<float> h_lse_kern(bufs.sz_LSE);
        std::vector<float> h_lse_ref(bufs.sz_LSE);
        ASSERT_EQ(hipMemcpy(h_lse_kern.data(), d_LSE_kern,
                            bufs.sz_LSE * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        ASSERT_EQ(hipMemcpy(h_lse_ref.data(), bufs.d_LSE,
                            bufs.sz_LSE * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

        int lse_mismatches = 0;
        float lse_max_abs = 0;
        for (size_t idx = 0; idx < bufs.sz_LSE; idx++) {
            float kern_val = h_lse_kern[idx];
            float ref_val  = h_lse_ref[idx];
            if (std::isinf(kern_val) && std::isinf(ref_val) &&
                kern_val < 0 && ref_val < 0)
                continue;
            float abs_err = fabsf(kern_val - ref_val);
            if (abs_err > lse_max_abs) lse_max_abs = abs_err;
            if (abs_err > 0.001f) {
                if (lse_mismatches < 5) {
                    fprintf(stderr, "  LSE mismatch [%zu]: kernel=%.6f ref=%.6f err=%.6f\n",
                            idx, kern_val, ref_val, abs_err);
                }
                lse_mismatches++;
            }
        }
        fprintf(stderr, "LSE max_abs=%.6f mismatches=%d/%zu\n",
                lse_max_abs, lse_mismatches, bufs.sz_LSE);
        EXPECT_EQ(lse_mismatches, 0)
            << "LSE mismatches: " << lse_mismatches << "/" << bufs.sz_LSE;

        hipFree(d_LSE_kern);
    }
}

INSTANTIATE_TEST_SUITE_P(Full, FmhaFwdD64Test,
    ::testing::ValuesIn(kAllFull),
    [](const auto& info) { return info.param.name; });
