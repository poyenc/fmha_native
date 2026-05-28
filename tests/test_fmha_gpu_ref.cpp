// GPU reference kernel correctness test.
//
// Verifies that the naive FP32 GPU reference kernel (gpu_ref_fmha_fwd)
// matches the CPU reference (cpu_ref_verify) across all D64 test configs.
// Both implementations follow the same BF16 truncation pipeline, so
// results should match within tight FP32 accumulation-order tolerance.

#include "test_configs.hpp"
#include "runner/buffers.hpp"
#include "runner/gpu_ref.hpp"
#include "runner/cpu_ref.hpp"
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

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
