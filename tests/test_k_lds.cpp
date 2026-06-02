// =============================================================================
// UNIT TEST: K-LDS component (src/components/k_lds.hpp), pipeline STAGE 1.
//
// What it validates: the GPU's swizzled DRAM->LDS K staging lands every element
// in the right LDS slot. HOW:
//   1. MatchesCpuRef — kernel's dumped LDS region vs cpu_ref_k_lds() image, BYTE
//      exact (uint16 bf16 compare). Always runs.
//   2. MatchesGolden — vs CK dump_lds.bin slot 0 (K_LDS), exact. Note the
//      golden stores each bf16 widened to fp32 at index (bufferBase + offset),
//      so the test compares only the real data slots, not the pad. SKIPS when
//      no golden dir is given.
// Edge cases (CPU-ref only) cover partial tiles where OOB rows must stay zero.
// =============================================================================
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "components_ref/cpu_ref_k_lds.hpp"
#include "components/k_lds.hpp"

// Optional golden directory (set via --golden=<dir>). Empty => skip golden check.
static std::string g_golden_full;
static std::string g_golden_partial;

namespace {

// Golden input formula (matches the CK dump): k[i] = ((i+64) % 256) / 256,
// where i is the linear index into the [seqlen_k, D] contiguous K matrix.
std::vector<uint16_t> make_golden_K(int seqlen_k, int D) {
    std::vector<uint16_t> K(static_cast<size_t>(seqlen_k) * D);
    for (int i = 0; i < seqlen_k * D; ++i) {
        float f = ((i + 64) % 256) / 256.0f;
        K[i] = float_to_bf16(f);
    }
    return K;
}

// Run the kernel, write result into `h_out`.
void run_kernel(const std::vector<uint16_t>& h_K,
                int stride_k, int kv_offset, int seqlen_k,
                std::vector<uint16_t>& h_out) {
    h_out.resize(kKLdsRegionElems);
    const size_t nbytes_k = h_K.size() * sizeof(uint16_t);
    void *d_K = nullptr, *d_out = nullptr;
    ASSERT_EQ(hipMalloc(&d_K, nbytes_k), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_out, kKLdsRegionElems * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_K, h_K.data(), nbytes_k, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(d_out, 0, kKLdsRegionElems * sizeof(uint16_t)), hipSuccess);

    dim3 grid(1), block(256);
    hipLaunchKernelGGL(test_k_lds_kernel, grid, block, 0, nullptr,
                       reinterpret_cast<const uint16_t*>(d_K),
                       reinterpret_cast<uint16_t*>(d_out),
                       stride_k, kv_offset, seqlen_k);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(h_out.data(), d_out,
                        kKLdsRegionElems * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(d_K);
    hipFree(d_out);
}

// Load golden dump_lds.bin slot 0 (K_LDS) as fp32 floats (>= 8192).
bool load_golden_klds(const std::string& dir, std::vector<float>& out) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_lds.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  could not open golden %s\n", path.c_str()); return false; }
    out.resize(8192);
    size_t got = fread(out.data(), sizeof(float), 8192, f);
    fclose(f);
    if (got < 8192) { fprintf(stderr, "  golden too small: %zu floats\n", got); return false; }
    return true;
}

constexpr int kGoldenBufferBase = 2304; // CK ring buffer-0 origin in the dump

// Byte-exact: kernel dump vs CPU reference over the full region.
void check_vs_cpu_ref(const std::vector<uint16_t>& kern,
                      const std::vector<uint16_t>& h_K,
                      int stride_k, int kv_offset, int seqlen_k) {
    std::vector<uint16_t> expected(kKLdsRegionElems);
    cpu_ref_k_lds(h_K.data(), stride_k, kv_offset, seqlen_k, expected.data());

    int mism = 0, shown = 0;
    for (int e = 0; e < kKLdsRegionElems; ++e) {
        if (kern[e] != expected[e]) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [cpuref] mismatch e=%d kern=0x%04x exp=0x%04x\n",
                        e, kern[e], expected[e]);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [cpuref] mism=%d/%d\n", mism, kKLdsRegionElems);
    EXPECT_EQ(mism, 0) << "cpuref: " << mism << " / " << kKLdsRegionElems
                       << " elements differ";
}

// Byte-exact: kernel dump vs golden, over the K (j,d) data slots only.
// Golden stores each bf16 widened to fp32 at index (kGoldenBufferBase + off).
void check_vs_golden(const std::vector<uint16_t>& kern, int kv_offset, int seqlen_k,
                     const std::string& golden_dir) {
    if (golden_dir.empty()) {
        GTEST_SKIP() << "no golden dir provided";
    }
    std::vector<float> golden;
    ASSERT_TRUE(load_golden_klds(golden_dir, golden));

    int mism = 0, shown = 0, checked = 0;
    for (int j = 0; j < 64; ++j) {
        int row = kv_offset + j;
        if (row >= seqlen_k) continue; // OOB rows: golden zero, not a data slot
        for (int d = 0; d < 64; ++d) {
            int e = k_lds_offset_elems(j, d);
            float kern_val = bf16_to_float(kern[e]);
            float gold_val = golden[kGoldenBufferBase + e];
            ++checked;
            if (kern_val != gold_val) {
                ++mism;
                if (shown < 10) {
                    fprintf(stderr,
                            "  golden mismatch (j=%d,d=%d) e=%d kern=%.8g gold=%.8g\n",
                            j, d, e, kern_val, gold_val);
                    ++shown;
                }
            }
        }
    }
    fprintf(stderr, "  [golden] mism=%d/%d\n", mism, checked);
    EXPECT_EQ(mism, 0) << "golden: " << mism << " / " << checked << " elements differ";
}

} // namespace

TEST(KLdsFullTile, MatchesCpuRef) {
    const int seqlen_k = 64, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_K, stride_k, kv_offset, seqlen_k);
}

TEST(KLdsFullTile, MatchesGolden) {
    const int seqlen_k = 64, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_golden(kern, kv_offset, seqlen_k, g_golden_full);
}

TEST(KLdsPartialTile, MatchesCpuRef) {
    const int seqlen_k = 33, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_K, stride_k, kv_offset, seqlen_k);
}

TEST(KLdsPartialTile, MatchesGolden) {
    const int seqlen_k = 33, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_golden(kern, kv_offset, seqlen_k, g_golden_partial);
}

// ---- Edge case tests (CPU ref only) ----

TEST(KLdsEdge, MinTile) {
    const int seqlen_k = 1, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_K, stride_k, kv_offset, seqlen_k);
}

TEST(KLdsEdge, FullM0) {
    const int seqlen_k = 64, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_K, stride_k, kv_offset, seqlen_k);
}

TEST(KLdsEdge, SingleCol) {
    const int seqlen_k = 1, D = 64, stride_k = 64, kv_offset = 0;
    auto h_K = make_golden_K(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_K, stride_k, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_K, stride_k, kv_offset, seqlen_k);
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
