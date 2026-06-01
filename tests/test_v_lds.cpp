// =============================================================================
// UNIT TEST: V-LDS component (src/components/v_lds.hpp), pipeline STAGE 5.
//
// What it validates: the load + v_perm transpose + ds_write staging of one
// 32-row V slice into LDS. HOW:
//   1. MatchesCpuRef — kernel's dumped LDS vs ref_v_lds() image, BYTE exact.
//   2. MatchesGolden — vs CK dump_lds.bin slot 1 (V_LDS, starts at float index
//      8192). Compares only valid data slots. SKIPS without a golden dir.
// V uses a distinct golden formula (v[i] = (i%256)/256 + 1) so V_LDS bugs can't
// hide behind K's values.
// =============================================================================
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "runner/bf16_utils.hpp"
#include "components_ref/ref_v_lds.hpp"
#include "components/v_lds.hpp"

static std::string g_golden_full;
static std::string g_golden_partial;

namespace {

// Golden input formula: v[i] = (i % 256) / 256 + 1
std::vector<uint16_t> make_golden_V(int seqlen_k, int D) {
    std::vector<uint16_t> V(static_cast<size_t>(seqlen_k) * D);
    for (int i = 0; i < seqlen_k * D; ++i) {
        float f = (i % 256) / 256.0f + 1.0f;
        V[i] = float_to_bf16(f);
    }
    return V;
}

void run_kernel(const std::vector<uint16_t>& h_V,
                int stride_v, int kv_offset, int seqlen_k,
                std::vector<uint16_t>& h_out) {
    h_out.resize(kVLdsRegionElems);
    const size_t nbytes_v = h_V.size() * sizeof(uint16_t);
    void *d_V = nullptr, *d_out = nullptr;
    ASSERT_EQ(hipMalloc(&d_V, nbytes_v), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_out, kVLdsRegionElems * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_V, h_V.data(), nbytes_v, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(d_out, 0, kVLdsRegionElems * sizeof(uint16_t)), hipSuccess);

    hipLaunchKernelGGL(test_v_lds_kernel, dim3(1), dim3(256), 0, nullptr,
                       reinterpret_cast<const uint16_t*>(d_V),
                       reinterpret_cast<uint16_t*>(d_out),
                       stride_v, kv_offset, seqlen_k);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(h_out.data(), d_out,
                        kVLdsRegionElems * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(d_V);
    hipFree(d_out);
}

// Load golden dump_lds slot 1 (V_LDS) as fp32 (bf16-widened). 8192 floats.
bool load_golden_vlds(const std::string& dir, std::vector<float>& out) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "dump_lds.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "  could not open golden %s\n", path.c_str()); return false; }
    // V_LDS is slot 1: starts at float index 8192
    fseek(f, 8192 * sizeof(float), SEEK_SET);
    out.resize(8192);
    size_t got = fread(out.data(), sizeof(float), 8192, f);
    fclose(f);
    if (got < 8192) { fprintf(stderr, "  golden too small: %zu floats\n", got); return false; }
    return true;
}

// Byte-exact compare: kernel LDS dump (uint16 bf16) vs CPU reference.
void check_vs_cpu_ref(const std::vector<uint16_t>& kern,
                      const std::vector<uint16_t>& h_V,
                      int stride_v, int kv_offset, int seqlen_k) {
    std::vector<uint16_t> expected(kVLdsRegionElems);
    ref_v_lds(h_V.data(), stride_v, kv_offset, seqlen_k, expected.data());

    int mism = 0, shown = 0;
    for (int e = 0; e < kVLdsRegionElems; ++e) {
        if (kern[e] != expected[e]) {
            ++mism;
            if (shown < 10) {
                fprintf(stderr, "  [cpuref] mismatch e=%d kern=0x%04x exp=0x%04x\n",
                        e, kern[e], expected[e]);
                ++shown;
            }
        }
    }
    fprintf(stderr, "  [cpuref] mism=%d/%d\n", mism, kVLdsRegionElems);
    EXPECT_EQ(mism, 0) << "cpuref: " << mism << " / " << kVLdsRegionElems
                       << " elements differ";
}

// Compare kernel dump vs golden over valid V data slots.
// Golden stores each bf16 as fp32 (zero-extended).
void check_vs_golden(const std::vector<uint16_t>& kern, int kv_offset, int seqlen_k,
                     const std::string& golden_dir) {
    if (golden_dir.empty()) {
        GTEST_SKIP() << "no golden dir provided";
    }
    std::vector<float> golden;
    ASSERT_TRUE(load_golden_vlds(golden_dir, golden));

    // Compare over the data region: V elements (n=0..31, d=0..63)
    int mism = 0, checked = 0, shown = 0;
    for (int n = 0; n < 32; ++n) {
        int row = kv_offset + n;
        if (row >= seqlen_k) continue;
        for (int d = 0; d < 64; ++d) {
            int off = kVLdsBufBase + v_lds_offset_elems(n, d);
            float kern_val = bf16_to_float(kern[off]);
            float gold_val = golden[off];
            ++checked;
            if (kern_val != gold_val) {
                ++mism;
                if (shown < 10) {
                    fprintf(stderr,
                            "  golden mismatch (n=%d,d=%d) off=%d kern=%.8g gold=%.8g\n",
                            n, d, off, kern_val, gold_val);
                    ++shown;
                }
            }
        }
    }
    fprintf(stderr, "  [golden] mism=%d/%d\n", mism, checked);
    EXPECT_EQ(mism, 0) << "golden: " << mism << " / " << checked << " elements differ";
}

} // namespace

TEST(VLdsFullTile, MatchesCpuRef) {
    const int seqlen_k = 64, D = 64, stride_v = 64, kv_offset = 0;
    auto h_V = make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_V, stride_v, kv_offset, seqlen_k);
}

TEST(VLdsFullTile, MatchesGolden) {
    const int seqlen_k = 64, D = 64, stride_v = 64, kv_offset = 0;
    auto h_V = make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_golden(kern, kv_offset, seqlen_k, g_golden_full);
}

TEST(VLdsPartialTile, MatchesCpuRef) {
    const int seqlen_k = 33, D = 64, stride_v = 64, kv_offset = 0;
    auto h_V = make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_V, stride_v, kv_offset, seqlen_k);
}

TEST(VLdsPartialTile, MatchesGolden) {
    const int seqlen_k = 33, D = 64, stride_v = 64, kv_offset = 0;
    auto h_V = make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_golden(kern, kv_offset, seqlen_k, g_golden_partial);
}

// ---- Edge case tests (CPU ref only) ----

TEST(VLdsEdge, MinTile) {
    const int seqlen_k=1, D=64, stride_v=64, kv_offset=0;
    auto h_V=make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_V, stride_v, kv_offset, seqlen_k);
}

TEST(VLdsEdge, FullM0) {
    const int seqlen_k=64, D=64, stride_v=64, kv_offset=0;
    auto h_V=make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_V, stride_v, kv_offset, seqlen_k);
}

TEST(VLdsEdge, SingleKCol) {
    const int seqlen_k=1, D=64, stride_v=64, kv_offset=0;
    auto h_V=make_golden_V(seqlen_k, D);
    std::vector<uint16_t> kern;
    run_kernel(h_V, stride_v, kv_offset, seqlen_k, kern);
    check_vs_cpu_ref(kern, h_V, stride_v, kv_offset, seqlen_k);
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
