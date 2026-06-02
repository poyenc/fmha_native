// =============================================================================
// UNIT TEST: split-K PRODUCER kernels (fmha_fwd_d64_bf16_msk{0,1}_split) — TDD
// RED step.
//
// THIS TEST IS INTENTIONALLY UN-LINKABLE RIGHT NOW.  The two split-forward
// __global__ entries do NOT exist yet (they arrive in Task 7).  This file
// references them, so the link step fails with "undefined reference to
// fmha_fwd_d64_bf16_msk0_split" (and/or _msk1_split).  That undefined-symbol
// error IS the RED state we want: the test must COMPILE cleanly (the param
// struct FmhaFwdSplitParams already lives in runner/params.hpp since Task 5),
// and only fail to LINK.  When Task 7 lands the kernels, this same file turns
// GREEN unchanged.
//
// WHAT IT VERIFIES.  The split PRODUCER runs the ordinary forward pass but each
// block walks ONLY its split's disjoint KV sub-range and writes a NORMALIZED
// fp32 partial (O_g, natural-log LSE_g) into the split-major "scratch" staging
// buffer (the same layout the combine pass reads — see FmhaFwdSplitParams in
// runner/params.hpp).  This test is the PRODUCER half of the G-INVARIANCE
// guarantee: it pins each per-split partial against an independent oracle over
// the EXACT same KV sub-range the kernel walks.  (The COMBINE half — folding G
// partials back into the full-range output — is the sibling test_combine.cpp.)
//
// THE ORACLES.
//   * cpu_ref_split(p, bufs, b, hq, i, kv_start, kv_end, o_row, lse_out)
//       host, exact, slow.  Computes one output row's partial attention over the
//       KV range [kv_start, kv_end), normalized by THAT range's local sum, with
//       the natural-log per-range LSE.  Used for the S=2048 correctness sweeps.
//   * gpu_ref_split(gp, kv_start, kv_end, d_o_g, d_lse_g)
//       device, exact, fast.  Same math, one fp32 thread per row; output is fp32
//       [total_rows*64] with row == flat [b,q_heads,seq] index.  Used for the
//       large-S (S~40000) sweeps where the host oracle would take minutes.
//   Both oracles already match each other (the Stage-1 trust test in
//   test_fmha_gpu_ref.cpp), so either is ground truth here.
//
// ---------------------------------------------------------------------------
// SPLIT LAUNCH CONTRACT (defined HERE; Task 7 must implement to match)
// ---------------------------------------------------------------------------
//   * Grid : dim3(nhead_q, m_tiles, batch*num_splits).  The z-axis carries BOTH
//            batch and split: the global decodes b = blockIdx.z / num_splits and
//            split_idx = blockIdx.z % num_splits.  (B is recovered device-side as
//            gridDim.z / num_splits — see pipeline.hpp's scratch-base math.)
//   * Block: kBlockSize (=256) threads, same as the forward/combine path.
//   * KV narrowing: from pipeline.hpp, the split path first does the ordinary
//            (causal-clamped, for msk1) per-M-tile tile count, then keeps only
//            tiles [split_idx*T, min((split_idx+1)*T, full)) where
//            T = ceil(full / num_splits) and a tile is kN0(=64) keys.  So for
//            mask0 every M-tile has the SAME full = ceil(Skv/64), and split g
//            owns absolute keys [g*T*64, min((g+1)*T*64, Skv)).  For mask1 the
//            full tile count is the per-M-tile CAUSAL-CLAMPED count, so the
//            split's key end is clamped to that M-tile's causal end (replicated
//            by causal_kv_end() below).
//   * Writes: epilog_store_split writes the NORMALIZED fp32 partial O_g + the
//            natural-log LSE_g to scratch at the split-major index
//              scratch_o  (g,b,h,row,d) = (((g*B+b)*Hq+h)*Sq+row)*64 + d
//              scratch_lse(g,b,h,row)   =  ((g*B+b)*Hq+h)*Sq+row
//            An EMPTY split (its narrowed range is empty — e.g. a trailing split
//            past the tile count, or a mask1 split entirely in the masked-future
//            region) writes the fp32 SENTINEL plane: LSE_g = -inf, O_g = 0 (via
//            the degenerate path in pipeline.hpp, which for IsSplit routes to
//            epilog_store_split, NOT the bf16 epilog_store).
//
// The scratch row index is the ABSOLUTE query row the block computed.  The msk1
// entry applies the causal M-tile reversal (m_tile = gridDim.y-1-blockIdx.y),
// but because the partial is keyed by the absolute row (not the launch order),
// comparing per-(b,h,row) is reversal-agnostic — we never need to know which
// physical block produced a row.
// =============================================================================
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "runner/bf16_utils.hpp"
#include "runner/params.hpp"        // FmhaFwdSplitParams (owned in params.hpp, T5)
#include "runner/fmha_params.hpp"   // FmhaParams (workload config)
#include "runner/buffers.hpp"       // FmhaBuffers + fill_random
#include "runner/cpu_ref.hpp"       // cpu_ref_split (host oracle, small S)
#include "runner/gpu_ref.hpp"       // gpu_ref_split (device oracle, large S)

// ---------------------------------------------------------------------------
// THE undefined symbols that make this the RED test.  They take the split
// kernarg (FmhaFwdSplitParams) by value, exactly as Task 7 will define them.
// ---------------------------------------------------------------------------
extern __global__ void fmha_fwd_d64_bf16_msk0_split(FmhaFwdSplitParams);
extern __global__ void fmha_fwd_d64_bf16_msk1_split(FmhaFwdSplitParams);

namespace {

constexpr int kD = 64;            // head_dim this kernel is specialized for
constexpr float kLog2e = 1.4426950408889634f;  // matches bench's scale fold

// ---------------------------------------------------------------------------
// Split-major scratch index helpers (identical to test_combine's, and to the
// layout FmhaFwdSplitParams documents).
// ---------------------------------------------------------------------------
inline size_t scratch_o_idx(int g, int b, int h, int row, int d,
                            int B, int Hq, int Sq) {
    return ((((size_t)g * B + b) * Hq + h) * Sq + row) * kD + d;
}
inline size_t scratch_lse_idx(int g, int b, int h, int row,
                              int B, int Hq, int Sq) {
    return (((size_t)g * B + b) * Hq + h) * Sq + row;
}

// A poison sentinel for unwritten scratch.  We pick a value that is NEITHER a
// valid partial output (those are small ~O(0.01..1) fp32) NOR the -inf/0 empty
// sentinel the kernel writes, so an UNWRITTEN plane is unambiguously detectable.
// We use a large finite magic number for O and a large finite magic for LSE.
constexpr float kPoisonO   = 1.0e30f;
constexpr float kPoisonLse = 7.0e30f;

// ---------------------------------------------------------------------------
// Build the split kernel's FmhaFwdSplitParams.base (a FmhaFwdParams) from a
// FmhaBuffers, MIRRORING bench_fmha_fwd.cpp's kparams fill exactly (dense /
// fixed-length only — split-K is a dense-mode feature here).  Device Q/K/V/O
// pointers + contiguous BHSD strides + the log2e-folded scale.
// ---------------------------------------------------------------------------
FmhaFwdParams make_base_params(const FmhaParams& p, const FmhaBuffers& bufs) {
    FmhaFwdParams kp{};
    kp.q = reinterpret_cast<const __hip_bfloat16*>(bufs.d_Q);
    kp.k = reinterpret_cast<const __hip_bfloat16*>(bufs.d_K);
    kp.v = reinterpret_cast<const __hip_bfloat16*>(bufs.d_V);
    kp.o = reinterpret_cast<__hip_bfloat16*>(bufs.d_O);  // unused by split path
    kp.lse = nullptr;                                    // split writes scratch_lse
    kp.seqlen_q = p.seq_len;
    kp.seqlen_k = p.kv_seq_len;
    kp.nhead_q = p.q_heads;
    kp.nhead_k = p.kv_heads;
    kp.scale = kLog2e / sqrtf(static_cast<float>(p.head_dim));

    const int D = p.hdim_dispatch();
    kp.stride_q = D; kp.stride_k = D; kp.stride_v = D; kp.stride_o = D;
    kp.nhead_stride_q = p.seq_len    * D;
    kp.nhead_stride_k = p.kv_seq_len * D;
    kp.nhead_stride_v = p.kv_seq_len * D;
    kp.nhead_stride_o = p.seq_len    * D;
    kp.batch_stride_q = p.q_heads  * p.seq_len    * D;
    kp.batch_stride_k = p.kv_heads * p.kv_seq_len * D;
    kp.batch_stride_v = p.kv_heads * p.kv_seq_len * D;
    kp.batch_stride_o = p.q_heads  * p.seq_len    * D;
    kp.seqstart_q = nullptr;
    kp.seqstart_k = nullptr;
    return kp;
}

// Build a GpuRefParams from FmhaBuffers/FmhaParams — same derivation as
// test_fmha_gpu_ref.cpp's make_gpu_ref_params (V reuses K strides; O reuses Q
// strides).  Used for the large-S oracle path.
GpuRefParams make_gpu_ref_params(const FmhaParams& p, const FmhaBuffers& bufs) {
    const int Dpad = p.hdim_dispatch();
    GpuRefParams gp{};
    gp.d_Q = reinterpret_cast<const uint16_t*>(bufs.d_Q);
    gp.d_K = reinterpret_cast<const uint16_t*>(bufs.d_K);
    gp.d_V = reinterpret_cast<const uint16_t*>(bufs.d_V);
    gp.d_O = reinterpret_cast<uint16_t*>(bufs.d_O);   // unused by split oracle
    gp.d_LSE = nullptr;
    gp.batch = p.batch; gp.q_heads = p.q_heads; gp.kv_heads = p.kv_heads; gp.gqa = p.gqa;
    gp.seq_len = p.seq_len; gp.kv_seq_len = p.kv_seq_len; gp.head_dim = p.head_dim;
    gp.stride_q_seq = Dpad; gp.stride_k_seq = Dpad;
    gp.stride_v_seq = Dpad; gp.stride_o_seq = Dpad;
    gp.stride_q_head = bufs.stride_q_head / 2; gp.stride_q_batch = bufs.stride_q_batch / 2;
    gp.stride_k_head = bufs.stride_k_head / 2; gp.stride_k_batch = bufs.stride_k_batch / 2;
    gp.stride_v_head = bufs.stride_k_head / 2; gp.stride_v_batch = bufs.stride_k_batch / 2;
    gp.stride_o_head = bufs.stride_q_head / 2; gp.stride_o_batch = bufs.stride_q_batch / 2;
    gp.mask = p.mask; gp.scalar = p.scalar();
    gp.d_seqstart_q = nullptr;
    gp.d_seqstart_k = nullptr;
    return gp;
}

// ---------------------------------------------------------------------------
// Replicate the kernel's per-M-tile CAUSAL KV end (pipeline.hpp HasMask block).
// For mask0 the end is just Skv (no clamp).  For mask1 the end depends on the
// M-tile: keys past the M-tile's diagonal are all masked, so the kernel rounds
// the diagonal end up to a whole kN0 tile and clamps to Skv.  The split tiling
// is taken over THIS end, so the test's KV ranges must use the same value.
//   m_tile : the ABSOLUTE M-tile index (the row's tile), NOT the launch order.
//            The msk1 reversal only changes WHICH block runs WHEN; the per-tile
//            causal end is a function of the absolute tile, so this is correct
//            regardless of reversal.
// ---------------------------------------------------------------------------
int causal_kv_end(int m_tile, int Sq, int Skv, bool mask) {
    if (!mask) return Skv;
    int mask_shift = Skv - Sq;
    int last_q_row = m_tile * kM0 + kM0 - 1;
    if (last_q_row >= Sq) last_q_row = Sq - 1;
    int raw_end = last_q_row + mask_shift + 1;
    if (raw_end > Skv) raw_end = Skv;
    int end = ((raw_end + kN0 - 1) / kN0) * kN0;   // round up to a whole kN0 tile
    if (end > Skv) end = Skv;
    return end;
}

// The KV sub-range [kv0, kv1) that split g owns for a row in absolute M-tile
// m_tile.  Mirrors pipeline.hpp's narrowing EXACTLY:
//   end       = causal_kv_end(m_tile)                     (Skv for mask0)
//   full      = ceil(end / kN0)                           tiles this M-tile walks
//   T         = ceil(full / G)                            tiles per split (ceil)
//   tile_lo   = g*T ; tile_hi = min((g+1)*T, full)
//   kv0       = tile_lo*kN0 ; kv1 = min(tile_hi*kN0, end)
//   (empty split => kv0 == kv1 — cpu_ref_split then returns o=0, lse=-inf,
//    matching the kernel's sentinel plane.)
void split_kv_range(int g, int G, int m_tile, int Sq, int Skv, bool mask,
                    int* kv0, int* kv1) {
    int end  = causal_kv_end(m_tile, Sq, Skv, mask);
    int full = (end + kN0 - 1) / kN0;
    int T    = (full + G - 1) / G;
    int tile_lo = g * T;
    int tile_hi = tile_lo + T;
    if (tile_hi > full) tile_hi = full;
    if (tile_lo > full) tile_lo = full;     // wholly-empty trailing split
    int a = tile_lo * kN0;
    int b = tile_hi * kN0;
    if (a > end) a = end;
    if (b > end) b = end;
    *kv0 = a;
    *kv1 = b;
}

// ---------------------------------------------------------------------------
// GPU split driver.  Uploads nothing extra (Q/K/V already on device via bufs),
// allocates split-major scratch sized [G*B*Hq*Sq*64] (O) + [G*B*Hq*Sq] (LSE),
// poisons it, launches the split kernel over grid z = B*G, and copies scratch
// back to host as fp32.  Returns the two host scratch vectors by out-param.
// ---------------------------------------------------------------------------
struct SplitDims { int G, B, Hq, Sq; };

void run_split(bool mask, const FmhaFwdParams& base, const SplitDims& dim,
               std::vector<float>* h_scratch_o,
               std::vector<float>* h_scratch_lse) {
    const int G = dim.G, B = dim.B, Hq = dim.Hq, Sq = dim.Sq;
    const size_t n_o   = (size_t)G * B * Hq * Sq * kD;
    const size_t n_lse = (size_t)G * B * Hq * Sq;

    // Poison host buffers first, then upload — so a kernel that fails to write a
    // plane leaves the poison value (detectable) rather than uninitialized junk.
    h_scratch_o->assign(n_o, kPoisonO);
    h_scratch_lse->assign(n_lse, kPoisonLse);

    float *d_sco = nullptr, *d_sclse = nullptr;
    ASSERT_EQ(hipMalloc(&d_sco,   n_o   * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_sclse, n_lse * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_sco,   h_scratch_o->data(),   n_o   * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_sclse, h_scratch_lse->data(), n_lse * sizeof(float),
                        hipMemcpyHostToDevice), hipSuccess);

    FmhaFwdSplitParams sp{};
    sp.base        = base;
    sp.scratch_o   = d_sco;
    sp.scratch_lse = d_sclse;
    sp.num_splits  = G;
    sp.split_idx   = 0;   // decoded device-side from blockIdx.z % G; field is informational

    // Grid: (Hq, m_tiles, B*G).  z-axis carries batch*split (see contract above).
    const int m_tiles = (Sq + kM0 - 1) / kM0;
    dim3 grid(Hq, m_tiles, B * G);
    dim3 block(kBlockSize);
    if (mask)
        hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk1_split, grid, block, 0, nullptr, sp);
    else
        hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0_split, grid, block, 0, nullptr, sp);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(h_scratch_o->data(),   d_sco,   n_o   * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(h_scratch_lse->data(), d_sclse, n_lse * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(d_sco); hipFree(d_sclse);
}

// The ragged G set: 2,8 divide a 32-tile count cleanly; 3,5 do not (ragged last
// split).  Used for the S=2048 correctness sweep.
const std::vector<int> kGSetRagged = {2, 3, 5, 8};

} // namespace

// ===========================================================================
// CASE 1 — One-range correctness, ragged-G sweep (mask0).  S=2048.  For each
// G in {2,3,5,8} and each split g, compare scratch_o[g]/scratch_lse[g] to
// cpu_ref_split over THE EXACT KV sub-range the kernel walks.  This is the
// producer half of G-INVARIANCE: each partial must be correct on its own range.
// ===========================================================================
TEST(SplitProducer, OneRangeRaggedG_Mask0) {
    FmhaParams p{};
    p.batch = 1; p.q_heads = 2; p.kv_heads = 2; p.gqa = 1;
    p.seq_len = 2048; p.kv_seq_len = 2048; p.head_dim = kD; p.mask = 0;

    const int B = p.batch, Hq = p.q_heads, Sq = p.seq_len, Skv = p.kv_seq_len;

    FmhaBuffers bufs(p);
    bufs.fill_random(42);
    bufs.copy_to_device();   // cpu_ref_split reads host Q/K/V; kernel reads device
    FmhaFwdParams base = make_base_params(p, bufs);

    for (int G : kGSetRagged) {
        std::vector<float> sco, sclse;
        run_split(/*mask=*/false, base, {G, B, Hq, Sq}, &sco, &sclse);

        float max_abs_o = 0.0f, max_abs_lse = 0.0f;
        int mism = 0, shown = 0;
        std::vector<float> ref_o(kD);
        for (int g = 0; g < G; ++g)
        for (int h = 0; h < Hq; ++h)
        for (int row = 0; row < Sq; ++row) {
            int kv0, kv1;
            int m_tile = row / kM0;                       // absolute M-tile of this row
            split_kv_range(g, G, m_tile, Sq, Skv, /*mask=*/false, &kv0, &kv1);

            float ref_lse = -INFINITY;
            cpu_ref_split(p, bufs, 0, h, row, kv0, kv1, ref_o.data(), &ref_lse);

            float got_lse = sclse[scratch_lse_idx(g, 0, h, row, B, Hq, Sq)];

            // Empty range: both sides must be the -inf sentinel; O all-zero.
            bool ref_inf = std::isinf(ref_lse) && ref_lse < 0;
            bool got_inf = std::isinf(got_lse) && got_lse < 0;
            EXPECT_EQ(ref_inf, got_inf)
                << "G=" << G << " g=" << g << " h=" << h << " row=" << row
                << " ref_lse=" << ref_lse << " got_lse=" << got_lse;
            if (ref_inf) {
                for (int d = 0; d < kD; ++d) {
                    float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                    max_abs_o = std::max(max_abs_o, std::fabs(v));   // must be 0
                }
                continue;
            }
            max_abs_lse = std::max(max_abs_lse, std::fabs(got_lse - ref_lse));
            for (int d = 0; d < kD; ++d) {
                float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                float ae = std::fabs(v - ref_o[d]);
                max_abs_o = std::max(max_abs_o, ae);
                if (ae > 2e-3f && shown < 10) {
                    fprintf(stderr, "  [G=%d g=%d h=%d row=%d d=%d] got=%.6f ref=%.6f\n",
                            G, g, h, row, d, v, ref_o[d]);
                    ++shown; ++mism;
                }
            }
        }
        fprintf(stderr, "[OneRangeRaggedG_Mask0 G=%d] max_abs_o=%.6g max_abs_lse=%.6g\n",
                G, max_abs_o, max_abs_lse);
        EXPECT_LT(max_abs_o,   2e-3f) << "O_g vs cpu_ref_split, G=" << G;
        EXPECT_LT(max_abs_lse, 2e-3f) << "LSE_g vs cpu_ref_split, G=" << G;
    }
}

// ===========================================================================
// CASE 2 — G=1 fp32 identity (mask0).  S=2048, G=1: the single split walks the
// WHOLE KV range, so scratch_o[0] must equal the full-range fp32 partial.  Both
// the kernel and the oracle are fp32 with identical bf16-truncation, so the only
// slack is GPU-vs-host fp32 accumulation ORDER.  We try a tight 1e-5 first; if
// the producer's tiled accumulation order differs enough from the host's serial
// order, fall back to gpu_ref_split (same device accumulation flavor) at 1e-5
// and keep a looser cpu_ref_split bound (2e-4) — documented inline.
// ===========================================================================
TEST(SplitProducer, G1Fp32Identity_Mask0) {
    FmhaParams p{};
    p.batch = 1; p.q_heads = 2; p.kv_heads = 2; p.gqa = 1;
    p.seq_len = 2048; p.kv_seq_len = 2048; p.head_dim = kD; p.mask = 0;

    const int B = p.batch, Hq = p.q_heads, Sq = p.seq_len, Skv = p.kv_seq_len;

    FmhaBuffers bufs(p);
    bufs.fill_random(7);
    bufs.copy_to_device();
    FmhaFwdParams base = make_base_params(p, bufs);

    std::vector<float> sco, sclse;
    run_split(/*mask=*/false, base, {1, B, Hq, Sq}, &sco, &sclse);

    // Oracle path: gpu_ref_split over [0,Skv) — fp32, same device accumulation
    // flavor as the producer, so the tightest comparison.
    const int total_rows = B * Hq * Sq;
    GpuRefParams gp = make_gpu_ref_params(p, bufs);
    float *d_o_g = nullptr, *d_lse_g = nullptr;
    ASSERT_EQ(hipMalloc(&d_o_g,   (size_t)total_rows * kD * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_lse_g, (size_t)total_rows * sizeof(float)), hipSuccess);
    gpu_ref_split(gp, 0, Skv, d_o_g, d_lse_g);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<float> g_o((size_t)total_rows * kD), g_lse(total_rows);
    ASSERT_EQ(hipMemcpy(g_o.data(),   d_o_g,   (size_t)total_rows * kD * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(g_lse.data(), d_lse_g, (size_t)total_rows * sizeof(float),
                        hipMemcpyDeviceToHost), hipSuccess);
    hipFree(d_o_g); hipFree(d_lse_g);

    float max_abs_o_gpu = 0.0f, max_abs_lse_gpu = 0.0f;
    float max_abs_o_cpu = 0.0f, max_abs_lse_cpu = 0.0f;
    std::vector<float> cpu_o(kD);
    for (int h = 0; h < Hq; ++h)
    for (int row = 0; row < Sq; ++row) {
        // gpu_ref_split row order is flat [b,q_heads,seq]; b=0 here.
        const int grow = (0 * Hq + h) * Sq + row;
        float got_lse = sclse[scratch_lse_idx(0, 0, h, row, B, Hq, Sq)];
        max_abs_lse_gpu = std::max(max_abs_lse_gpu, std::fabs(got_lse - g_lse[grow]));
        for (int d = 0; d < kD; ++d) {
            float v = sco[scratch_o_idx(0, 0, h, row, d, B, Hq, Sq)];
            max_abs_o_gpu = std::max(max_abs_o_gpu, std::fabs(v - g_o[(size_t)grow * kD + d]));
        }
        // Looser cross-check vs the host oracle (different accumulation order).
        float clse = -INFINITY;
        cpu_ref_split(p, bufs, 0, h, row, 0, Skv, cpu_o.data(), &clse);
        max_abs_lse_cpu = std::max(max_abs_lse_cpu, std::fabs(got_lse - clse));
        for (int d = 0; d < kD; ++d) {
            float v = sco[scratch_o_idx(0, 0, h, row, d, B, Hq, Sq)];
            max_abs_o_cpu = std::max(max_abs_o_cpu, std::fabs(v - cpu_o[d]));
        }
    }
    fprintf(stderr, "[G1Fp32Identity_Mask0] vs gpu: O=%.6g LSE=%.6g | vs cpu: O=%.6g LSE=%.6g\n",
            max_abs_o_gpu, max_abs_lse_gpu, max_abs_o_cpu, max_abs_lse_cpu);
    // Primary (tight) bound against the same-flavor device oracle.
    EXPECT_LT(max_abs_o_gpu,   1e-5f) << "G=1 fp32 identity O vs gpu_ref_split";
    EXPECT_LT(max_abs_lse_gpu, 1e-5f) << "G=1 fp32 identity LSE vs gpu_ref_split";
    // Secondary (looser) bound against the host oracle: fp32 accumulation order
    // differs (tiled-online on device vs serial on host), so 2e-4 not 1e-5.
    EXPECT_LT(max_abs_o_cpu,   2e-4f) << "G=1 fp32 identity O vs cpu_ref_split";
    EXPECT_LT(max_abs_lse_cpu, 2e-4f) << "G=1 fp32 identity LSE vs cpu_ref_split";
}

// ===========================================================================
// CASE 3 — Degenerate/empty trailing split (mask0).  Choose G so at least one
// trailing split has an EMPTY KV range.  S=2048 => num_tiles = 32.  With G=33,
// T = ceil(32/33) = 1, so split g owns tile g; splits 0..31 are non-empty and
// split 32 owns tile 32 which is past the 32-tile count => EMPTY.  Assert the
// empty split's plane is the fp32 sentinel (LSE=-inf, O=0) for every row, AND
// that the poison was overwritten (proving the degenerate path actually ran).
// ===========================================================================
TEST(SplitProducer, DegenerateEmptyTrailingSplit_Mask0) {
    FmhaParams p{};
    p.batch = 1; p.q_heads = 2; p.kv_heads = 2; p.gqa = 1;
    p.seq_len = 2048; p.kv_seq_len = 2048; p.head_dim = kD; p.mask = 0;

    const int B = p.batch, Hq = p.q_heads, Sq = p.seq_len, Skv = p.kv_seq_len;
    const int num_tiles = (Skv + kN0 - 1) / kN0;   // 32
    const int G = num_tiles + 1;                    // 33 => last split empty (T=1)

    FmhaBuffers bufs(p);
    bufs.fill_random(99);
    bufs.copy_to_device();
    FmhaFwdParams base = make_base_params(p, bufs);

    std::vector<float> sco, sclse;
    run_split(/*mask=*/false, base, {G, B, Hq, Sq}, &sco, &sclse);

    // For mask0 the per-M-tile count is uniform (= num_tiles), so the empty
    // splits are exactly those whose tile_lo >= num_tiles.
    const int T = (num_tiles + G - 1) / G;          // 1
    int empty_count = 0;
    for (int g = 0; g < G; ++g) {
        if (g * T < num_tiles) continue;            // non-empty split
        ++empty_count;
        for (int h = 0; h < Hq; ++h)
        for (int row = 0; row < Sq; ++row) {
            float got_lse = sclse[scratch_lse_idx(g, 0, h, row, B, Hq, Sq)];
            // Sentinel: LSE = -inf (NOT the poison), O = 0 (NOT the poison).
            EXPECT_TRUE(std::isinf(got_lse) && got_lse < 0)
                << "empty split g=" << g << " row=" << row << " lse=" << got_lse;
            EXPECT_NE(got_lse, kPoisonLse) << "poison not overwritten (path skipped)";
            for (int d = 0; d < kD; ++d) {
                float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                EXPECT_EQ(v, 0.0f) << "empty split O not zero g=" << g
                                   << " row=" << row << " d=" << d;
            }
        }
    }
    EXPECT_GT(empty_count, 0) << "expected at least one empty trailing split";
}

// ===========================================================================
// CASE 4 — All-but-few empty splits, G > tile count (mask0).  S=512 => 8 tiles,
// G=16 => T=1 so splits 0..7 are non-empty and splits 8..15 are ALL empty.
// Every empty plane must be the fp32 sentinel (-inf, 0) written via
// epilog_store_split (the fp32 path, NOT bf16), and the poison must be
// overwritten — proving the degenerate fp32-plane write actually executes for a
// fully-empty split (not just left as input poison).
// ===========================================================================
TEST(SplitProducer, AllEmptySplits_Mask0) {
    FmhaParams p{};
    p.batch = 1; p.q_heads = 2; p.kv_heads = 2; p.gqa = 1;
    p.seq_len = 512; p.kv_seq_len = 512; p.head_dim = kD; p.mask = 0;

    const int B = p.batch, Hq = p.q_heads, Sq = p.seq_len, Skv = p.kv_seq_len;
    const int num_tiles = (Skv + kN0 - 1) / kN0;   // 8
    const int G = 16;                               // 8 empty trailing splits

    FmhaBuffers bufs(p);
    bufs.fill_random(11);
    bufs.copy_to_device();
    FmhaFwdParams base = make_base_params(p, bufs);

    std::vector<float> sco, sclse;
    run_split(/*mask=*/false, base, {G, B, Hq, Sq}, &sco, &sclse);

    const int T = (num_tiles + G - 1) / G;          // 1
    int empty_count = 0;
    for (int g = 0; g < G; ++g) {
        if (g * T < num_tiles) continue;            // non-empty
        ++empty_count;
        for (int h = 0; h < Hq; ++h)
        for (int row = 0; row < Sq; ++row) {
            float got_lse = sclse[scratch_lse_idx(g, 0, h, row, B, Hq, Sq)];
            EXPECT_TRUE(std::isinf(got_lse) && got_lse < 0)
                << "all-empty g=" << g << " row=" << row << " lse=" << got_lse;
            EXPECT_NE(got_lse, kPoisonLse)
                << "poison LSE not overwritten — fp32 sentinel path did not run";
            for (int d = 0; d < kD; ++d) {
                float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                EXPECT_EQ(v, 0.0f) << "all-empty O not zero g=" << g
                                   << " row=" << row << " d=" << d;
                EXPECT_NE(v, kPoisonO)
                    << "poison O not overwritten — fp32 plane not written";
            }
        }
    }
    EXPECT_EQ(empty_count, G - num_tiles) << "expected G-num_tiles empty splits";
}

// ===========================================================================
// CASE 5 — mask1 (causal), ragged + masked-future splits.  S=2048, G in {3,5},
// _msk1_split entry.  For low M-tiles the causal end is small (few tiles), so a
// high-index split's whole sub-range is in the masked-future region => its plane
// is the -inf/0 sentinel.  Surviving splits compare to cpu_ref_split WITH the
// causal mask (p.mask=1) over the EXACT causal-clamped KV range the kernel
// walks (causal_kv_end + split narrowing).  Producer-side proof of the causal
// (A4) invariant: the split tiling never reaches a masked-future key.
//
// Reversal note: the msk1 entry launches m_tiles in reverse, but partials are
// keyed by ABSOLUTE row, so comparing per-(h,row) is reversal-agnostic.
// ===========================================================================
TEST(SplitProducer, CausalRaggedAndMaskedFuture_Mask1) {
    FmhaParams p{};
    p.batch = 1; p.q_heads = 2; p.kv_heads = 2; p.gqa = 1;
    p.seq_len = 2048; p.kv_seq_len = 2048; p.head_dim = kD; p.mask = 1;

    const int B = p.batch, Hq = p.q_heads, Sq = p.seq_len, Skv = p.kv_seq_len;

    FmhaBuffers bufs(p);
    bufs.fill_random(2024);
    bufs.copy_to_device();
    FmhaFwdParams base = make_base_params(p, bufs);

    for (int G : {3, 5}) {
        std::vector<float> sco, sclse;
        run_split(/*mask=*/true, base, {G, B, Hq, Sq}, &sco, &sclse);

        float max_abs_o = 0.0f, max_abs_lse = 0.0f;
        int empty_seen = 0;
        std::vector<float> ref_o(kD);
        for (int g = 0; g < G; ++g)
        for (int h = 0; h < Hq; ++h)
        for (int row = 0; row < Sq; ++row) {
            int kv0, kv1;
            int m_tile = row / kM0;
            split_kv_range(g, G, m_tile, Sq, Skv, /*mask=*/true, &kv0, &kv1);

            float ref_lse = -INFINITY;
            cpu_ref_split(p, bufs, 0, h, row, kv0, kv1, ref_o.data(), &ref_lse);

            float got_lse = sclse[scratch_lse_idx(g, 0, h, row, B, Hq, Sq)];
            bool ref_inf = std::isinf(ref_lse) && ref_lse < 0;
            bool got_inf = std::isinf(got_lse) && got_lse < 0;
            EXPECT_EQ(ref_inf, got_inf)
                << "mask1 G=" << G << " g=" << g << " h=" << h << " row=" << row
                << " ref_lse=" << ref_lse << " got_lse=" << got_lse;
            if (ref_inf) {
                ++empty_seen;
                for (int d = 0; d < kD; ++d) {
                    float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                    max_abs_o = std::max(max_abs_o, std::fabs(v));   // must be 0
                }
                continue;
            }
            max_abs_lse = std::max(max_abs_lse, std::fabs(got_lse - ref_lse));
            for (int d = 0; d < kD; ++d) {
                float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                max_abs_o = std::max(max_abs_o, std::fabs(v - ref_o[d]));
            }
        }
        fprintf(stderr, "[CausalRaggedAndMaskedFuture_Mask1 G=%d] max_abs_o=%.6g "
                "max_abs_lse=%.6g empty_seen=%d\n", G, max_abs_o, max_abs_lse, empty_seen);
        EXPECT_LT(max_abs_o,   2e-3f) << "mask1 O_g vs cpu_ref_split, G=" << G;
        EXPECT_LT(max_abs_lse, 2e-3f) << "mask1 LSE_g vs cpu_ref_split, G=" << G;
        // The low M-tiles + high-index splits MUST produce masked-future empties,
        // otherwise the test isn't exercising the A4 invariant it claims to.
        EXPECT_GT(empty_seen, 0) << "expected masked-future empty splits, G=" << G;
    }
}

// ===========================================================================
// CASE 6 — Large-S via the trusted fast GPU oracle (mask0).  S in {40000,
// 39936, 40064}, G in {8,32}.  Compare scratch_o[g]/scratch_lse[g] to
// gpu_ref_split over the EXACT split KV range (host oracle would take minutes).
// Catches S-dependent scratch-offset / overflow bugs: at S=40000, the split
// offset for g is g*B*Hq*Sq*64 ~ 41M elems per split.  G=32 at S=40000 forces
// empty trailing splits (ceil(40000/64)=625 tiles, T=ceil(625/32)=20, splits
// 0..31 -> 32*20=640>=625, so the LAST split(s) trail past 625 => empty).  The
// 39936 (=624*64, exact tile multiple) / 40064 (=626*64) pair brackets the
// Q-tail edge.
//
// Memory: each case allocates one G*B*Hq*Sq*64 fp32 scratch (G=32,S40000,B1,H2
// ~= 655 MB) PLUS the oracle's per-call total_rows*64 buffer.  We free both
// between cases (run_split frees its device scratch; we free the oracle's), so
// only one big allocation is live at a time — fine on MI300X 192 GB.
// ===========================================================================
TEST(SplitProducer, LargeS_vs_GpuRefSplit_Mask0) {
    struct LCase { int S, G; };
    const std::vector<LCase> cases = {
        {39936, 8}, {40000, 8}, {40064, 8},
        {39936, 32}, {40000, 32}, {40064, 32},
    };

    for (const LCase& c : cases) {
        FmhaParams p{};
        p.batch = 1; p.q_heads = 2; p.kv_heads = 2; p.gqa = 1;
        p.seq_len = c.S; p.kv_seq_len = c.S; p.head_dim = kD; p.mask = 0;

        const int B = p.batch, Hq = p.q_heads, Sq = p.seq_len, Skv = p.kv_seq_len;
        const int G = c.G;
        const int total_rows = B * Hq * Sq;

        FmhaBuffers bufs(p);
        bufs.fill_random(555);
        bufs.copy_to_device();
        FmhaFwdParams base = make_base_params(p, bufs);

        // Producer scratch (freed inside run_split's device side; host kept here).
        std::vector<float> sco, sclse;
        run_split(/*mask=*/false, base, {G, B, Hq, Sq}, &sco, &sclse);

        // One reusable oracle buffer (total_rows*64 fp32), re-run per split range.
        GpuRefParams gp = make_gpu_ref_params(p, bufs);
        float *d_o_g = nullptr, *d_lse_g = nullptr;
        ASSERT_EQ(hipMalloc(&d_o_g,   (size_t)total_rows * kD * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_lse_g, (size_t)total_rows * sizeof(float)), hipSuccess);
        std::vector<float> g_o((size_t)total_rows * kD), g_lse(total_rows);

        float max_abs_o = 0.0f, max_abs_lse = 0.0f;
        int empty_seen = 0;
        for (int g = 0; g < G; ++g) {
            // mask0 => uniform per-M-tile range; use m_tile=0 (any tile gives the
            // same range for mask0) to derive the split's absolute KV bounds.
            int kv0, kv1;
            split_kv_range(g, G, /*m_tile=*/0, Sq, Skv, /*mask=*/false, &kv0, &kv1);

            gpu_ref_split(gp, kv0, kv1, d_o_g, d_lse_g);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            ASSERT_EQ(hipMemcpy(g_o.data(),   d_o_g,   (size_t)total_rows * kD * sizeof(float),
                                hipMemcpyDeviceToHost), hipSuccess);
            ASSERT_EQ(hipMemcpy(g_lse.data(), d_lse_g, (size_t)total_rows * sizeof(float),
                                hipMemcpyDeviceToHost), hipSuccess);

            for (int h = 0; h < Hq; ++h)
            for (int row = 0; row < Sq; ++row) {
                const int grow = (0 * Hq + h) * Sq + row;
                float got_lse = sclse[scratch_lse_idx(g, 0, h, row, B, Hq, Sq)];
                float ref_lse = g_lse[grow];
                bool ref_inf = std::isinf(ref_lse) && ref_lse < 0;
                bool got_inf = std::isinf(got_lse) && got_lse < 0;
                ASSERT_EQ(ref_inf, got_inf)
                    << "S=" << c.S << " G=" << G << " g=" << g
                    << " h=" << h << " row=" << row
                    << " ref_lse=" << ref_lse << " got_lse=" << got_lse;
                if (ref_inf) {
                    if (h == 0 && row == 0) ++empty_seen;
                    for (int d = 0; d < kD; ++d) {
                        float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                        max_abs_o = std::max(max_abs_o, std::fabs(v));   // must be 0
                    }
                    continue;
                }
                max_abs_lse = std::max(max_abs_lse, std::fabs(got_lse - ref_lse));
                for (int d = 0; d < kD; ++d) {
                    float v = sco[scratch_o_idx(g, 0, h, row, d, B, Hq, Sq)];
                    max_abs_o = std::max(max_abs_o, std::fabs(v - g_o[(size_t)grow * kD + d]));
                }
            }
        }
        hipFree(d_o_g); hipFree(d_lse_g);

        fprintf(stderr, "[LargeS S=%d G=%d] max_abs_o=%.6g max_abs_lse=%.6g empty_splits=%d\n",
                c.S, G, max_abs_o, max_abs_lse, empty_seen);
        // Both kernel and gpu_ref_split run the same bf16 P-truncation path, so
        // 2e-3 is a safe upper bound on the O / LSE drift.
        EXPECT_LT(max_abs_o,   2e-3f) << "S=" << c.S << " G=" << G << " O vs gpu_ref_split";
        EXPECT_LT(max_abs_lse, 2e-3f) << "S=" << c.S << " G=" << G << " LSE vs gpu_ref_split";
    }
}

int main(int argc, char** argv) {
    // Mirror the component tests' arg handling: strip any --golden* flags so the
    // gate's shared invocation (which passes them to all standalone bins) does
    // not confuse GoogleTest.  This suite has no golden dependency.
    std::vector<char*> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--golden", 8) == 0) continue;
        rest.push_back(argv[i]);
    }
    int rc = static_cast<int>(rest.size());
    ::testing::InitGoogleTest(&rc, rest.data());
    return RUN_ALL_TESTS();
}
