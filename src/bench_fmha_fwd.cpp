// FMHA forward benchmark — timing-only CLI for native HIP kernel performance.
//
// Dispatches the compiled-in D64 bf16 kernel with configurable problem
// dimensions, runs warmup + timed iterations, and reports per-iteration
// TFLOPS.  Optionally runs the two-pass split-K dispatch (--splitk G) and/or
// an on-device correctness check (--verify) against the trusted GPU reference.
//
// Usage:
//   bench_fmha_fwd -b 2 -h 16 -s 4096 -d 64
//   bench_fmha_fwd -b 1 -h 2 -s 40000 -d 64 --splitk 8        # two-pass split-K
//   bench_fmha_fwd -b 1 -h 2 -s 40000 -d 64 --splitk 8 --verify  # + correctness

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <getopt.h>
#include "runner/params.hpp"
#include "runner/fmha_params.hpp"
#include "runner/buffers.hpp"
#include "runner/gpu_ref.hpp"      // gpu_ref_fmha_fwd — trusted oracle for --verify
#include "runner/bf16_utils.hpp"   // bf16_to_float — host-side O comparison

// Kernel declarations (defined in src/fused/kernel.cpp)
__global__ void fmha_fwd_d64_bf16_msk0(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk1(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk0_varlen(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk1_varlen(FmhaFwdParams params);

// Split-K globals (defined in src/fused/kernel.cpp).  The two producers write
// G normalized fp32 partials into the split-major scratch; the combine pass
// folds those partials back into the single global-softmax bf16 output.
__global__ void fmha_fwd_d64_bf16_msk0_split(FmhaFwdSplitParams sp);
__global__ void fmha_fwd_d64_bf16_msk1_split(FmhaFwdSplitParams sp);
__global__ void fmha_fwd_d64_bf16_combine(FmhaFwdCombineParams cp);

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t e = (expr);                                                 \
        if (e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", e,                \
                    hipGetErrorString(e), __FILE__, __LINE__);                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct BenchConfig {
    FmhaParams params;
    int warmup  = 5;
    int iters   = 20;
    int splitk  = 0;       // G: 0 = single-pass (default), >0 = two-pass split-K
    bool verify = false;   // compare kernel O against gpu_ref oracle, fail nonzero
};

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  -b  --batch N    Batch size [2]\n"
           "  -h  --heads N    Q head count [16]\n"
           "  --kv-heads N     KV head count [0 = q_heads]\n"
           "  -s  --seq N      Sequence length [4096]\n"
           "  -d  --hdim N     Head dimension [64]\n"
           "  --mask N         Causal mask 0|1 [0]\n"
           "  --lse N          LSE output enable [0]\n"
           "  --varlen-seqs L  Per-batch seq lengths (comma-separated)\n"
           "  --warmup N       Warmup iterations [5]\n"
           "  --iters N        Benchmark iterations [20]\n"
           "  --splitk G       Split-K split count (0=single-pass) [0]\n"
           "  --verify         Verify O against GPU reference (nonzero exit on fail)\n"
           , prog);
}

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    cfg.params.head_dim = 64;
    FmhaParams& p = cfg.params;
    static struct option long_opts[] = {
        {"varlen-seqs", required_argument, 0, 6},
        {"batch",    required_argument, 0, 'b'},
        {"heads",    required_argument, 0, 'h'},
        {"kv-heads", required_argument, 0,  1 },
        {"seq",      required_argument, 0, 's'},
        {"hdim",     required_argument, 0, 'd'},
        {"mask",     required_argument, 0, 'm'},
        {"lse",      required_argument, 0, 'l'},
        {"warmup",   required_argument, 0, 'w'},
        {"iters",    required_argument, 0, 'i'},
        {"splitk",   required_argument, 0,  7 },
        {"verify",   no_argument,       0,  8 },
        {"help",     no_argument,       0,  0 },
        {0, 0, 0, 0}
    };
    int c, idx;
    while ((c = getopt_long(argc, argv, "b:h:s:d:m:l:w:i:", long_opts, &idx)) != -1) {
        switch (c) {
        case  6 : {
            const char* s = optarg;
            while (*s) {
                p.varlen_seqs.push_back(atoi(s));
                while (*s && *s != ',') s++;
                if (*s) s++;
            }
            break;
        }
        case 'b': p.batch           = atoi(optarg); break;
        case 'h': p.q_heads         = atoi(optarg); break;
        case  1 : p.kv_heads        = atoi(optarg); break;
        case 's': p.seq_len         = atoi(optarg); break;
        case 'd': p.head_dim        = atoi(optarg); break;
        case 'm': p.mask            = atoi(optarg); break;
        case 'l': p.lse             = atoi(optarg); break;
        case 'w': cfg.warmup        = atoi(optarg); break;
        case 'i': cfg.iters         = atoi(optarg); break;
        case  7 : cfg.splitk        = atoi(optarg); break;
        case  8 : cfg.verify        = true;         break;
        case  0 : print_usage(argv[0]); exit(0);
        default:  print_usage(argv[0]); exit(1);
        }
    }
    if (p.kv_heads == 0) p.kv_heads = p.q_heads;
    if (p.gqa == 1 && p.q_heads != p.kv_heads)
        p.gqa = p.q_heads / p.kv_heads;
    if (p.q_heads % p.kv_heads != 0) {
        fprintf(stderr, "Error: q_heads (%d) must be divisible by kv_heads (%d)\n",
                p.q_heads, p.kv_heads);
        exit(1);
    }
    if (!p.varlen_seqs.empty()) {
        p.batch = (int)p.varlen_seqs.size();
        int max_s = 0;
        for (int s : p.varlen_seqs) if (s > max_s) max_s = s;
        p.seq_len = max_s;
    }
    if (p.kv_seq_len == 0) p.kv_seq_len = p.seq_len;
    return cfg;
}

int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);
    const FmhaParams& p = cfg.params;

    const bool varlen_mode = !p.varlen_seqs.empty();
    constexpr float kLog2e = 1.4426950408889634f;

    FmhaBuffers bufs(p);
    bufs.fill_random(42);
    bufs.copy_to_device();

    // Build FmhaFwdParams
    FmhaFwdParams kparams{};
    kparams.q = reinterpret_cast<const __hip_bfloat16*>(bufs.d_Q);
    kparams.k = reinterpret_cast<const __hip_bfloat16*>(bufs.d_K);
    kparams.v = reinterpret_cast<const __hip_bfloat16*>(bufs.d_V);
    kparams.o = reinterpret_cast<__hip_bfloat16*>(bufs.d_O);
    kparams.lse = p.lse ? reinterpret_cast<float*>(bufs.d_LSE) : nullptr;
    kparams.seqlen_q = p.seq_len;
    kparams.seqlen_k = p.kv_seq_len;
    kparams.nhead_q = p.q_heads;
    kparams.nhead_k = p.kv_heads;
    kparams.scale = kLog2e / sqrtf(static_cast<float>(p.head_dim));

    const int D = p.hdim_dispatch();
    kparams.stride_q = D;
    kparams.stride_k = D;
    kparams.stride_v = D;
    kparams.stride_o = D;

    if (varlen_mode) {
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

    // Grid / block
    const int m_tiles = (p.seq_len + kM0 - 1) / kM0;
    dim3 grid(p.q_heads, m_tiles, p.batch);
    dim3 block(kBlockSize);

    // Select kernel (single-pass).  This is the byte-identical legacy path used
    // whenever --splitk is not given (cfg.splitk == 0).
    auto launch = [&]() {
        if (varlen_mode) {
            if (p.mask)
                hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk1_varlen, grid, block, 0, nullptr, kparams);
            else
                hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0_varlen, grid, block, 0, nullptr, kparams);
        } else {
            if (p.mask)
                hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk1, grid, block, 0, nullptr, kparams);
            else
                hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0, grid, block, 0, nullptr, kparams);
        }
    };

    // -------------------------------------------------------------------------
    // Split-K two-pass setup (--splitk G, G>0).
    //
    // Split-K is DENSE-ONLY: the split grid's z-axis packs batch*num_splits, and
    // the split-major scratch layout assumes fixed-length batches.  Varlen has no
    // split path, so reject the combination cleanly rather than launch garbage.
    // -------------------------------------------------------------------------
    const int G = cfg.splitk;
    const bool splitk_mode = (G > 0);
    if (splitk_mode && varlen_mode) {
        fprintf(stderr, "Error: --splitk is dense-only; cannot combine with --varlen-seqs\n");
        return 1;
    }

    // Split-K scratch (allocated only when G>0).  scratch_o holds the G
    // normalized fp32 partial outputs (split-major [G][B][Hq][Sq][64]); scratch_lse
    // holds the matching per-row natural-log partial LSEs ([G][B][Hq][Sq]).
    float* scratch_o   = nullptr;
    float* scratch_lse = nullptr;
    FmhaFwdSplitParams   sp{};
    FmhaFwdCombineParams cp{};
    if (splitk_mode) {
        const size_t rows   = (size_t)p.batch * p.q_heads * p.seq_len;   // B*Hq*Sq
        const size_t n_o    = (size_t)G * rows * kHeadDim;               // partial O elems
        const size_t n_lse  = (size_t)G * rows;                          // partial LSE elems
        HIP_CHECK(hipMalloc(&scratch_o,   n_o   * sizeof(float)));
        HIP_CHECK(hipMalloc(&scratch_lse, n_lse * sizeof(float)));

        // Producer kernarg: carries the ordinary forward kernarg verbatim plus the
        // scratch pointers + split count.  split_idx is decoded in-kernel from
        // blockIdx.z, so the struct field is unused by the global (set to 0).
        sp.base        = kparams;
        sp.scratch_o   = scratch_o;
        sp.scratch_lse = scratch_lse;
        sp.num_splits  = G;
        sp.split_idx   = 0;

        // Combine kernarg: reads the same scratch (const) and writes the final O
        // (and optional LSE).  O strides mirror kparams (contiguous dense layout).
        cp.scratch_o      = scratch_o;
        cp.scratch_lse    = scratch_lse;
        cp.o              = kparams.o;
        cp.lse            = kparams.lse;
        cp.num_splits     = G;
        cp.seqlen_q       = p.seq_len;
        cp.nhead_q        = p.q_heads;
        cp.stride_o       = kparams.stride_o;
        cp.nhead_stride_o = kparams.nhead_stride_o;
        cp.batch_stride_o = kparams.batch_stride_o;
        cp.scale          = kparams.scale;
    }

    // Two-pass split-K launch: producer (G splits over the KV axis) then combine.
    // ★ TIMED-REGION REQUIREMENT: callers MUST bracket BOTH launches in the
    // hipEvent start/stop window — the combine pass is part of the split-K cost,
    // so timing the producer alone would report an INVALID (too-fast) result.
    // The timing loop below records start, calls this lambda (both launches), then
    // records stop, so both kernels are always inside the measured region.
    auto launch_splitk = [&]() {
        dim3 grid_split(p.q_heads, m_tiles, p.batch * G);
        if (p.mask)
            hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk1_split, grid_split, block, 0, nullptr, sp);
        else
            hipLaunchKernelGGL(fmha_fwd_d64_bf16_msk0_split, grid_split, block, 0, nullptr, sp);
        dim3 grid_combine(p.q_heads, m_tiles, p.batch);
        hipLaunchKernelGGL(fmha_fwd_d64_bf16_combine, grid_combine, block, 0, nullptr, cp);
    };

    // Single helper that runs whichever path is active (split-K or single-pass).
    auto launch_active = [&]() {
        if (splitk_mode) launch_splitk();
        else             launch();
    };

    // Warmup
    for (int i = 0; i < cfg.warmup; i++)
        launch_active();
    HIP_CHECK(hipDeviceSynchronize());

    // Per-iteration timing.  ★ For split-K, launch_active() == launch_splitk(),
    // which issues BOTH the producer and combine launches between the start/stop
    // event records below — the entire two-pass cost is inside the timed region.
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    std::vector<float> times(cfg.iters);
    for (int i = 0; i < cfg.iters; i++) {
        HIP_CHECK(hipEventRecord(ev_start));
        launch_active();   // single-pass: 1 launch; split-K: producer + combine
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
    }

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    float t_min = times[0], t_sum = 0;
    for (auto t : times) {
        t_sum += t;
        if (t < t_min) t_min = t;
    }
    float t_avg = t_sum / cfg.iters;

    // TFLOPS: 2 * B * H * Sq * Sk * D * 2 (Q*K and P*V GEMMs, multiply-add)
    // Causal mask computes ~half the Sq*Sk area, so halve the FLOP count to
    // match CK's convention (CK scales by mask.get_unmaskarea()).
    double flops = 4.0 * p.batch * p.q_heads * (double)p.seq_len * p.kv_seq_len * p.head_dim;
    if (p.mask) flops *= 0.5;
    double tflops_avg = flops / (t_avg * 1e-3) / 1e12;
    double tflops_min = flops / (t_min * 1e-3) / 1e12;

    // Output in format expected by run-benchmark.sh
    printf("Avg: %.3f ms\n", t_avg);
    printf("min=%.3f ms\n", t_min);
    printf("TFLOPS: %.2f\n", tflops_avg);

    // -------------------------------------------------------------------------
    // Correctness check (--verify).
    //
    // The timed loop above leaves bufs.d_O holding the LAST kernel result (the
    // active path — single-pass when G==0, two-pass split-K when G>0).  We re-run
    // the active path once more (post-timing) purely for clarity / determinism so
    // d_O is guaranteed to hold a complete result, then compare it against the
    // trusted gpu_ref_fmha_fwd oracle computed into a SEPARATE device buffer.
    //
    // Tolerance: the kernel output is bf16-truncated and (under split-K) is the
    // sum of G reweighted fp32 partials, so element-wise error is dominated by the
    // bf16 rounding of O (~2^-8 ≈ 0.4% relative at the output magnitude).  Because
    // the O magnitudes here are tiny (~1e-3..1e-2), an ABSOLUTE tolerance is not
    // discriminating, so we gate on two SCALE-AWARE metrics and require BOTH:
    //   * cosine similarity over all elements >= 0.99995 — catches directional /
    //     structural bugs (a wrong batch/split index garbles whole rows, which
    //     collapses cosine), and
    //   * relative L2 error ||O_kernel - O_ref|| / ||O_ref|| <= 2e-2 — catches
    //     magnitude bugs (a zeroed / wrong-scale output blows up relative L2 even
    //     when cosine cannot see it).
    // A correct kernel sits at cos~0.99999 and rel-L2~0.3% (the bf16 noise floor),
    // well inside both gates.  This matches the split-K spec's G-E2E acceptance
    // (oracle tol + cos >= 0.99995).  max_abs is printed for information only.
    // -------------------------------------------------------------------------
    int exit_code = 0;
    if (cfg.verify) {
        // Ensure d_O holds the kernel result (single-pass or two-pass split-K).
        launch_active();
        HIP_CHECK(hipDeviceSynchronize());

        // Copy kernel O back to host (bf16 -> fp32).
        bufs.copy_from_device();
        std::vector<float> kern_o(bufs.sz_O);
        for (size_t i = 0; i < bufs.sz_O; ++i)
            kern_o[i] = bf16_to_float(bufs.h_O[i]);

        // Trusted reference into a SEPARATE device buffer so it never clobbers the
        // kernel's d_O (split producer/combine and the ref share no storage).
        uint16_t* d_O_ref = nullptr;
        HIP_CHECK(hipMalloc(&d_O_ref, bufs.sz_O * sizeof(uint16_t)));
        HIP_CHECK(hipMemset(d_O_ref, 0, bufs.sz_O * sizeof(uint16_t)));

        // GpuRefParams strides are in bf16 ELEMENTS.  The bench packs Q/K/V/O
        // contiguously, so for D=hdim_dispatch the element strides are:
        //   seq   = D
        //   head  = Sq * D   (Skv * D for K/V)
        //   batch = Hq * Sq * D   (Hkv * Skv * D for K/V)
        // (We compute these directly rather than dividing bufs.stride_*/2 — the
        // bufs strides are in BYTES; these element strides are the /2 equivalents.)
        GpuRefParams gp{};
        gp.d_Q = reinterpret_cast<const uint16_t*>(bufs.d_Q);
        gp.d_K = reinterpret_cast<const uint16_t*>(bufs.d_K);
        gp.d_V = reinterpret_cast<const uint16_t*>(bufs.d_V);
        gp.d_O = d_O_ref;          // <-- separate ref output buffer
        gp.d_LSE = nullptr;        // O-only comparison
        gp.batch = p.batch; gp.q_heads = p.q_heads; gp.kv_heads = p.kv_heads; gp.gqa = p.gqa;
        gp.seq_len = p.seq_len; gp.kv_seq_len = p.kv_seq_len; gp.head_dim = p.head_dim;
        gp.stride_q_seq = D; gp.stride_k_seq = D; gp.stride_v_seq = D; gp.stride_o_seq = D;
        gp.stride_q_head  = p.seq_len * D;            gp.stride_q_batch = p.q_heads * p.seq_len * D;
        gp.stride_k_head  = p.kv_seq_len * D;         gp.stride_k_batch = p.kv_heads * p.kv_seq_len * D;
        gp.stride_v_head  = p.kv_seq_len * D;         gp.stride_v_batch = p.kv_heads * p.kv_seq_len * D;
        gp.stride_o_head  = p.seq_len * D;            gp.stride_o_batch = p.q_heads * p.seq_len * D;
        gp.mask = p.mask; gp.scalar = p.scalar();
        gp.d_seqstart_q = nullptr; gp.d_seqstart_k = nullptr;   // dense only

        gpu_ref_fmha_fwd(gp);
        HIP_CHECK(hipDeviceSynchronize());

        // Copy reference O back to host (bf16 -> fp32).
        std::vector<uint16_t> h_O_ref(bufs.sz_O);
        HIP_CHECK(hipMemcpy(h_O_ref.data(), d_O_ref, bufs.sz_O * sizeof(uint16_t),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_O_ref));

        // Compare over all B*Hq*Sq*D elements: max abs error, cosine, relative L2.
        double max_abs = 0.0, dot = 0.0, na = 0.0, nb = 0.0, sse = 0.0;
        for (size_t i = 0; i < bufs.sz_O; ++i) {
            const float a = kern_o[i];
            const float b = bf16_to_float(h_O_ref[i]);
            const double d = (double)a - (double)b;
            if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
            sse += d * d;
            dot += (double)a * (double)b;
            na  += (double)a * (double)a;
            nb  += (double)b * (double)b;
        }
        const double denom = std::sqrt(na) * std::sqrt(nb);
        const double cos = denom > 0.0 ? dot / denom : 1.0;
        // Relative L2 vs the reference norm.  A zero reference (||O_ref||==0) makes
        // any nonzero diff infinitely wrong, so map that to +inf (-> FAIL).
        const double ref_norm = std::sqrt(nb);
        const double rel_l2 = ref_norm > 0.0 ? std::sqrt(sse) / ref_norm : INFINITY;

        const double kCosTol   = 0.99995;  // directional/structural criterion
        const double kRelL2Tol = 2e-2;     // magnitude criterion (2% rel L2)
        const bool pass = (cos >= kCosTol) && (rel_l2 <= kRelL2Tol);
        printf("VERIFY: %s max_abs=%.6g cos=%.8f rel_l2=%.6g\n",
               pass ? "PASS" : "FAIL", max_abs, cos, rel_l2);
        if (!pass) exit_code = 1;        // ★ nonzero exit so gates/scripts detect it
    }

    // Free split-K scratch (no-op when single-pass).
    if (splitk_mode) {
        HIP_CHECK(hipFree(scratch_o));
        HIP_CHECK(hipFree(scratch_lse));
    }

    return exit_code;
}
