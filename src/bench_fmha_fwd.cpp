// FMHA forward benchmark — timing-only CLI for native HIP kernel performance.
//
// Dispatches the compiled-in D64 bf16 kernel with configurable problem
// dimensions, runs warmup + timed iterations, and reports per-iteration
// TFLOPS.  No correctness verification — use test_fmha_fwd_d64 for that.
//
// Usage:
//   bench_fmha_fwd -b 2 -h 16 -s 4096 -d 64

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

// Kernel declarations (defined in src/fused/kernel.cpp)
__global__ void fmha_fwd_d64_bf16_msk0(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk1(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk0_varlen(FmhaFwdParams params);
__global__ void fmha_fwd_d64_bf16_msk1_varlen(FmhaFwdParams params);

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

    // Select kernel
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

    // Warmup
    for (int i = 0; i < cfg.warmup; i++)
        launch();
    HIP_CHECK(hipDeviceSynchronize());

    // Per-iteration timing
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    std::vector<float> times(cfg.iters);
    for (int i = 0; i < cfg.iters; i++) {
        HIP_CHECK(hipEventRecord(ev_start));
        launch();
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

    return 0;
}
