// FmhaBuffers implementation — see buffers.hpp for API documentation.

#include "buffers.hpp"
#include "fmha_params.hpp"
#include "bf16_utils.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t e = (expr);                                                 \
        if (e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", e,                \
                    hipGetErrorString(e), __FILE__, __LINE__);                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

FmhaBuffers::FmhaBuffers(const FmhaParams& p) : params_(p) {
    const int B    = p.batch;
    const int Hq   = p.q_heads;
    const int Hkv  = p.kv_heads;
    const int Sq   = p.seq_len;
    const int Skv  = p.kv_seq_len;
    const int Dpad = p.hdim_dispatch();

    const bool varlen_mode = !p.varlen_seqs.empty();
    if (varlen_mode)
        for (int s : p.varlen_seqs) total_seqlen += s;

    sz_Q   = varlen_mode ? (size_t)Hq  * total_seqlen * Dpad
                         : (size_t)B * Hq  * Sq  * Dpad;
    sz_K   = varlen_mode ? (size_t)Hkv * total_seqlen * Dpad
                         : (size_t)B * Hkv * Skv * Dpad;
    sz_V   = varlen_mode ? (size_t)Hkv * total_seqlen * Dpad
                         : (size_t)B * Hkv * Skv * Dpad;
    sz_O   = varlen_mode ? (size_t)Hq  * total_seqlen * Dpad
                         : (size_t)B * Hq  * Sq  * Dpad;
    sz_LSE = varlen_mode ? (size_t)Hq  * total_seqlen
                         : (size_t)B * Hq  * Sq;

    // Byte strides
    stride_q_seq   = Dpad * 2;
    stride_q_tg    = p.sub_Q * stride_q_seq;
    stride_q_head  = varlen_mode ? total_seqlen * stride_q_seq
                                 : Sq * stride_q_seq;
    stride_q_batch = varlen_mode ? 0 : Hq * stride_q_head;

    stride_k_seq   = Dpad * 2;
    stride_k_head  = varlen_mode ? total_seqlen * stride_k_seq
                                 : Skv * stride_k_seq;
    stride_k_batch = varlen_mode ? 0 : Hkv * stride_k_head;

    // Allocate host
    h_Q.resize(sz_Q, 0);
    h_K.resize(sz_K, 0);
    h_V.resize(sz_V, 0);

    // Allocate device
    HIP_CHECK(hipMalloc(&d_Q,   sz_Q   * 2));
    HIP_CHECK(hipMalloc(&d_K,   sz_K   * 2));
    HIP_CHECK(hipMalloc(&d_V,   sz_V   * 2));
    HIP_CHECK(hipMalloc(&d_O,   sz_O   * 2));
    HIP_CHECK(hipMalloc(&d_LSE, sz_LSE * 4));

    // Varlen offset tables
    if (varlen_mode) {
        const int VB = (int)p.varlen_seqs.size();
        std::vector<uint32_t> offsets(VB + 1, 0);
        for (int i = 0; i < VB; i++)
            offsets[i + 1] = offsets[i] + p.varlen_seqs[i];

        HIP_CHECK(hipMalloc(&d_qseq, (VB + 1) * 4));
        HIP_CHECK(hipMalloc(&d_kseq, (VB + 1) * 4));
        HIP_CHECK(hipMalloc(&d_qpad, (VB + 1) * 4));
        HIP_CHECK(hipMalloc(&d_kpad, (VB + 1) * 4));
        HIP_CHECK(hipMemcpy(d_qseq, offsets.data(), (VB + 1) * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_kseq, offsets.data(), (VB + 1) * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_qpad, offsets.data(), (VB + 1) * 4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_kpad, offsets.data(), (VB + 1) * 4, hipMemcpyHostToDevice));
    }
}

FmhaBuffers::~FmhaBuffers() {
    if (d_qseq) hipFree(d_qseq);
    if (d_kseq) hipFree(d_kseq);
    if (d_qpad) hipFree(d_qpad);
    if (d_kpad) hipFree(d_kpad);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);
    hipFree(d_LSE);
}

void FmhaBuffers::fill_random(unsigned seed) {
    const int B    = params_.batch;
    const int Hq   = params_.q_heads;
    const int Hkv  = params_.kv_heads;
    const int Sq   = params_.seq_len;
    const int Skv  = params_.kv_seq_len;
    const int Dlog = params_.head_dim;
    const int Dpad = params_.hdim_dispatch();
    const bool varlen_mode = !params_.varlen_seqs.empty();

    srand(seed);
    auto rand_bf16 = []() -> uint16_t {
        float v = ((rand() % 1000) - 500) / 5000.0f;
        return float_to_bf16(v);
    };
    auto fill_padded = [&](std::vector<uint16_t>& buf, int rows) {
        for (int r = 0; r < rows; r++)
            for (int d = 0; d < Dlog; d++)
                buf[(size_t)r * Dpad + d] = rand_bf16();
    };
    if (varlen_mode) {
        fill_padded(h_Q, Hq  * total_seqlen);
        fill_padded(h_K, Hkv * total_seqlen);
        fill_padded(h_V, Hkv * total_seqlen);
    } else {
        fill_padded(h_Q, B * Hq  * Sq);
        fill_padded(h_K, B * Hkv * Skv);
        fill_padded(h_V, B * Hkv * Skv);
    }

    // Probe env vars
    if (const char* p = getenv("PROBE_V_CONST")) {
        float v = atof(p);
        uint16_t bf = float_to_bf16(v);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < Hkv; ++h) {
                size_t base = ((size_t)b * Hkv + h) * Skv * Dpad;
                for (int r = 0; r < Skv; ++r)
                    for (int d = 0; d < Dlog; ++d)
                        h_V[base + (size_t)r * Dpad + d] = bf;
            }
        fprintf(stderr, "[PROBE H] V tensor set to constant %g\n", v);
    }
    if (const char* p = getenv("PROBE_Q_CONST")) {
        float v = atof(p);
        uint16_t bf = float_to_bf16(v);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < Hq; ++h) {
                size_t base = ((size_t)b * Hq + h) * Sq * Dpad;
                for (int r = 0; r < Sq; ++r)
                    for (int d = 0; d < Dlog; ++d)
                        h_Q[base + (size_t)r * Dpad + d] = bf;
            }
        fprintf(stderr, "[PROBE K] Q tensor set to constant %g\n", v);
    }
    if (const char* p = getenv("PROBE_K_CONST")) {
        float v = atof(p);
        uint16_t bf = float_to_bf16(v);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < Hkv; ++h) {
                size_t base = ((size_t)b * Hkv + h) * Skv * Dpad;
                for (int r = 0; r < Skv; ++r)
                    for (int d = 0; d < Dlog; ++d)
                        h_K[base + (size_t)r * Dpad + d] = bf;
            }
        fprintf(stderr, "[PROBE K_K] K tensor set to constant %g\n", v);
    }
    if (const char* p = getenv("PROBE_ZERO_TILE")) {
        int t_lo = -1, t_hi = -1;
        if (std::string(p) == "last") { t_lo = std::max(0, Skv - 32); t_hi = Skv; }
        else { int t = atoi(p); t_lo = t * 32; t_hi = std::min(Skv, t_lo + 32); }
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < Hkv; ++h) {
                size_t base = ((size_t)b * Hkv + h) * Skv * Dpad;
                for (int r = t_lo; r < t_hi; ++r)
                    for (int d = 0; d < Dpad; ++d)
                        h_V[base + (size_t)r * Dpad + d] = 0;
            }
        fprintf(stderr, "[PROBE A] Zeroed V rows [%d, %d) per (b,hkv).\n", t_lo, t_hi);
    }
}

void FmhaBuffers::copy_to_device() {
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), sz_Q * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), sz_K * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), sz_V * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_O, 0, sz_O * 2));
    HIP_CHECK(hipMemset(d_LSE, 0, sz_LSE * 4));
}

void FmhaBuffers::copy_from_device() {
    h_O.resize(sz_O);
    HIP_CHECK(hipMemcpy(h_O.data(), d_O, sz_O * 2, hipMemcpyDeviceToHost));
}
