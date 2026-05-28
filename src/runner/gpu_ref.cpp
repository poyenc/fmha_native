// Naive FP32 GPU reference kernel — see gpu_ref.hpp for API documentation.

#include "gpu_ref.hpp"
#include "bf16_utils.hpp"
#include <hip/hip_runtime.h>
#include <cmath>

__global__ void gpu_ref_fmha_kernel(GpuRefParams p) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_rows = p.batch * p.q_heads * p.seq_len;
    if (tid >= total_rows) return;

    const int b  = tid / (p.q_heads * p.seq_len);
    const int hq = (tid / p.seq_len) % p.q_heads;
    const int i  = tid % p.seq_len;
    const int hkv = hq / p.gqa;

    const int D = p.head_dim;

    int i_q = i;
    int i_k = 0;
    int Sq_b = p.seq_len;
    int Skv_b = p.kv_seq_len;

    if (p.d_seqstart_q) {
        uint32_t qstart = p.d_seqstart_q[b];
        Sq_b = p.d_seqstart_q[b + 1] - qstart;
        if (i >= Sq_b) return;
        i_q = qstart + i;

        uint32_t kstart = p.d_seqstart_k[b];
        Skv_b = p.d_seqstart_k[b + 1] - kstart;
        i_k = kstart;
    }

    const uint16_t* Q_row  = p.d_Q + (size_t)b * p.stride_q_batch + (size_t)hq  * p.stride_q_head + (size_t)i_q * p.stride_q_seq;
    const uint16_t* K_base = p.d_K + (size_t)b * p.stride_k_batch + (size_t)hkv * p.stride_k_head + (size_t)i_k * p.stride_k_seq;
    const uint16_t* V_base = p.d_V + (size_t)b * p.stride_v_batch + (size_t)hkv * p.stride_v_head + (size_t)i_k * p.stride_v_seq;
    uint16_t*       O_row  = p.d_O + (size_t)b * p.stride_o_batch + (size_t)hq  * p.stride_o_head + (size_t)i_q * p.stride_o_seq;

    float max_s = -INFINITY;
    for (int j = 0; j < Skv_b; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        if (s > max_s) max_s = s;
    }

    float sum_exp = 0;
    for (int j = 0; j < Skv_b; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        float e = (s == -INFINITY) ? 0.0f : expf(s - max_s);
        sum_exp += e;
    }

    float inv_sum = (sum_exp > 0) ? 1.0f / sum_exp : 0.0f;

    float O_acc[256];
    for (int d = 0; d < D; d++) O_acc[d] = 0;

    for (int j = 0; j < Skv_b; j++) {
        float s = 0;
        for (int d = 0; d < D; d++) {
            float q = bf16_to_float(Q_row[d]);
            float k = bf16_to_float(K_base[(size_t)j * p.stride_k_seq + d]);
            s += q * k;
        }
        s *= p.scalar;
        if (p.mask && j > i + (Skv_b - Sq_b)) s = -INFINITY;
        float p_val = ((s == -INFINITY) ? 0.0f : expf(s - max_s)) * inv_sum;

        uint16_t p_bf16 = float_to_bf16(p_val);
        float p_trunc = bf16_to_float(p_bf16);

        for (int d = 0; d < D; d++) {
            float v = bf16_to_float(V_base[(size_t)j * p.stride_v_seq + d]);
            O_acc[d] += p_trunc * v;
        }
    }

    for (int d = 0; d < D; d++) {
        O_row[d] = float_to_bf16(O_acc[d]);
    }

    if (p.d_LSE) {
        float lse_val = (sum_exp > 0) ? logf(sum_exp) + max_s : -INFINITY;
        size_t lse_idx = (size_t)b * (p.stride_q_batch / p.stride_q_seq)
                       + (size_t)hq * (p.stride_q_head / p.stride_q_seq)
                       + i_q;
        p.d_LSE[lse_idx] = lse_val;
    }
}

void gpu_ref_fmha_fwd(const GpuRefParams& p, hipStream_t stream) {
    const int total_rows = p.batch * p.q_heads * p.seq_len;
    const int block = 256;
    const int grid = (total_rows + block - 1) / block;
    hipLaunchKernelGGL(gpu_ref_fmha_kernel, dim3(grid), dim3(block), 0, stream, p);
}
