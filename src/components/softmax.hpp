#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Phase 1 Kernel 4 -- test_softmax (scale + mask + exp2 + rowsum + bf16 cast)
//
// Reproduces CK's single-tile softmax, matching golden dump_reg slot 3 (P).
// Input: RAW unscaled S_acc (256×32 fp32) from Kernel 2.
// Output: P (256×32 fp32, bf16-promoted), rmax (256×1 fp32), rsum (256×1 fp32).
//
// Pipeline:
//   1. Mask: S[n_col >= (seqlen_k - kv_offset)] = -INFINITY
//   2. Rmax: intra-lane max + 1 cross-half ds_bpermute (K3 algorithm)
//   3. Exp2: P = exp2(scale_s_log2e * (S - rmax))  [CK FAST_EXP2=1]
//   4. Rsum: intra-lane sum + 1 cross-half ds_bpermute
//   5. Cast: P_bf16 = truncate(P_fp32)
//
// S_acc layout: m=(lane%32)+32*warp, n=(r/8)*16+(lane/32)*8+(r%8).
// scale_s_log2e = scale_s * log2(e).

// ds_bpermute: read src_lane's copy of val.
__device__ __forceinline__ float sm_bpermute(int src_lane, float val) {
    int src_byte = src_lane * 4;
    int ret;
    asm volatile("ds_bpermute_b32 %0, %1, %2\n"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(ret) : "v"(src_byte), "v"(val) : "memory");
    return __builtin_bit_cast(float, ret);
}

// bf16 truncation: keep upper 16 bits of fp32.
__device__ __forceinline__ float bf16_trunc_dev(float v) {
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    u &= 0xFFFF0000u;
    return __builtin_bit_cast(float, u);
}

__global__ void test_softmax_kernel(const float* __restrict__ s_acc,
                                    float* __restrict__ p_out,
                                    float* __restrict__ rmax_out,
                                    float* __restrict__ rsum_out,
                                    int seqlen_k,
                                    int kv_offset,
                                    float scale_s) {
    const int tid     = threadIdx.x;
    const int lane_id = tid & 63;
    const int k_sub   = lane_id >> 5;

    const float log2e = 1.4426950408889634f;
    const float scale_s_log2e = scale_s * log2e;
    const int tile_cols = seqlen_k - kv_offset; // valid N-columns in this tile

    // Load S_acc (32 regs), apply seqlen_k mask, compute intra-lane max
    float s[32];
    float local_max = -INFINITY;

    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        int n_col = (r / 8) * 16 + k_sub * 8 + (r % 8);
        float v = s_acc[tid * 32 + r];
        if (n_col >= tile_cols)
            v = -INFINITY;
        s[r] = v;
        local_max = fmaxf(local_max, v);
    }

    // Cross-half exchange for rmax (1 ds_bpermute)
    int partner = (tid & ~63) | ((lane_id ^ 32) & 63);
    float other_max = sm_bpermute(partner, local_max);
    float rmax = fmaxf(local_max, other_max);

    rmax_out[tid] = rmax;

    // Compute exp2 and accumulate intra-lane rowsum
    float local_sum = 0.0f;

    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        float p;
        if (s[r] == -INFINITY) {
            p = 0.0f;
        } else {
            float arg = scale_s_log2e * (s[r] - rmax);
            p = __builtin_amdgcn_exp2f(arg);
        }
        local_sum += p;
        // Truncate to bf16 and promote back to fp32
        s[r] = bf16_trunc_dev(p);
    }

    // Cross-half exchange for rowsum (1 ds_bpermute)
    float other_sum = sm_bpermute(partner, local_sum);
    float rsum = local_sum + other_sum;

    rsum_out[tid] = rsum;

    // Output P (bf16-promoted fp32)
    #pragma unroll
    for (int r = 0; r < 32; ++r) {
        p_out[tid * 32 + r] = s[r];
    }
}
