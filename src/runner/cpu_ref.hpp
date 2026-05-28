// FP32 CPU reference implementation and verification for FMHA forward.
//
// cpu_ref_verify() computes FMHA on the host using the same BF16
// truncation pipeline as the GPU kernels, then compares against the
// kernel output stored in bufs.h_O.  It reports per-element absolute
// error, relative error (for |ref| >= 0.01), and per-row cosine
// similarity.  Results are printed to stdout and returned in
// CpuRefResult.
//
// The precision pipeline emulates the kernel's data path:
//   1. QK GEMM:  BF16 inputs -> FP32 accumulation -> FP32 S
//   2. Softmax:  FP32 S -> FP32 P
//   3. P trunc:  FP32 P -> BF16 -> FP32 (matches v_cvt_pk_bf16_f32)
//   4. PV GEMM:  BF16 P * BF16 V -> FP32 accumulation -> FP32 O_ref
//
// The inner loops are parallelized with OpenMP.

#pragma once
#include <cstddef>

struct FmhaParams;
class FmhaBuffers;

// Verification result returned by cpu_ref_verify().
struct CpuRefResult {
    bool pass;                    // true if all tolerances met

    double max_abs;               // worst absolute error across all elements
    double max_rel;               // worst relative error (|ref| >= 0.01 only)
    double min_cos;               // worst per-row cosine similarity
    double mean_cos;              // mean per-row cosine similarity

    size_t mismatch;              // elements exceeding abs tolerance (0.001)
    size_t nonfinite;             // NaN/Inf elements in kernel output
    size_t total_elems;           // total elements compared

    size_t cos_fail;              // rows below cosine threshold (0.99998)
    size_t cos_rows;              // rows with nonzero norm (denominator > 0)

    // Location of worst absolute error
    int worst_b, worst_h, worst_i, worst_d;
    float worst_ref, worst_kern;

    // Location of worst cosine similarity
    int cos_wb, cos_wh, cos_wi;
};

// Run CPU reference FMHA and compare against bufs.h_O.
// Caller must call bufs.copy_from_device() before this.
CpuRefResult cpu_ref_verify(const FmhaParams& p, const FmhaBuffers& bufs);
