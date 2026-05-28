// Host and device memory management for FMHA forward pass.
//
// FmhaBuffers owns all Q/K/V/O/LSE buffers on both host and device,
// computes byte strides from FmhaParams, and handles data transfers.
// It also manages varlen cumulative offset tables when variable-length
// batching is active.
//
// Typical usage:
//   FmhaBuffers bufs(params);
//   bufs.fill_random(42);
//   bufs.copy_to_device();
//   /* launch kernel */
//   bufs.copy_from_device();   // copies O (and LSE) back to host

#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <vector>

struct FmhaParams;

class FmhaBuffers {
public:
    // Allocate host vectors and device memory sized for the given params.
    explicit FmhaBuffers(const FmhaParams& p);

    // Free all device memory.
    ~FmhaBuffers();

    FmhaBuffers(const FmhaBuffers&) = delete;
    FmhaBuffers& operator=(const FmhaBuffers&) = delete;

    // Fill Q/K/V with deterministic random BF16 values.  Positions beyond
    // head_dim (in padded mode) are zero.  Also applies PROBE_* env-var
    // overrides for debugging (PROBE_V_CONST, PROBE_Q_CONST, etc.).
    void fill_random(unsigned seed);

    // Upload Q/K/V to device; zero-initialize O and LSE on device.
    void copy_to_device();

    // Download O from device to h_O (resized on first call).
    void copy_from_device();

    // --- Host buffers (BF16 stored as uint16_t) ---
    std::vector<uint16_t> h_Q, h_K, h_V, h_O;

    // --- Device pointers (BF16 for Q/K/V/O, FP32 for LSE) ---
    void *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    void *d_O = nullptr, *d_LSE = nullptr;

    // --- Varlen offset table device pointers (null when fixed-length) ---
    void *d_qseq = nullptr, *d_kseq = nullptr;
    void *d_qpad = nullptr, *d_kpad = nullptr;

    // --- Element counts (number of BF16 elements, not bytes) ---
    size_t sz_Q, sz_K, sz_V, sz_O, sz_LSE;

    // Total tokens across all varlen batches (0 for fixed-length).
    uint32_t total_seqlen = 0;

    // --- Byte strides (matching HSACO kernarg layout) ---
    uint32_t stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch;
    uint32_t stride_k_seq, stride_k_head, stride_k_batch;

private:
    const FmhaParams& params_;
};
