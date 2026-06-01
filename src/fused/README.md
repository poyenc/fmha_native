# src/fused — the production FMHA forward kernel

This directory is the kernel that ships: a D64 BF16 fused multi-head attention forward kernel
for gfx942 (MI300X, CDNA3). See [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) for
the pipeline walkthrough and data layouts, and the repo [`README.md`](../../README.md) for
build/test/benchmark instructions.

## Files

| File | Role |
|------|------|
| `kernel.cpp` | The 4 `__global__` entry points (dense/varlen × nomask/causal) and the grid mapping. |
| `pipeline.hpp` | The per-block device function — the whole forward pass for one M-tile. |
| `op_lds.hpp` | K/V staging DRAM→LDS (async copy), the LDS byte layout, waitcnt helpers. |
| `op_gemm.hpp` | GEMM0 (Q·Kᵀ) and GEMM1 (P·V) via `v_mfma_f32_32x32x8_bf16`. |
| `op_softmax.hpp` | Mask, row-max, exp2, row-sum, online rescale. |
| `op_epilog.hpp` | O normalize + bf16 store (Default2D) + optional LSE. |

## Performance

Measured on **MI300X (gfx942)**, d64 bf16, via `scripts/run-benchmark.sh` (6 runs per config,
first dropped). Compared against Composable Kernel (CK) `tile_example_fmha_fwd` built and run
back-to-back on the same GPU. TFLOPS is average-based (both tools); the FLOP convention matches
within <0.1% (see ARCHITECTURE §6), so the comparison is apples-to-apples.

`native/CK` is the throughput ratio (native TFLOPS ÷ CK TFLOPS); higher is better, ≥100% means
native matches or beats CK.

### Non-causal (`--mask 0`)

| B | H | S | native TFLOPS | CK TFLOPS | native/CK |
|---|---|------|------|------|------|
| 8 | 16 | 1024  | 291.7 | 318.4 | 91.6% |
| 4 | 16 | 2048  | 340.1 | 356.1 | 95.5% |
| 2 | 16 | 4096  | 367.9 | 378.4 | 97.2% |
| 1 | 8  | 8192  | 327.1 | 353.7 | 92.5% |
| 1 | 16 | 8192  | 381.1 | 396.1 | 96.2% |
| 1 | 8  | 16384 | 398.1 | 415.1 | 95.9% |
| 1 | 4  | 32768 | 397.2 | 412.6 | 96.3% |
| 1 | 2  | 40000 | 327.9 | 345.3 | 95.0% |

### Causal (`--mask 1`)

| B | H | S | native TFLOPS | CK TFLOPS | native/CK |
|---|---|------|------|------|------|
| 8 | 16 | 1024  | 206.4 | 228.0 | 90.5% |
| 4 | 16 | 2048  | 263.3 | 289.7 | 90.9% |
| 2 | 16 | 4096  | 297.8 | 312.8 | 95.2% |
| 1 | 8  | 8192  | 248.6 | 253.3 | 98.2% |
| 1 | 16 | 8192  | 352.2 | 366.7 | 96.1% |
| 1 | 8  | 16384 | 370.2 | 378.8 | 97.7% |
| 1 | 4  | 32768 | 387.1 | 397.8 | 97.3% |
| 1 | 2  | 40000 | 299.7 | 241.7 | 124.0% |

Notes:
- Causal TFLOPS use the ×0.5 FLOP convention (half the score matrix is masked), matching CK.
- Small-S causal configs gain least from the causal M-tile load-balance reversal (few M-tiles
  → little tail to recover); large-S configs benefit most. See ARCHITECTURE §5.
- The `B1 H2 S40000` config is ragged/low-occupancy; treat it as an outlier for parity reading.
