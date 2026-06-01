# fmha_native — Architecture & Internals

This document is the deep-dive companion to `README.md` (build/test) and `CLAUDE.md`
(quick index). It explains how the kernel works, the data layouts it relies on, where the
project stands against Composable Kernel (CK), and how the CK comparison is done. It is
written for a GPU kernel engineer who is new to *this* repository.

## 1. What this is

`fmha_native` is a hand-written HIP implementation of a **D64 (head-dim 64) BF16 fused
multi-head attention (FMHA) forward kernel** for AMD **gfx942 (MI300X, CDNA3)**. The project
goal is **ISA- and performance-parity with Composable Kernel (CK)** for the same problem:
the kernel is built bottom-up so its generated assembly and its runtime match CK's
equivalent D64 FMHA forward kernel as closely as possible.

Scope: forward pass only; head dim fixed at 64; BF16 inputs with FP32 accumulation; dense
and variable-length (varlen) batching; optional causal mask; optional log-sum-exp (LSE)
output.

## 2. Source map

```
src/
  fused/            THE PRODUCTION KERNEL (this is what ships)
    kernel.cpp        4 __global__ entry points (dense/varlen x nomask/causal)
    pipeline.hpp      per-block device function = the whole forward pass for one block
    op_lds.hpp        K/V staging DRAM->LDS (async copy) + LDS byte layout + waitcnt helpers
    op_gemm.hpp       GEMM0 (Q.K^T) and GEMM1 (P.V) via v_mfma_f32_32x32x8_bf16
    op_softmax.hpp    mask, row-max, exp2, row-sum, online rescale
    op_epilog.hpp     O normalize + bf16 store (Default2D) + optional LSE
  components/        7 STANDALONE building-block kernels — TEST ONLY, not used by fused/
    k_lds, qk_gemm, row_max, softmax, v_lds, pv_gemm, epilog
  components_ref/    CPU/GPU reference oracles paired with the components
  runner/            shared infra: params, device buffers, CPU ref, naive GPU ref, bf16 utils
tests/              GoogleTest: full-kernel suite + gpu-ref suite + 7 component suites
scripts/            run-gates.sh (build+test+asm), run-benchmark.sh, common.sh
docs/               this file (ARCHITECTURE.md)
```

`src/fused/` is self-contained: it includes only its own `op_*` siblings plus `runner/params.hpp`.
The `components/` + `components_ref/` tree exists purely so each pipeline stage can be unit-tested
in isolation against a golden dump; the production kernel does **not** include it.

### Key tile constants (`src/runner/params.hpp`)
- `kM0 = 128` query rows per M-tile (one threadblock's Q work)
- `kN0 = 64` key columns per KV-tile
- `kK0 = 32` contraction depth per GEMM0 step
- `kBlockSize = 256` threads = `kNumWarps = 4` waves x `kWarpSize = 64` lanes (CDNA is 64-wide)
- `kLdsBytes = 13824` LDS bytes per block

## 3. Kernel pipeline (one block)

`kernel.cpp` launches a 3D grid `dim3(q_heads, m_tiles, batch)`, so for each block
`blockIdx.x = head`, `blockIdx.y = m_tile`, `blockIdx.z = batch`. Each block computes one
128-row M-tile of the output for one (batch, head). `pipeline.hpp` is the per-block body:

1. **Q load** — each thread loads its Q registers for the M-tile; varlen vs dense changes
   only the base offsets / seqlens.
2. **Prologue** — issue the first K sub-tile async copy into LDS.
3. **Tile loop over KV tiles** (`kN0 = 64` keys each):
   - **GEMM0**: Q·Kᵀ → `S_acc` (two `kK0`-deep sub-tiles, MFMA).
   - **Softmax**: `softmax_mask` (causal/boundary) → `softmax_row_max` → `softmax_exp2`
     → `softmax_row_sum`. Online: a running max/sum is carried across tiles.
   - **V staging**: V is loaded DRAM → registers → shuffled (`v_perm_b32`) → LDS.
   - **GEMM1**: P·V → `O_acc` (MFMA). When the running max grows, `O_acc` and the running
     sum are rescaled by `exp2(scale*(old_max - new_max))` (online softmax correction).
   - The next tile's K is prefetched while GEMM1 runs.
4. **Epilogue** (`op_epilog.hpp`): normalize `O_acc` by the reciprocal running sum, truncate
   to bf16, store to DRAM (Default2D — no shuffle), and optionally write LSE.

**LDS buffering** uses `LdsSeq[4] = {1, 2, 1, 0}` to index distinct LDS regions so K sub-tile
prefetch, V halves, and GEMM consumption don't alias (see the constant's comment in
`pipeline.hpp`).

**`sched_barrier` calls**: `pipeline.hpp` contains `__builtin_amdgcn_sched_barrier(...)` calls
placed to mirror CK's barrier structure so the compiler's instruction scheduling lines up with
CK's. This is a **codegen-parity** aid, not a measured perf lever — barrier-for-barrier matching
on its own produced ~0% runtime change (verified). Keep them for ISA alignment, but do not
expect to tune performance by moving them.

## 4. Data layouts & swizzles

These conventions are shared across `op_gemm.hpp`, `op_softmax.hpp`, and `op_epilog.hpp`.

- **TransposedC register layout** (the universal layout for Q, S_acc, P, O_acc, O_final):
  for a thread, the M-row it owns is `m_row = (lane % 32) + 32 * warp`; for register index
  `r` in 0..15, the other-dimension index is `free = (r/8)*16 + (lane/32)*8 + (r%8)`.
  There are no cross-lane shuffles in the register path (except the one V shuffle below).
- **k_sub halves**: `k_sub = lane / 32`. The 32 lanes within one k_sub half hold the *same*
  32 columns; the two halves are complementary. A row reduction therefore needs only an
  intra-lane reduction over the 16+16 registers plus **one** `ds_bpermute` with partner lane
  `lane ^ 32` to merge the complementary half — no butterfly.
- **GEMM operand convention**: A from LDS, B from registers. GEMM0: A = K (LDS), B = Q (reg);
  GEMM1: A = V (LDS), B = P (reg). Reason: TransposedC maps each lane to B's dimension, so
  B = Q/P means each lane owns exactly one M-row.
- **SwizzleA**: GEMM0 applies a bit-2/bit-3 swap ("SwizzleA") to its K LDS reads; GEMM1 does
  **not** swizzle V. `O_acc` inherits SwizzleA from P, so the epilogue store un-applies it on
  the head-dim index.
- **bf16 cast = truncation** (not round-to-nearest-even): the bf16 helper drops the low 16
  mantissa bits (`u >> 16`) with no rounding term, matching the hardware data path.
- **base-2 softmax**: the softmax runs in base 2. `__builtin_amdgcn_exp2f` lowers to
  `v_exp_f32` (2^x). `params.scale` is pre-multiplied by `log2(e)` (i.e. `log2(e)/sqrt(d)`),
  so `exp2(scale * x) = exp(x / sqrt(d))`. Note that `__builtin_amdgcn_logf` lowers to
  `v_log_f32`, which is **log2**, not natural log — relevant to the LSE math in `op_epilog.hpp`,
  where `(log2(rsum) + scale*rmax) * ln(2)` yields the natural-log LSE.

## 5. Performance status (vs CK)

Measured on MI300X (gfx942), d64 bf16, both sides built and benchmarked back-to-back.

- **Non-causal (mask=0)**: ~95–99% of CK across the sweep — effectively at parity.
- **Causal (mask=1)**: ~97–99% of CK across mid/large sequence lengths after the M-tile
  load-balance fix below (was 84–93% before it).

### The causal load-balance fix (commit `1c24bab`)
Causal work per block is **linear in `m_tile`** (M-tile *i* processes ~*i*+1 KV-tiles). With
the natural grid order `m_tile_idx = blockIdx.y`, all the heavy high-index tiles land in the
final wave of blocks with no light work left to overlap — a tail that drops time-averaged
occupancy and inflates kernel cycles even though total work is unchanged. The masked entries
in `kernel.cpp` therefore **reverse** the M-tile index:

```cpp
const int m_tile_idx = gridDim.y - 1 - blockIdx.y;   // msk1 / msk1_varlen only
```

so the heavy tiles launch first. This matches CK's scheme. The no-mask entries keep natural
order (uniform work, no imbalance). Profiled effect at one config (s8k, B1H16): achieved
occupancy 387 → 464 (CK ≈ 468), kernel cycles 787k → 646k (CK ≈ 635k), with identical active
cycles (same work) — i.e. the gap was a scheduling tail, not extra compute. Net +7–16% on
causal depending on size.

## 6. CK comparison methodology

Parity work compares two builds of the same problem: this repo (`fmha_native`) and CK's
`tile_example_fmha_fwd`. The two were built in separate ROCm containers with each repo
bind-mounted; any equivalent setup works.

- **Benchmark protocol**: warm up, then repeat many iterations; compare **minimum wall-clock
  ms** (convention-free) or **average-based TFLOPS** — but compare like with like. Each tool
  derives its own TFLOPS from its own average time, so do not put one tool's min-ms next to the
  other's avg-TFLOPS. The two FLOP formulas differ by <0.1% for causal (this repo counts
  `0.5*S^2`, CK counts the exact `S*(S+1)/2`; they converge for S ≥ 1024), so TFLOPS is a fair
  cross-tool metric as long as both are average-based.
- **Assembly diffing** (how ISA parity is checked): build, then extract the kernel's generated
  assembly. The `--save-temps` compile flag emits `build/kernel-hip-amdgcn-amd-amdhsa-gfx942.s`;
  `scripts/run-gates.sh` (stage G3) copies it to `native_d64_kernel.s` at the repo root. CK's
  reference assembly lives under `asm_compare/`. Comparison is done by reading both `.s` files
  and diffing the instruction mix and scheduling (e.g. counts of MFMA, VALU, LDS, `s_waitcnt`
  placement, register usage). This is plain assembly reading — no special tooling is required.

## 7. Verified dead-ends (do not re-chase)

These were investigated, measured, and ruled out. They are recorded so they are not reopened:

- **The `softmax_mask` OR-scan**: the constant-offset causal predicate makes LLVM emit a deep
  serial `s_or_b64` chain on the edge tile. Rewriting it to break the chain was **perf-neutral**
  — the chain is SALU latency fully hidden behind MFMA. The real causal lever was block
  load-balance (Section 5), not the mask arithmetic.
- **`sched_barrier` tuning**: matching CK barrier-for-barrier is a codegen-alignment goal; on
  its own it gave ~0% runtime change.
- **Mask compute in general** is not the causal bottleneck; the masking path is shared with
  mask=0, which is already at parity.
