# Phase 2: Fused Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking. Launch with
> `/hip-kernel-team load fmha-native-isa-match`.

**Goal:** Assemble the 7 Phase 1 golden-verified inner loops into a
single fused kernel that passes 50/50 correctness + 48/48 GPU ref tests,
matching CK's ISA profile.

**Architecture:** Gut the existing `src/kernel/` files (5 HPPs) and
replace their internals with Phase 1 logic, wired into a 3-buffer async
pipeline with sub-tile interleaving. Keep the existing entry point,
template params `<HasMask, IsVarlen>`, and test infrastructure.

**Tech Stack:** HIP, gfx942 inline asm, MFMA intrinsics, async
buffer-to-LDS copies, GTest.

**Spec:** `docs/superpowers/specs/2026-05-30-phase2-fused-kernel-design.md`

---

## File Structure

```
src/kernel/
  fmha_fwd_d64_lds.hpp       — K async copy, V DRAM load, V shuffle+LDS store
  fmha_fwd_d64_gemm.hpp      — GEMM0/GEMM1 sub-tile functions
  fmha_fwd_d64_softmax.hpp   — scale, mask, rmax, exp2, rsum, bf16 cast
  fmha_fwd_d64_epilog.hpp    — normalize O, LSE, bf16 store
  fmha_fwd_d64_device.hpp    — Q load, pipeline loop, online softmax rescale
  fmha_fwd_d64_kernel.cpp    — __global__ wrappers (minor edit only)
CMakeLists.txt                — compile flags update
```

All functions in the inner HPPs are `__device__ __forceinline__`. The
`_device.hpp` orchestrates calls. `_kernel.cpp` instantiates 4 template
variants and is mostly unchanged (add `__launch_bounds__` if missing).

---

## Task 1: Update CMake compile flags

**Files:**
- Modify: `CMakeLists.txt:37-41`

- [ ] **Step 1: Read the current fmha_kernel compile options**

```bash
grep -A5 "target_compile_options(fmha_kernel" CMakeLists.txt
```

- [ ] **Step 2: Replace compile options with full CK flag set**

Replace the existing `target_compile_options(fmha_kernel PRIVATE ...)`
block with:

```cmake
target_compile_options(fmha_kernel PRIVATE
    --offload-arch=${GPU_TARGET} --save-temps
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1
    -mllvm -amdgpu-early-inline-all=true
    -mllvm -amdgpu-function-calls=false
    -mllvm --lsr-drop-solution=1
    -mllvm -enable-post-misched=0
    -fbracket-depth=1024
    -fno-offload-uniform-block
    -fgpu-flush-denormals-to-zero)
```

- [ ] **Step 3: Verify build still works**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja 2>&1 && ninja fmha_kernel 2>&1"
```

Expected: builds clean. The `--save-temps` flag produces a `.s` file.

- [ ] **Step 4: Verify Phase 1 tests still pass (no regression)**

```bash
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_k_lds --golden-full=/tmp/fmha-native-isa-match/golden/full --golden-partial=/tmp/fmha-native-isa-match/golden/partial 2>&1"
```

Repeat for all 7 Phase 1 binaries. Expected: 49/49 PASS, all EXIT 0.

- [ ] **Step 5: Verify .s file is produced**

```bash
docker exec poyenc-fmha bash -c "find /root/workspace/build -name '*fmha_fwd_d64*gfx942*.s' | head -5"
```

Expected: at least one `.s` file found.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt
git commit -m "fmha_native: add CK-matched compile flags to fmha_kernel target"
```

---

## Task 2: Rewrite `_lds.hpp` — K async copy + V shuffle+store

**Files:**
- Modify: `src/kernel/fmha_fwd_d64_lds.hpp`
- Reference: `src/kernels/k_lds.hpp` (Phase 1 K1), `src/kernels/v_lds.hpp` (Phase 1 K5)

- [ ] **Step 1: Backup the existing file**

```bash
cp src/kernel/fmha_fwd_d64_lds.hpp src/kernel/fmha_fwd_d64_lds.hpp.bak
```

- [ ] **Step 2: Write the new `_lds.hpp`**

Replace the entire file contents. The new file provides three functions:

```cpp
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Constants
constexpr int kSingleSmemElements = 2304; // elements per LDS buffer
constexpr int kKLdsPadStride = 136;       // 128 data + 8 pad per lane-group row

// ---- K async DRAM→LDS for one sub-tile (kK0=32 headdim slice) ----
//
// Loads one headdim chunk (32 bf16) of K for all 64 seqlen_k rows
// into the specified LDS buffer using async buffer_load_dword...lds.
//
// K LDS layout (golden-verified Phase 0.6):
//   offset(j,d) = buf_base + (j%4)*136 + ((j/4)%4)*32 + (j/16)*544
//                 + (d%32) + (d/32)*2304
//   where j=seqlen_k index, d=headdim index within this chunk.
//
// n_base decomposition: n = issue*16 + (lane>>4)*4 + warp
//   (NOT warp*4 + lane>>4 — the existing code had this WRONG).
//
// Each call handles one kK0=32 headdim chunk. The caller selects
// which chunk via k_col_offset (0 or 32).
//
// Parameters:
//   lds_base    — char* to start of LDS allocation
//   buf_idx     — which of the 3 LDS buffers (0,1,2)
//   k_srd       — buffer resource descriptor for K DRAM
//   stride_k    — K row stride in bf16 elements
//   kv_offset   — base seqlen_k offset for this tile
//   k_col_offset— headdim offset (0 or kK0=32)
//   seqlen_k    — total seqlen_k for OOB gating
__device__ __forceinline__ void async_copy_k_subtile(
    char* lds_base, int buf_idx,
    __amdgpu_buffer_rsrc_t k_srd, int stride_k,
    int kv_offset, int k_col_offset, int seqlen_k);

// ---- V DRAM load into register buffer (one sub-tile) ----
//
// Loads one kK1=32 chunk of V from DRAM into register buffer
// using buffer_load_dwordx2 (4 bf16 per load, 2 loads per thread).
//
// Parameters:
//   v_reg       — output: 2 x dwordx2 register pairs
//   v_srd       — buffer resource descriptor for V DRAM
//   stride_v    — V row stride in bf16 elements
//   kv_offset   — base seqlen_k offset for this sub-tile
//   seqlen_k    — total seqlen_k for OOB gating
__device__ __forceinline__ void load_v_from_dram(
    uint32_t v_reg[4],
    __amdgpu_buffer_rsrc_t v_srd, int stride_v,
    int kv_offset, int seqlen_k);

// ---- V shuffle + store to LDS ----
//
// Takes V register buffer (from load_v_from_dram), applies
// intra-thread v_perm_b32 transpose (4×2→2×4, selectors
// 0x01000504/0x03020706), then stores via ds_write2_b32.
//
// V LDS layout (golden-verified Phase 0.10):
//   k = n % 32
//   offset(n,d) = buf_base + (k/8)*576 + (d/8)*72 + (d%8)*8 + (k%8)
//   headdim = row axis (72 stride = 64+8 pad)
//
// Parameters:
//   v_reg       — input: shuffled register pairs from v_perm
//   lds_base    — char* to start of LDS allocation
//   buf_idx     — which LDS buffer (0,1,2)
__device__ __forceinline__ void store_v_to_lds(
    const uint32_t v_reg[4],
    char* lds_base, int buf_idx);

// ---- Buffer base helper ----
__device__ __forceinline__ int buf_base_bytes(int buf_idx) {
    return buf_idx * kSingleSmemElements * 2; // 2 bytes per bf16 element
}
```

The function BODIES must be adapted from Phase 1 K1
(`src/kernels/k_lds.hpp`) and K5 (`src/kernels/v_lds.hpp`). Key
differences from standalone:
- Take `buf_idx` parameter instead of hardcoded buffer 0
- Take `lds_base` char* instead of using LDS from `__shared__`
- Functions are `__device__ __forceinline__`, not `__global__`
- SRD and stride passed as parameters, not computed from kargs

Copy the m0 computation from K1 (lines 60-90), adjusting for buf_idx:
```cpp
int m0_base = buf_base_bytes(buf_idx) + warp * 0x110;
// ... same async copy loop as K1, using m0_base
```

Copy the v_perm + ds_write from K5 (the shuffle + store logic),
adjusting for buf_idx in the LDS offset.

- [ ] **Step 3: Build to verify syntax**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja 2>&1 && ninja fmha_kernel 2>&1"
```

Expected: builds clean (functions are `__forceinline__` but not yet
called — no dead code warnings expected since they're in a header).

- [ ] **Step 4: Verify Phase 1 tests still pass**

Run all 7 Phase 1 binaries with golden. Expected: 49/49, EXIT 0.

- [ ] **Step 5: Verify old functions are gone**

```bash
grep -n "copy_k_to_lds_2x_guarded\|gemm1_bpermute\|v_load_pair" src/kernel/fmha_fwd_d64_lds.hpp
```

Expected: no matches (all old functions replaced).

- [ ] **Step 6: Commit**

```bash
git add src/kernel/fmha_fwd_d64_lds.hpp
git commit -m "fmha_native: rewrite _lds.hpp with K async copy + V shuffle+store"
```

---

## Task 3: Rewrite `_gemm.hpp` — GEMM0 + GEMM1 sub-tile functions

**Files:**
- Modify: `src/kernel/fmha_fwd_d64_gemm.hpp`
- Reference: `src/kernels/qk_gemm.hpp` (Phase 1 K2), `src/kernels/pv_gemm.hpp` (Phase 1 K6)

- [ ] **Step 1: Backup the existing file**

```bash
cp src/kernel/fmha_fwd_d64_gemm.hpp src/kernel/fmha_fwd_d64_gemm.hpp.bak
```

- [ ] **Step 2: Write the new `_gemm.hpp`**

Replace the entire file contents. Provides sub-tile GEMM functions:

```cpp
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef float v16f __attribute__((ext_vector_type(16)));
typedef int v4i __attribute__((ext_vector_type(4)));

// ---- SwizzleA helper (GEMM0 only) ----
// Swaps bits 2,3 of the K seqlen lane index.
// This is why S_acc has groups-of-8 (not groups-of-4).
__device__ __forceinline__ int qk_swz(int x) {
    return (x & ~0xC) | (((x >> 2) & 1) << 3) | (((x >> 3) & 1) << 2);
}

// ---- GEMM0 sub-tile: one kK0=32 headdim chunk ----
//
// Reads K from LDS buffer `buf_idx` at headdim chunk `k_subtile_idx`,
// multiplies with Q registers, accumulates into S_acc.
//
// MFMA convention: A=K(LDS), B=Q(register), C=S_acc(accumulator).
// SwizzleA applied to K LDS read addresses.
//
// Executes 2 N-tiles × 2 HW passes = 8 v_mfma_f32_32x32x8_bf16.
//
// Parameters:
//   s_n0, s_n1  — accumulator halves (16 fp32 each, total 32 regs)
//   q_regs      — Q register buffer (pre-sliced for this sub-tile)
//   lds_base    — LDS base pointer
//   buf_idx     — which LDS buffer (0,1,2)
__device__ __forceinline__ void gemm0_subtile(
    v16f& s_n0, v16f& s_n1,
    const v4i q_b[4],
    char* lds_base, int buf_idx);

// ---- GEMM1 sub-tile: one kK1=32 seqlen_k chunk ----
//
// Reads V from LDS buffer `buf_idx`, multiplies with P registers,
// accumulates into O_acc.
//
// MFMA convention: A=V(LDS), B=P(register), C=O_acc(accumulator).
// NO SwizzleA on V reads (unlike GEMM0).
//
// Executes 2 N-tiles × 2 HW passes = 8 v_mfma_f32_32x32x8_bf16.
//
// Parameters:
//   o_d0, o_d1  — accumulator halves (16 fp32 each, total 32 regs)
//   p_packed    — P bf16 packed into dwords (4 dwords for this sub-tile)
//   lds_base    — LDS base pointer
//   buf_idx     — which LDS buffer (0,1,2)
__device__ __forceinline__ void gemm1_subtile(
    v16f& o_d0, v16f& o_d1,
    const int p_packed[4],
    char* lds_base, int buf_idx);
```

The function BODIES are adapted from Phase 1 K2 (`qk_gemm.hpp` inner
loop, lines ~130-160) and K6 (`pv_gemm.hpp` inner loop). Key changes:
- Take `buf_idx` and `lds_base` instead of hardcoded offsets
- Sub-tile: each call handles one kK0/kK1 chunk (8 MFMA), not the
  full 16
- Q is pre-sliced by the caller (4 × v4i for one headdim chunk)
- P is pre-packed by the caller (4 × int for one seqk chunk)

K LDS read address (GEMM0, with SwizzleA):
```cpp
int k_sub = lane_id >> 5;
int seqk_in_tile = qk_swz(lane_id & 31);
int lds_byte = buf_base_bytes(buf_idx)
    + /* same offset formula as K1, using seqk_in_tile and k_sub */;
```

V LDS read address (GEMM1, NO SwizzleA):
```cpp
int d_pos = lane_id & 31;  // bare, no swizzle
int lds_byte = buf_base_bytes(buf_idx)
    + /* V LDS formula from K5/K6 */;
```

- [ ] **Step 3: Build to verify syntax**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja && ninja fmha_kernel 2>&1"
```

Expected: builds clean.

- [ ] **Step 4: Verify old functions are gone**

```bash
grep -n "gemm0\b\|gemm1_bpermute\|gemm0_interleaved" src/kernel/fmha_fwd_d64_gemm.hpp
```

Expected: no matches for old function names. Only `gemm0_subtile`,
`gemm1_subtile`, `qk_swz`.

- [ ] **Step 5: Commit**

```bash
git add src/kernel/fmha_fwd_d64_gemm.hpp
git commit -m "fmha_native: rewrite _gemm.hpp with sub-tile GEMM0/GEMM1 functions"
```

---

## Task 4: Rewrite `_softmax.hpp` — correct softmax functions

**Files:**
- Modify: `src/kernel/fmha_fwd_d64_softmax.hpp`
- Reference: `src/kernels/row_max.hpp` (K3), `src/kernels/softmax.hpp` (K4)

- [ ] **Step 1: Backup the existing file**

```bash
cp src/kernel/fmha_fwd_d64_softmax.hpp src/kernel/fmha_fwd_d64_softmax.hpp.bak
```

- [ ] **Step 2: Write the new `_softmax.hpp`**

Replace the entire file. Provides 5 functions:

```cpp
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef float v16f __attribute__((ext_vector_type(16)));

// ds_bpermute helper
__device__ __forceinline__ float sm_bpermute(int src_lane, float val) {
    int ret;
    asm volatile("ds_bpermute_b32 %0, %1, %2\n"
                 "s_waitcnt lgkmcnt(0)"
                 : "=v"(ret) : "v"(src_lane * 4), "v"(val));
    return __builtin_bit_cast(float, ret);
}

// TransposedC n_col formula (inline, used by scale_and_mask)
__device__ __forceinline__ int n_col_from_reg(int r, int k_sub) {
    return (r / 8) * 16 + k_sub * 8 + (r % 8);
}

// ---- Scale S_acc and apply masks ----
// Scales by scale_s_log2e, applies boundary mask (n >= seqlen_k)
// and causal mask (n > m_row + shift) by setting to -inf.
__device__ __forceinline__ void softmax_scale_and_mask(
    v16f& s_n0, v16f& s_n1,
    float scale_s_log2e,
    int seqlen_k, int kv_tile_offset,
    int m_row, bool has_mask, int mask_shift);

// ---- Row max: intra-lane + 1 cross-half bpermute ----
// Returns the max over all 64 N-columns for this thread's M-row.
__device__ __forceinline__ float softmax_row_max(
    const v16f& s_n0, const v16f& s_n1);

// ---- Exp2: P_fp32 = exp2(S_scaled - row_max) ----
__device__ __forceinline__ void softmax_exp2(
    v16f& s_n0, v16f& s_n1, float row_max);

// ---- Row sum: intra-lane sum + 1 cross-half bpermute ----
__device__ __forceinline__ float softmax_row_sum(
    const v16f& s_n0, const v16f& s_n1);

// ---- Cast P fp32 → bf16 (truncation, NOT round-to-nearest) ----
// Packs 32 fp32 values into 16 dwords (2 bf16 per dword).
__device__ __forceinline__ void softmax_p_to_bf16(
    const v16f& p_n0, const v16f& p_n1,
    int p_packed[16]);
```

The function BODIES are adapted from Phase 1 K3 (`row_max.hpp`) and K4
(`softmax.hpp`). Key adaptation:
- Operate on `v16f& s_n0, v16f& s_n1` (the existing accumulator type)
  instead of `float[32]`
- `s_n0[i]` for i=0..15 = regs for N-tile 0, `s_n1[i]` for N-tile 1
- Row max: `fmaxf` over all 32 elements (16 from s_n0 + 16 from s_n1),
  then 1 `sm_bpermute(lane ^ 32, local_max)`
- Row sum: same pattern with `+` instead of `fmaxf`
- Causal mask: for each reg r in 0..15 of each tile, compute
  `n = kv_tile_offset + n_col_from_reg(r, k_sub)`, compare with
  `m_row + mask_shift`

- [ ] **Step 3: Build to verify syntax**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja && ninja fmha_kernel 2>&1"
```

- [ ] **Step 4: Verify butterfly functions are GONE**

```bash
grep -n "butterfly\|offset.*=.*16\|offset.*=.*8.*4.*2.*1" src/kernel/fmha_fwd_d64_softmax.hpp
```

Expected: no matches. The 5-round butterfly pattern is eliminated.

- [ ] **Step 5: Verify ds_bpermute pattern is correct**

```bash
grep -n "bpermute" src/kernel/fmha_fwd_d64_softmax.hpp
```

Expected: `sm_bpermute` helper + calls in `softmax_row_max` and
`softmax_row_sum` only. No 5-round loop.

- [ ] **Step 6: Commit**

```bash
git add src/kernel/fmha_fwd_d64_softmax.hpp
git commit -m "fmha_native: rewrite _softmax.hpp — fix butterfly bug, use intra-lane + bpermute"
```

---

## Task 5: Rewrite `_epilog.hpp` — normalize + LSE + bf16 store

**Files:**
- Modify: `src/kernel/fmha_fwd_d64_epilog.hpp`
- Reference: `src/kernels/epilog.hpp` (Phase 1 K7)

- [ ] **Step 1: Backup the existing file**

```bash
cp src/kernel/fmha_fwd_d64_epilog.hpp src/kernel/fmha_fwd_d64_epilog.hpp.bak
```

- [ ] **Step 2: Write the new `_epilog.hpp`**

Replace the entire file. One function:

```cpp
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef float v16f __attribute__((ext_vector_type(16)));

// SwizzleA headdim correction (inherited from P → O_acc)
__device__ __forceinline__ int ep_swz(int d) {
    return (d & ~0xC) | (((d >> 2) & 1) << 3) | (((d >> 3) & 1) << 2);
}

// ---- Epilog: normalize O, compute LSE, cast bf16, store to DRAM ----
//
// O_final = O_acc / rsum (per element, guard rsum==0 → 0)
// LSE = log(rsum) + rmax * (1/log2e)  (stored if lse_base != nullptr)
// bf16 truncation (not RNE) + buffer_store_dwordx2 (8 stores/thread)
//
// Store address uses SwizzleA'd headdim:
//   m_row = (lane%32) + 32*warp
//   d_col = ep_swz((r/8)*16 + k_sub*8 + (r%8))
//
// Parameters:
//   o_d0, o_d1  — O_acc accumulator halves (modified in place)
//   rsum        — row sum scalar (from online softmax)
//   rmax        — row max scalar (for LSE computation)
//   o_srd       — buffer resource descriptor for O DRAM
//   stride_o    — O row stride in bf16 elements
//   lse_base    — LSE output pointer (nullptr to skip)
//   seqlen_q    — for OOB gating
//   m_tile_idx  — Q tile index
//   scale_s     — original scale (before log2e multiplication)
__device__ __forceinline__ void epilog_store(
    v16f& o_d0, v16f& o_d1,
    float rsum, float rmax,
    __amdgpu_buffer_rsrc_t o_srd, int stride_o,
    float* lse_base,
    int seqlen_q, int m_tile_idx, float scale_s);
```

The function BODY is adapted from Phase 1 K7 (`epilog.hpp`). Key
additions vs standalone:
- LSE computation: `lse = __logf(rsum) + rmax * (1.0f / (scale_s * log2e))`
  Store via scalar write to `lse_base[m_row]` (only k_sub==0 lane writes)
- Guard: `if (rsum == 0.0f) inv_rs = 0.0f` (prevents inf on padding rows)
- Takes SRD instead of raw pointer (for buffer_store_dwordx2)

- [ ] **Step 3: Build to verify syntax**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja && ninja fmha_kernel 2>&1"
```

- [ ] **Step 4: Verify old functions are gone**

```bash
grep -n "epilog_store_o\b\|epilog_store_o_buffer" src/kernel/fmha_fwd_d64_epilog.hpp
```

Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add src/kernel/fmha_fwd_d64_epilog.hpp
git commit -m "fmha_native: rewrite _epilog.hpp — SwizzleA store + LSE + buffer_store"
```

---

## Task 6: Rewrite `_device.hpp` — main pipeline loop

**Files:**
- Modify: `src/kernel/fmha_fwd_d64_device.hpp`
- Modify: `src/kernel/fmha_fwd_d64_kernel.cpp` (add `__launch_bounds__` if missing)

This is the largest and most critical task. It wires all inner functions
into the 3-buffer async pipeline.

- [ ] **Step 1: Backup the existing file**

```bash
cp src/kernel/fmha_fwd_d64_device.hpp src/kernel/fmha_fwd_d64_device.hpp.bak
```

- [ ] **Step 2: Write the pipeline constants and helpers at the top**

Keep the existing `#pragma once`, includes, and `FmhaFwdParams` include.
Add after includes:

```cpp
#include "fmha_fwd_d64_lds.hpp"
#include "fmha_fwd_d64_gemm.hpp"
#include "fmha_fwd_d64_softmax.hpp"
#include "fmha_fwd_d64_epilog.hpp"

// Pipeline constants
constexpr int kM0 = 128;
constexpr int kN0 = 64;
constexpr int kK0 = 32;
constexpr int kK1 = 32;
constexpr int kBlockSize = 256;
constexpr int kNumWarps = 4;
constexpr int kWarpSize = 64;
constexpr int kHeadDim = 64;
constexpr int kLdsBytes = 13824;

// Sub-tile loop counts
constexpr int k0_loops = kHeadDim / kK0;  // 2
constexpr int k1_loops = kN0 / kK1;       // 2

// LDS buffer sequence: {K sub 0, K sub 1, V sub 0, V sub 1}
constexpr int LdsSeq[4] = {1, 2, 1, 0};
```

- [ ] **Step 3: Write the `fmha_fwd_d64_device` function — setup section**

Keep the existing template signature and varlen setup. Replace the
body starting after varlen pointer resolution:

```cpp
template <bool HasMask, bool IsVarlen>
__device__ __forceinline__ void fmha_fwd_d64_device(
    const FmhaFwdParams& params, char* lds,
    int batch_idx, int head_idx, int m_tile_idx)
{
    // ---- Thread identity ----
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane_id = tid % kWarpSize;
    const int k_sub = lane_id >> 5;
    const int m_row = (lane_id & 31) + 32 * warp_id;

    // ---- Varlen setup (keep existing) ----
    int seqlen_q = params.seqlen_q;
    int seqlen_k = params.seqlen_k;
    // ... varlen pointer offset code ...

    // ---- Early exit: m_tile out of range ----
    if (m_tile_idx * kM0 >= seqlen_q) return;

    // ---- Build SRDs for Q, K, V, O ----
    // ... (construct __amdgpu_buffer_rsrc_t for each tensor) ...

    // ---- Q load (once, 32 bf16 regs = 16 packed dwords) ----
    v4i q_regs[k0_loops * 2]; // 2 sub-tiles × 2 passes = 4 v4i
    // ... load Q using buffer_load_b128, same as Phase 1 K2 ...
    // Zero-fill for m_row >= seqlen_q

    // ---- Initialize online softmax state ----
    float m_old = -INFINITY;  // running row max
    float l_old = 0.0f;       // running row sum
    v16f o_d0 = {0}; // O_acc n-tile 0 (hdim 0..31)
    v16f o_d1 = {0}; // O_acc n-tile 1 (hdim 32..63)
```

- [ ] **Step 4: Write the tile loop bounds (causal skip)**

```cpp
    // ---- Tile range computation (causal skip) ----
    int kv_start = 0;
    int kv_end = seqlen_k;
    if constexpr (HasMask) {
        // Causal: this Q tile at m_tile_idx only needs K tiles
        // where n <= m_row_max + shift
        int m_max = m_tile_idx * kM0 + kM0 - 1;
        int shift = seqlen_k - seqlen_q;
        kv_end = min(kv_end,
            ((min(m_max + shift + 1, seqlen_k) + kN0 - 1) / kN0) * kN0);
    }
    int num_total_loop = (kv_end - kv_start + kN0 - 1) / kN0;

    // ---- Early exit: no tiles to process ----
    if (num_total_loop <= 0) {
        // Zero O, store, return
        epilog_store(o_d0, o_d1, 0.0f, -INFINITY, o_srd,
                     params.stride_o, params.lse, seqlen_q,
                     m_tile_idx, params.scale);
        return;
    }
```

- [ ] **Step 5: Write the prologue (first K prefetch)**

```cpp
    // ---- Prologue: prefetch first K sub-tile ----
    int kv_offset = kv_start;
    async_copy_k_subtile(lds, LdsSeq[0], k_srd, params.stride_k,
                         kv_offset, 0, seqlen_k);
    // Advance K DRAM window to next headdim chunk
    // (k_col_offset will be kK0=32 for the next sub-tile)

    int i_total_loops = 0;
```

- [ ] **Step 6: Write the main do...while loop body — Stage 1 (GEMM0)**

```cpp
    do {
        // ---- Stage 1: GEMM0 (QK) ----
        v16f s_n0 = {0}, s_n1 = {0};  // clear S_acc

        // k0 inner loop: i_k0 = 0 .. k0_loops-2 (just i_k0=0)
        // Prefetch next K sub-tile while computing on current
        async_copy_k_subtile(lds, LdsSeq[1], k_srd, params.stride_k,
                             kv_offset, kK0, seqlen_k);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __builtin_amdgcn_s_barrier();
        gemm0_subtile(s_n0, s_n1, &q_regs[0], lds, LdsSeq[0]);

        // GEMM0 tail: last K sub-tile
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __builtin_amdgcn_s_barrier();
        // Overlap: load V[0] from DRAM during tail compute
        uint32_t v_reg[4];
        load_v_from_dram(v_reg, v_srd, params.stride_v,
                         kv_offset, seqlen_k);
        gemm0_subtile(s_n0, s_n1, &q_regs[2], lds, LdsSeq[1]);
```

- [ ] **Step 7: Write Stage 2 (Softmax)**

```cpp
        // ---- Stage 2: Softmax ----
        float scale_s_log2e = params.scale; // already pre-multiplied
        int mask_shift = HasMask ? (seqlen_k - seqlen_q) : 0;
        softmax_scale_and_mask(s_n0, s_n1, scale_s_log2e,
                               seqlen_k, kv_offset,
                               m_tile_idx * kM0 + m_row,
                               HasMask, mask_shift);

        float m_new = softmax_row_max(s_n0, s_n1);
        m_new = fmaxf(m_old, m_new);

        // V shuffle + store to LDS (overlaps with softmax compute)
        // Barrier if K tail and V share buffer
        if constexpr (LdsSeq[k0_loops - 1] == LdsSeq[k0_loops]) {
            __builtin_amdgcn_s_barrier();
        }
        store_v_to_lds(v_reg, lds, LdsSeq[k0_loops]);

        // Prefetch next V sub-tile from DRAM (for GEMM1 k1=1)
        uint32_t v_reg_next[4];
        if constexpr (k1_loops > 1) {
            load_v_from_dram(v_reg_next, v_srd, params.stride_v,
                             kv_offset + kK1, seqlen_k);
        }

        // Exp2
        softmax_exp2(s_n0, s_n1, m_new);

        // Row sum
        float l_new = softmax_row_sum(s_n0, s_n1);

        // Rescale O_acc and running sum
        float rescale = __builtin_amdgcn_exp2f(m_old - m_new);
        #pragma unroll
        for (int i = 0; i < 16; i++) { o_d0[i] *= rescale; }
        #pragma unroll
        for (int i = 0; i < 16; i++) { o_d1[i] *= rescale; }
        l_old = rescale * l_old + l_new;
        m_old = m_new;

        // Pack P to bf16
        int p_packed[16];
        softmax_p_to_bf16(s_n0, s_n1, p_packed);
```

- [ ] **Step 8: Write Stage 3 (GEMM1 + next-tile K prefetch)**

```cpp
        // ---- Stage 3: GEMM1 (PV) ----
        // k1 inner loop: i_k1 = 0 (just one inner iteration)
        __builtin_amdgcn_s_barrier();
        gemm1_subtile(o_d0, o_d1, &p_packed[0], lds, LdsSeq[k0_loops]);

        // Store next V sub-tile to LDS
        store_v_to_lds(v_reg_next, lds, LdsSeq[k0_loops + 1]);

        // Advance iteration counter
        i_total_loops++;
        kv_offset += kN0;

        // Prefetch next tile's first K sub-tile
        if (i_total_loops < num_total_loop) {
            // Check if K prefetch buffer conflicts with V tail read
            if constexpr (LdsSeq[0] == LdsSeq[k0_loops + k1_loops - 2]) {
                __builtin_amdgcn_s_barrier();
            }
            async_copy_k_subtile(lds, LdsSeq[0], k_srd,
                                 params.stride_k, kv_offset, 0,
                                 seqlen_k);
        }

        // GEMM1 tail
        __builtin_amdgcn_s_barrier();
        gemm1_subtile(o_d0, o_d1, &p_packed[4], lds,
                       LdsSeq[k0_loops + k1_loops - 1]);

    } while (i_total_loops < num_total_loop);
```

- [ ] **Step 9: Write the epilogue (after loop)**

```cpp
    // ---- Epilogue: normalize + LSE + store ----
    epilog_store(o_d0, o_d1, l_old, m_old, o_srd,
                 params.stride_o, params.lse, seqlen_q,
                 m_tile_idx, params.scale);
}
```

- [ ] **Step 10: Update `_kernel.cpp` with `__launch_bounds__`**

Check if `__launch_bounds__(256, 3)` is present on each `__global__`
kernel. If not (currently only `__launch_bounds__(kBlockSize)`), update:

```cpp
__global__ __launch_bounds__(kBlockSize, 3)
void fmha_fwd_d64_bf16_msk0(FmhaFwdParams params) { ... }
```

Apply to all 4 kernel wrappers.

- [ ] **Step 11: Build the fused kernel**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja && ninja fmha_kernel 2>&1" > /tmp/fmha-native-isa-match/impl/build_fused_001.txt 2>&1
```

Expected: builds clean with the full CK flag set.

- [ ] **Step 12: Verify Phase 1 tests still pass**

Run all 7 Phase 1 binaries. Expected: 49/49, EXIT 0 (no regression).

- [ ] **Step 13: Run initial fused kernel tests**

```bash
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_fwd_d64 2>&1" > /tmp/fmha-native-isa-match/impl/fused_test_001.txt 2>&1
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_gpu_ref 2>&1" > /tmp/fmha-native-isa-match/impl/gpu_ref_test_001.txt 2>&1
```

Report pass/fail counts. Not expected to pass all yet — this is the
initial wiring.

- [ ] **Step 14: Commit**

```bash
git add src/kernel/fmha_fwd_d64_device.hpp src/kernel/fmha_fwd_d64_kernel.cpp
git commit -m "fmha_native: rewrite _device.hpp with 3-buffer async pipeline"
```

---

## Task 7: Debug and fix until all tests pass

**Files:**
- Modify: all `src/kernel/` files as needed

This is an iterative debug loop. No fixed steps — work until the
acceptance criteria are met.

- [ ] **Step 1: Run full test suite and capture output**

```bash
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_fwd_d64 2>&1" > /tmp/fmha-native-isa-match/impl/fused_test_NNN.txt 2>&1
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_gpu_ref 2>&1" > /tmp/fmha-native-isa-match/impl/gpu_ref_test_NNN.txt 2>&1
```

- [ ] **Step 2: For each failure, identify root cause**

Use Phase 1 golden data as intermediate checkpoints. For example, if
the output is wrong, load golden S_acc, P, O_acc to check which stage
diverges.

- [ ] **Step 3: Fix the issue and re-run**

After each fix, rebuild and re-run. Document what was wrong and what
was changed.

- [ ] **Step 4: Acceptance gate**

All must pass in a single run, no filters:
- `test_fmha_fwd_d64`: 50/50 PASS
- `test_fmha_gpu_ref`: 48/48 PASS
- Phase 1 standalone: 49/49 PASS (no regression)

- [ ] **Step 5: Commit**

```bash
git add src/kernel/
git commit -m "fmha_native: Phase 2 correctness — 50/50 + 48/48 tests pass"
```

---

## Task 8: Golden end-to-end test — bit-match vs CK O output

**Files:**
- Create: `tests/test_fmha_golden_e2e.cpp` (or add to existing)
- Modify: `CMakeLists.txt` (add target)

- [ ] **Step 1: Write the golden end-to-end test**

A new test binary that:
1. Runs the fused kernel with the same deterministic inputs as the
   golden dump (q[i]=(i%256)/256, k[i]=((i+64)%256)/256,
   v[i]=(i%256)/256+1, scale_s=0.125)
2. Compares the bf16 O output against `o_dram.bin` from the golden dir
3. Tests both full (sq=64, sk=64) and partial (sq=17, sk=33)
4. Uses `--golden-full=` and `--golden-partial=` flags

- [ ] **Step 2: Add CMake target**

```cmake
add_executable(test_fmha_golden_e2e tests/test_fmha_golden_e2e.cpp)
target_link_libraries(test_fmha_golden_e2e PRIVATE fmha_kernel fmha_runner GTest::gtest)
set_source_files_properties(tests/test_fmha_golden_e2e.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET}")
```

- [ ] **Step 3: Build and run**

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja && ninja test_fmha_golden_e2e 2>&1"
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_golden_e2e --golden-full=/tmp/fmha-native-isa-match/golden/full --golden-partial=/tmp/fmha-native-isa-match/golden/partial 2>&1"
```

Expected: 0 mismatches vs CK golden O for both tile sizes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_fmha_golden_e2e.cpp CMakeLists.txt
git commit -m "fmha_native: add golden end-to-end test vs CK O output"
```

---

## Task 9: ISA gate — structural analysis vs CK assembly

**Files:**
- Read: build output `.s` file (native)
- Reference: `~/workspace/repo/rocm-libraries/projects/composablekernel/ck_d64_kernel.s` (CK)

Instruction counting alone tells nothing — two kernels can have
identical instruction counts but completely different schedules,
dependency chains, and performance. This task does **structural**
comparison.

- [ ] **Step 1: Extract native assembly**

```bash
docker exec poyenc-fmha bash -c "find /root/workspace/build -name '*fmha_fwd_d64_kernel*gfx942*.s'"
docker cp poyenc-fmha:<path-to-s-file> /home/poyenc/workspace/repo/fmha_native/native_d64_kernel.s
```

- [ ] **Step 2: Run `/amdgpu-asm-analyzer` on BOTH assemblies**

Analyze BOTH the native kernel and the CK reference assembly using the
`/amdgpu-asm-analyzer` skill. For each, extract:

1. **Control flow graph (CFG):** basic block structure, loop headers,
   loop back-edges, branch targets. The main loop body should be one
   or two basic blocks. If the native kernel has a different number of
   BBs or different branch structure, that's a structural divergence.

2. **Dependency chains within the main loop:** for each MFMA, what are
   its producer instructions (ds_read for A/B operands) and consumer
   instructions (next MFMA that reuses the accumulator, or the softmax
   that reads S_acc)? Map the chain:
   ```
   async_load → s_waitcnt → s_barrier → ds_read_b128 → v_mfma
   ds_read_b128 → v_mfma → v_mfma (accumulator chain)
   v_mfma (last GEMM0) → softmax (reads S_acc regs)
   softmax → p_pack → v_mfma (first GEMM1)
   v_mfma (last GEMM1) → epilog (reads O_acc regs)
   ```

3. **Waitcnt placement:** for each `s_waitcnt` and `s_barrier`, what
   instruction does it guard? Is it placed at the same point relative
   to its consumer as in CK? A waitcnt too early = unnecessary stall.
   A waitcnt too late = data hazard.

4. **Scheduling order within basic blocks:** are MFMA instructions
   interleaved with ds_read/ds_write the same way? CK interleaves
   V ds_write with softmax compute, and K async_load with GEMM1 —
   does the native kernel do the same?

- [ ] **Step 3: Compare CFG structure**

| Property | CK | Native | Match? |
|----------|-----|--------|--------|
| Basic blocks in main loop | ? | ? | |
| Loop back-edge target | ? | ? | |
| Branch structure (conditional tiles) | ? | ? | |
| Prologue BB (before loop) | ? | ? | |
| Epilogue BB (after loop) | ? | ? | |

- [ ] **Step 4: Compare dependency tables**

For each pipeline stage (GEMM0 sub0, GEMM0 sub1, softmax, V-store,
GEMM1 sub0, K-prefetch, GEMM1 sub1), compare the dependency chain:

| Stage | CK producer→consumer | Native producer→consumer | Match? |
|-------|---------------------|-------------------------|--------|
| GEMM0 sub0: K read → MFMA | ? | ? | |
| GEMM0 sub1: K read → MFMA | ? | ? | |
| GEMM0→softmax: last MFMA → scale | ? | ? | |
| softmax→V store: V load fence → ds_write | ? | ? | |
| V store→GEMM1: ds_write → barrier → ds_read | ? | ? | |
| GEMM1→K prefetch: async_load timing | ? | ? | |
| GEMM1 tail→epilog: last MFMA → store | ? | ? | |

- [ ] **Step 5: Compare waitcnt/barrier placement**

For each `s_waitcnt` in the native assembly, verify it matches CK's
placement relative to its consumer. Document any differences:
- `vmcnt(N)` value (how many outstanding loads allowed)
- Distance (in instructions) between waitcnt and consumer
- Whether the native kernel has extra/missing waitcnts vs CK

- [ ] **Step 6: Resource usage (secondary check)**

| Resource | CK target | Native | Match? |
|----------|----------|--------|--------|
| VGPR | ≤ 127 | ? | |
| AGPR | 0 | ? | |
| Spill | 0 | ? | |
| LDS | 13824 | ? | |

This is a sanity check, not the primary gate. A kernel can have
matching VGPR count but completely wrong scheduling.

- [ ] **Step 7: Document results — per-region report**

Write to `/tmp/fmha-native-isa-match/prof/phase2_isa_gate_001.md`.
The report MUST be structured per-region so that an independent
verifier can check each claim against the actual assembly. Regions:

```
Region 1: Prologue (kernel entry → first async_load)
Region 2: GEMM0 sub-tile 0 (barrier → 8 MFMA)
Region 3: GEMM0 sub-tile 1 (barrier → 8 MFMA + V DRAM load overlap)
Region 4: Softmax (scale → mask → rmax → exp2 → rsum → rescale)
Region 5: V shuffle + LDS store
Region 6: GEMM1 sub-tile 0 (barrier → 8 MFMA)
Region 7: K prefetch (next tile async_load, guarded)
Region 8: GEMM1 sub-tile 1 (barrier → 8 MFMA)
Region 9: Epilogue (normalize → LSE → bf16 store)
```

For EACH region, the report must include:
- CK line range (e.g., `ck_d64_kernel.s:420-510`)
- Native line range (e.g., `native_d64_kernel.s:380-470`)
- Instructions in this region (both sides)
- Dependency chain (producer→consumer within and across regions)
- Waitcnt/barrier placement
- MATCH / DIVERGENT verdict with specific evidence

If a region is DIVERGENT, state: what differs, why it matters, and
whether it affects correctness/performance.

- [ ] **Step 8: Independent verification of the comparison**

Spawn a SEPARATE fresh 1M-context subagent that:
1. Reads the ISA gate report (`phase2_isa_gate_001.md`)
2. Reads BOTH assembly files (native + CK) — via Explore subagents
3. For each region in the report, independently locates the claimed
   line ranges and verifies the comparison claims are accurate
4. Reports: VERIFIED / INVALID per region, with evidence

If ANY region is INVALID (the comparison report made a wrong claim
about the assembly), the entire Task 9 FAILS and must be redone from
Step 2. The comparison agent's work cannot be trusted for regions
the verifier couldn't confirm.

- [ ] **Step 9: If ALL regions VERIFIED + MATCH, commit**

```bash
git add native_d64_kernel.s
git commit -m "fmha_native: Phase 2 ISA snapshot — structural match vs CK verified"
```

If any region is DIVERGENT but verified-accurate, escalate to user
with the specific delta before committing.

---

## Task 10: Final commit — Phase 2 complete

- [ ] **Step 1: Run the full verification suite**

All in a single run each, no filters:
```bash
# Fused kernel correctness
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_fwd_d64 2>&1"
# GPU ref
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_gpu_ref 2>&1"
# Golden end-to-end
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/test_fmha_golden_e2e --golden-full=... --golden-partial=... 2>&1"
# Phase 1 regression (all 7)
for t in test_k_lds test_qk_gemm test_row_max test_softmax test_v_lds test_pv_gemm test_epilog; do
  docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 /root/workspace/build/$t --golden-full=... --golden-partial=... 2>&1"
done
```

Expected: 56/56 + 56/56 + golden e2e 0 mismatch + 49/49 Phase 1.

- [ ] **Step 2: Independent spec review**

Spawn 1M-context subagent to verify all 5 kernel files against the
Phase 2 spec.

- [ ] **Step 3: Independent doc audit**

Spawn subagent to verify knowledge.md, status.md, and docs match
actual code/test state.

- [ ] **Step 4: Update knowledge.md and status.md**

Add Phase 2 results, ISA gate numbers, any new findings.

- [ ] **Step 5: Final commit**

```bash
git add -A  # review staged files carefully
git commit -m "fmha_native: Phase 2 complete — fused kernel, 50/50+48/48 pass, ISA matched"
```

---

## Task Dependencies

```
Task 1 (CMake flags)
  ↓
Task 2 (_lds) ──→ Task 6 (_device) ──→ Task 7 (debug) ──→ Task 8 (golden e2e)
Task 3 (_gemm) ─↗                                            ↓
Task 4 (_softmax) ─↗                                      Task 9 (ISA gate)
Task 5 (_epilog) ─↗                                          ↓
                                                          Task 10 (commit)
```

Tasks 2–5 can run in parallel (no file conflicts). Task 6 depends on
all four. Task 7 depends on 6. Tasks 8–9 depend on 7. Task 10 depends
on everything.

---

## Quality Gate — Applied Identically at EVERY Task

Every task runs the same 5 gates. No exceptions, no shortcuts. The
only thing that varies is which test binaries exist at that point.

### G0: Every Member Verifies Their Own Inputs

Before acting on ANY input — a task description, a formula from the
spec, a file path from a teammate, golden data from a prior phase —
the member must sanity-check it:

- **Formula from spec/plan:** re-derive from first principles or
  cross-check against Phase 1 golden-verified code. The spec had
  wrong formulas before (Q groups-of-4, S_acc bare-HW layout,
  GEMM operand labels). Don't copy blindly.
- **File path from teammate:** `ls` it. Files move, get renamed,
  or were never written.
- **Golden data:** `md5sum` before using. Containers don't share
  /tmp. Golden may not survive reboots.
- **Claim from another member:** "K LDS is done" means nothing
  until you grep the function, build, and run the test yourself.
- **Your own prior work:** re-read the file after editing. Edits
  can silently fail (wrong old_string, partial match).

If something looks wrong, STOP and escalate. Don't implement on
top of a bad foundation — Phase 1 proved that golden catches errors
source analysis misses. Catching one wrong formula at Task 2 saves
a week of debugging at Task 7.

### G1: Artifact Verification

Lead reads the actual file to confirm changes. Never trust the claim.

```bash
# Confirm claimed changes exist
grep -n "<expected_pattern>" <modified_file>
# Confirm old code is removed (if task replaces functions)
grep -n "<old_pattern>" <modified_file>   # expect: no matches
```

### G2: Clean Rebuild

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace && rm -rf build && \
  mkdir build && cd build && cmake .. -GNinja 2>&1 && ninja 2>&1"
```

Zero errors. Warnings reviewed.

### G3: Run ALL Available Tests

Run every test binary that exists at this point. No filter. Every
binary must exit 0.

```bash
# Phase 1 standalone (always available)
for t in test_k_lds test_qk_gemm test_row_max test_softmax \
         test_v_lds test_pv_gemm test_epilog; do
  docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 \
    /root/workspace/build/$t \
    --golden-full=/tmp/fmha-native-isa-match/golden/full \
    --golden-partial=/tmp/fmha-native-isa-match/golden/partial 2>&1"
done

# Fused kernel (available once it compiles)
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 \
  /root/workspace/build/test_fmha_fwd_d64 2>&1"

# GPU ref (available once fused kernel compiles)
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 \
  /root/workspace/build/test_fmha_gpu_ref 2>&1"

# Golden end-to-end (available after Task 8)
docker exec poyenc-fmha bash -c "HIP_VISIBLE_DEVICES=1 \
  /root/workspace/build/test_fmha_golden_e2e \
    --golden-full=... --golden-partial=... 2>&1"
```

### G4: Spec/Plan Compliance Review

Spawn a fresh 1M-context subagent that:
1. Reads this plan's task description for the current task
2. Reads the spec
3. Reads the ACTUAL modified file(s) on disk
4. Verifies the implementation matches both task description and spec
5. Reports COMPLIANT / NON-COMPLIANT with file:line citations

### G5: Record Decisions + Knowledge Update

Every design decision made during the task — deviations from spec,
parameter choices, workarounds, formula corrections — must be written
to `/tmp/fmha-native-isa-match/lead/decisions_phase2.md` BEFORE
marking the task done. Format:

```
YYYY-MM-DD HH:MM | Task N | <decision> | <rationale>
```

Update recall knowledge.md with any new findings (compiler behavior,
hardware surprises, correctness insights). Incremental — one or two
entries per task, not a full rewrite. Skip knowledge update only if
nothing new was learned; NEVER skip decisions recording.

---

### Test Availability by Task

| Test Suite | Available from | Expected pass count |
|-----------|:--------------:|:-------------------:|
| Phase 1 standalone (7 binaries) | always | 49/49 |
| test_fmha_fwd_d64 | Task 6 (first compile) | 56/56 (after Task 7) |
| test_fmha_gpu_ref | Task 6 (first compile) | 56/56 (after Task 7) |
| test_fmha_golden_e2e | Task 8 (created) | 0 mismatch full+partial |
| ISA structural analysis | Task 9 | CFG + dependency match |
