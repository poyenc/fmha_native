# Phase 2: Fused Kernel Design — FMHA D64 BF16 gfx942

> **For agentic workers:** This spec governs Phase 2 of the
> fmha-native-isa-match team. Each task maps to a specific file and has
> explicit acceptance criteria + quality gates. No task is done until
> it passes independent verification.

**Goal:** Assemble the 7 Phase 1 golden-verified inner loops into a
single fused kernel that passes 50/50 correctness + 48/48 GPU ref
tests, with CK-matched ISA.

**Prior art:** Phase 1 commit `e4f1d85` (7 standalone kernels, 49/49
tests). All layouts golden-verified in Phase 0.

---

## Design Decisions (resolved via /grill-me)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Rewrite strategy | Gut existing `src/kernel/` files, keep entry point |
| 2 | LDS layout | Shared K/V, 3 buffers × 2304 elems, 13824 bytes |
| 3 | Pipeline loop | 3-buffer async, sub-tile interleaving |
| 4 | Online softmax | Rescale O_acc BEFORE softmax of new tile |
| 5 | Q load | QLoadOnce before loop; Q = B-operand (lane→M-row) |
| 6 | Causal mask | Tile skip via loop bounds + element mask on boundary |
| 7 | Register budget | Compiler allocates; check after build; intervene if >127 |
| 8 | V staging | Overlap V DRAM load with softmax compute |
| 9 | K prefetch | Sub-tile interleaving per CK's schedule |
| 10 | LDS buffer seq | Hardcode `{1, 2, 1, 0}` (k0=2, k1=2, 3 buffers) |
| 11 | File organization | Keep 5-file split, replace internals |
| 12 | GEMM structure | Sub-tile callable functions in `_gemm.hpp` |
| 13 | Prologue/epilogue | Prefetch-before-loop + guarded-last-iteration |
| 14 | LSE output | Computed in epilog: `log(rsum) + rmax / log2e` |
| 15 | Varlen | Keep `<HasMask, IsVarlen>` template |
| 16 | Launch bounds | `__launch_bounds__(256, 3)` |
| 17 | Waitcnt/barriers | Mirror CK's exact pattern from assembly |
| 18 | Acceptance | 50/50 + 48/48 existing tests + golden O end-to-end |
| 19 | Compile flags | Full CK flag set on `fmha_kernel` target |

---

## MFMA Operand Convention

CK convention — A = LDS, B = register:

| GEMM | A (LDS) | B (register) | C (accumulator) |
|------|---------|-------------|-----------------|
| GEMM0 (QK) | K | Q | S_acc |
| GEMM1 (PV) | V | P | O_acc |

**WHY B=Q/P:** TransposedC maps lane→B's dimension. With B=Q/P,
lane→seqlen_q, so each lane owns exactly ONE M-row. This enables
intra-lane softmax (register-level row reduction, no cross-lane
butterfly).

---

## Pipeline Architecture

### LDS Layout (13824 bytes, shared K/V)

```
Buffer 0: elems [0, 2304)        — 4608 bytes
Buffer 1: elems [2304, 4608)     — 4608 bytes
Buffer 2: elems [4608, 6912)     — 4608 bytes
Total: 6912 elems × 2 bytes = 13824 bytes
```

K and V share the same 3-buffer LDS space. After GEMM0 finishes
reading K from a buffer, V is staged into the freed buffer for GEMM1.

### LDS Buffer Sequence

```c
constexpr int LdsSeq[4] = {1, 2, 1, 0};
//                          ^^^^  ^^^^
//                        K subs  V subs
// K sub 0 → buf 1, K sub 1 → buf 2
// V sub 0 → buf 1, V sub 1 → buf 0
```

Derived from CK's `LdsBufferSequence<3, 3, 2, 2>` specialization.
Guarantees: last V buffer (0) ≠ first two K buffers (1, 2) of the
next iteration.

### Sub-tile counts

```
k0_loops = kQKHeaddim / kK0 = 64 / 32 = 2   (GEMM0 K-dim sub-tiles)
k1_loops = kN0 / kK1        = 64 / 32 = 2   (GEMM1 K-dim sub-tiles)
```

Each sub-tile's `gemm_0` / `gemm_1` call does 2 N-tiles × 2 HW passes
(K16 logical = 2×K8 on gfx942) = 8 MFMA per call, 16 MFMA per GEMM.

### Per-Tile Loop Body (one KV tile iteration)

```
PROLOGUE (before loop):
  async_load K sub 0 → LdsSeq[0] = buf 1
  move K DRAM window by kK0

TILE LOOP (do...while i_total_loops < num_total_loop):

  STAGE 1 — GEMM0 (QK):
    clear S_acc
    for i_k0 = 0 .. k0_loops-2:                    // i_k0=0 only
      async_load K sub i_k0+1 → LdsSeq[i_k0+1]    // K sub 1 → buf 2
      move K DRAM window by kK0
      async_load_fence
      s_barrier
      gemm_0(S_acc, Q_slice[i_k0], K_LDS[LdsSeq[i_k0]])  // read buf 1
    async_load_fence
    s_barrier
    load V[0] from DRAM → V register buffer          // overlap with tail
    gemm_0(S_acc, Q_slice[k0_loops-1], K_LDS[LdsSeq[k0_loops-1]])  // read buf 2

  STAGE 2 — Softmax:
    scale S_acc by scale_s * log2e
    apply causal mask (per-element where n_col > m_row + shift → -inf)
    m_local = intra-lane max over 32 regs
    m_local = cross-half max via ds_bpermute(lane^32)
    m_new = max(m_old, m_local)
    if LdsSeq[k0_loops-1] == LdsSeq[k0_loops]: s_barrier  // K/V share buf
    V shuffle (v_perm_b32) + store V to LDS[LdsSeq[k0_loops]]  // buf 1
    if k1_loops > 1: prefetch next V sub from DRAM
    P = exp2(S_scaled - m_new)
    l_local = intra-lane sum of P + cross-half sum via ds_bpermute
    rescale: o_acc *= exp2(m_old - m_new); l = exp2(m_old-m_new)*l + l_local
    m_old = m_new
    P_bf16 = bf16_trunc(P)

  STAGE 3 — GEMM1 (PV):
    for i_k1 = 0 .. k1_loops-2:                     // i_k1=0 only
      s_barrier (LDS sync for V)
      gemm_1(O_acc, P_slice[i_k1], V_LDS[LdsSeq[k0_loops+i_k1]])  // read buf 1
      V shuffle + store next V → LdsSeq[k0_loops+i_k1+1]  // buf 0
      move V DRAM window
    i_total_loops++
    if i_total_loops < num_total_loop:
      move K DRAM window by kN0 (next tile)
      if LdsSeq[0] == LdsSeq[k0_loops+k1_loops-2]: s_barrier
      async_load K sub 0 → LdsSeq[0]  // prefetch next tile K → buf 1
      move K DRAM window by kK0
    s_barrier (LDS sync for last V)
    gemm_1(O_acc, P_slice[k1_loops-1], V_LDS[LdsSeq[k0_loops+k1_loops-1]])  // read buf 0

  while (i_total_loops < num_total_loop)

EPILOGUE (after loop):
  if kStoreLSE: lse = log(l) + m (with scaling); store to DRAM
  o_acc /= l (per-element, guard l==0 → 0)
  bf16_trunc + buffer_store_dwordx2 (8 stores, SwizzleA'd headdim)
```

### Causal Mask — Tile Skip

Before the loop, compute tile range:
```c
auto [seqlen_k_start, seqlen_k_end] = mask.GetTileRangeAlongX(
    q_origin, kM0, kN0);
num_total_loop = ceildiv(seqlen_k_end - seqlen_k_start, kN0);
```

For causal: `seqlen_k_end = ceil_to_tile(min(m_tile*kM0 + kM0 - 1 + seqlen_k - seqlen_q + 1, seqlen_k))`.
Q tile at row 0 processes fewer KV tiles than Q tile at the last row.
If `num_total_loop <= 0`: zero O_acc, return.

Within the loop, the boundary tile applies per-element mask:
```c
for r in 0..31:
    n_col = kv_tile_offset + (r/8)*16 + (lane/32)*8 + (r%8)
    if n_col > m_row + shift: S[r] = -inf
```

---

## File Organization

```
src/kernel/
  fmha_fwd_d64_device.hpp   — main loop, Q load, tile skip, rescale,
                               online softmax orchestration
  fmha_fwd_d64_lds.hpp      — K async copy, V shuffle+store (from K1, K5)
  fmha_fwd_d64_gemm.hpp     — gemm_0 sub-tile, gemm_1 sub-tile (from K2, K6)
  fmha_fwd_d64_softmax.hpp  — scale, mask, rmax, exp2, rsum (from K3, K4)
  fmha_fwd_d64_epilog.hpp   — normalize, LSE, bf16 store (from K7)
  fmha_fwd_d64_kernel.cpp   — template instantiations (unchanged)
```

Each inner file provides `__device__ __forceinline__` functions called
by `_device.hpp`. Functions are sub-tile granularity for GEMMs (one
call per k0/k1 loop iteration).

---

## Compile Flags (fmha_kernel target)

```cmake
target_compile_options(fmha_kernel PRIVATE
    --offload-arch=gfx942 --save-temps
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1
    -mllvm -amdgpu-early-inline-all=true
    -mllvm -amdgpu-function-calls=false
    -mllvm --lsr-drop-solution=1
    -mllvm -enable-post-misched=0
    -fbracket-depth=1024
    -fno-offload-uniform-block
    -fgpu-flush-denormals-to-zero)
```

`__launch_bounds__(256, 3)` on the kernel entry point.

---

## Quality Gates

Every task follows the same verification protocol as Phase 1:

### Gate 1: Build (per task)
- Clean build with full CK flags. Zero errors. Warnings reviewed.

### Gate 2: Test (per task)
- Run `test_fmha_fwd_d64` and `test_fmha_gpu_ref` — report pass/fail
  count and any regressions.
- Run Phase 1 standalone tests — must not regress (49/49).

### Gate 3: Independent QA (per task)
- Lead spawns fresh subagent for clean rebuild + test run.
- Lead verifies claimed changes exist in files (grep, not trust).

### Gate 4: Spec Review (per phase milestone)
- Independent 1M-context subagent reads spec + code, verifies
  compliance. COMPLIANT/NON-COMPLIANT verdict per file.

### Gate 5: ISA Structural Review (after full pipeline compiles)
- Use `/amdgpu-asm-analyzer` on BOTH native and CK assembly.
- Compare CFG structure (basic blocks, loop shape, branch targets).
- Compare dependency chains (producer→consumer for each pipeline stage).
- Compare waitcnt/barrier placement relative to consumers.
- Compare scheduling order (MFMA interleaving with memory ops).
- Resource usage (VGPR ≤ 127, 0 spill) is a secondary sanity check.
- Instruction counts alone are NOT sufficient — structural match is
  the gate.

### Gate 6: Doc Audit (before commit)
- Independent subagent verifies knowledge.md, status.md, and docs
  match actual code/test state. Stale claims flagged.

### Enforcement Rule
**"Done" is a claim. The artifact is the fact.**
- Before marking any task done: grep the file, run the test, read the
  output. Never trust a teammate's or subagent's report without
  checking the artifact.

---

## Task Breakdown

### 2.1 — Update CMake compile flags for fmha_kernel target
**File:** `CMakeLists.txt`
**What:** Replace existing `target_compile_options(fmha_kernel ...)` with
full CK flag set (see Compile Flags section above). Add
`__launch_bounds__(256, 3)` in the kernel source if not present.
**Acceptance:** `fmha_kernel` builds with all flags. `--save-temps`
produces `.s` assembly file. Phase 1 tests unaffected (49/49).
**Role:** impl

### 2.2 — Rewrite `_lds.hpp`: K async copy + V shuffle+store
**File:** `src/kernel/fmha_fwd_d64_lds.hpp`
**What:** Replace all existing functions with:
- `async_copy_k_subtile(k_lds_base, k_srd, stride_k, kv_offset, seqlen_k, buf_idx)` —
  async DRAM→LDS for one K sub-tile (kK0=32 headdim slice). Uses
  `__builtin_amdgcn_raw_ptr_buffer_load_lds`. Correct n_base =
  `(lane>>4)*4 + warp`. From Phase 1 K1.
- `store_v_to_lds(v_regs, v_lds_base, buf_idx)` — shuffle via
  `v_perm_b32` (selectors 0x01000504/0x03020706) + `ds_write2_b32`.
  From Phase 1 K5.
- `load_v_from_dram(v_regs, v_srd, stride_v, kv_offset, seqlen_k)` —
  `buffer_load_dwordx2` into register buffer for one V sub-tile.
- Buffer base computation: `buf_base_elems(buf_idx) = buf_idx * 2304`.
**Blocks:** 2.3, 2.4 (GEMMs need LDS functions)
**Acceptance:** Build clean. Functions are `__device__ __forceinline__`.
Phase 1 tests unaffected.
**Role:** impl

### 2.3 — Rewrite `_gemm.hpp`: GEMM0 + GEMM1 sub-tile functions
**File:** `src/kernel/fmha_fwd_d64_gemm.hpp`
**What:** Replace all existing functions with:
- `gemm0_subtile(s_acc, q_regs, k_lds_base, buf_idx, k_subtile_idx)` —
  One sub-tile of GEMM0: read K from LDS via `ds_read_b128`, execute
  2 N-tiles × 2 HW passes = 8 MFMA. Q is B-operand (register), K is
  A-operand (LDS). SwizzleA on K reads (bit2/3 swap). From Phase 1 K2.
- `gemm1_subtile(o_acc, p_regs, v_lds_base, buf_idx, v_subtile_idx)` —
  One sub-tile of GEMM1: read V from LDS via `ds_read_b128`, execute
  8 MFMA. P is B-operand (register), V is A-operand (LDS). No SwizzleA
  on V reads. From Phase 1 K6.
- `clear_acc(acc)`, `slice_q(q_regs, k_subtile_idx)`,
  `slice_p(p_regs, v_subtile_idx)` — helpers.
**Blocks:** 2.6 (device loop calls these)
**Acceptance:** Build clean. Functions match Phase 1 K2/K6 inner loops.
**Role:** impl

### 2.4 — Rewrite `_softmax.hpp`: scale + mask + rmax + exp2 + rsum
**File:** `src/kernel/fmha_fwd_d64_softmax.hpp`
**What:** Replace the buggy butterfly functions with:
- `softmax_scale_and_mask(s_acc, scale_s_log2e, seqlen_k, kv_offset,
  m_row, has_mask, mask_shift)` — scale S_acc, apply boundary mask
  and causal mask per-element. From Phase 1 K4.
- `softmax_row_max(s_acc)` — intra-lane max over 32 regs + 1
  ds_bpermute cross-half. Returns 1 fp32 scalar. From Phase 1 K3.
- `softmax_row_sum(p_fp32)` — intra-lane sum + 1 ds_bpermute
  cross-half. Returns 1 fp32 scalar. From Phase 1 K4.
- `softmax_exp2(s_acc, row_max)` — `exp2(s - m)` per element.
  Returns P in fp32. From Phase 1 K4.
- `softmax_p_to_bf16(p_fp32, p_bf16)` — bf16 truncation (not RNE).
  From Phase 1 K4.
**Acceptance:** Build clean. The butterfly functions are GONE. Phase 1
K3/K4 standalone tests unaffected.
**Role:** impl

### 2.5 — Rewrite `_epilog.hpp`: normalize + LSE + bf16 store
**File:** `src/kernel/fmha_fwd_d64_epilog.hpp`
**What:** Replace both existing functions with:
- `epilog_store(o_acc, rsum, rmax, o_srd, stride_o, lse_base,
  seqlen_q, m_tile_idx, scale_s)` — normalize O_acc by rsum, compute
  LSE (`log(rsum) + rmax / log2e`), bf16 truncation, 8×
  `buffer_store_dwordx2` with SwizzleA'd headdim. From Phase 1 K7 +
  LSE from existing code.
**Acceptance:** Build clean. Function matches Phase 1 K7 store pattern
+ LSE output.
**Role:** impl

### 2.6 — Rewrite `_device.hpp`: main pipeline loop
**File:** `src/kernel/fmha_fwd_d64_device.hpp`
**What:** Replace the inner pipeline with the 3-buffer async schedule:
- Keep: template `<HasMask, IsVarlen>`, kargs unpacking, varlen
  pointer setup, batch/head indexing, LDS pointer.
- Replace: Q load (QLoadOnce, 32 bf16 regs), tile range computation
  (causal skip), early exit, K prefetch prologue, `do...while` loop
  body (GEMM0 → softmax → GEMM1 with sub-tile interleaving per
  Pipeline Architecture section), online softmax rescaling, epilog call.
- LdsSeq: `constexpr int LdsSeq[4] = {1, 2, 1, 0}`.
- `__launch_bounds__(256, 3)` on kernel entry.
**Depends on:** 2.2, 2.3, 2.4, 2.5 (calls all four).
**Acceptance:** Build clean. `test_fmha_fwd_d64` runs (may not pass
all 50 yet — this is the wiring task).
**Role:** impl

### 2.7 — First correctness pass: debug and fix until tests pass
**File:** all `src/kernel/` files
**What:** Run `test_fmha_fwd_d64` (50 tests) + `test_fmha_gpu_ref`
(48 tests). Debug failures using Phase 1 golden data as intermediate
checkpoints. Fix issues iteratively.
**Acceptance:** 50/50 + 48/48 pass. Phase 1 standalone tests still
49/49. All in a single run, no filters.
**Role:** impl + debug (on-demand)

### 2.8 — Golden end-to-end test: bit-match vs CK O output
**File:** `tests/test_fmha_fwd_d64.cpp` (add test) or new test
**What:** Add a test that compares fused kernel O_dram against CK's
`o_dram.bin` golden output (full + partial tiles). Bit-exact bf16
comparison.
**Acceptance:** 0 mismatches vs CK golden O for both tile sizes.
**Role:** impl

### 2.9 — ISA gate: extract assembly, verify instruction profile
**File:** assembly output from `--save-temps`
**What:** Extract the fused kernel assembly. Count instructions and
compare against CK target:
  - 32 v_mfma_f32_32x32x8_bf16
  - 8 buffer_store_dwordx2
  - 16 ds_read_b128
  - 4 ds_write2_b32
  - 2 ds_bpermute_b32
  - 44 v_perm_b32
  - VGPR ≤ 127, AGPR = 0, 0 spill, LDS = 13824
**Acceptance:** All counts match CK. If not, document deltas and
decide: fix (code restructure) or accept (justified deviation).
**Role:** prof

### 2.10 — Commit Phase 2
**What:** After all gates pass:
1. Run full test suite (50/50 + 48/48 + 49/49 + golden end-to-end)
2. Independent spec review (subagent vs this doc)
3. Independent doc audit (knowledge.md, status.md)
4. Commit with descriptive message

**Acceptance:** All tests pass, all reviews clean, docs accurate.
**Role:** lead

---

## Task Dependencies

```
2.1 (CMake flags)
  ↓
2.2 (_lds.hpp) ──→ 2.6 (_device.hpp) ──→ 2.7 (debug) ──→ 2.8 (golden e2e)
2.3 (_gemm.hpp) ─↗                                          ↓
2.4 (_softmax.hpp) ─↗                                    2.9 (ISA gate)
2.5 (_epilog.hpp) ─↗                                        ↓
                                                         2.10 (commit)
```

Tasks 2.2–2.5 can run in parallel (no file conflicts). 2.6 depends on
all four. 2.7 depends on 2.6. 2.8–2.9 depend on 2.7. 2.10 depends on
everything.

---

## Verification Checklist (per task)

For every task marked "done":

- [ ] Lead greps the modified file to confirm changes exist
- [ ] Lead runs clean rebuild in container
- [ ] Lead runs test suite (no filters, all tests, exit 0)
- [ ] If task claims a function was removed: grep confirms it's gone
- [ ] If task claims a flag was added: grep CMakeLists.txt confirms
- [ ] Phase 1 standalone tests still 49/49 (no regression)
