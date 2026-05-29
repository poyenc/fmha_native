# FMHA Native D64 — Incremental Kernel Validation Design

**Date:** 2026-05-29
**Branch:** `isa-match-rewrite`
**Target:** gfx942 (MI-300X, CDNA3)
**Supersedes:** `docs/isa-rewrite-spec.md`

## Goal

### Primary: Correctness (must achieve)

Pass all 50 correctness tests (`test_fmha_fwd_d64`: 50/50 PASS) using
CK-matched data layouts verified by Phase 0 golden dumps. No regression
on GPU ref tests (`test_fmha_gpu_ref`: 48/48 PASS).

**Done when:** 50/50 + 48/48 tests pass.

### Secondary: ISA Structure Match (target)

Match CK's assembly structure in the generated kernel:
- VGPR ≤ 127, accum_offset = 128, AGPR = 0
- 0 scratch spills, 0 function calls (`s_swappc`)
- 32 `v_mfma_f32_32x32x8bf16` (fully unrolled)
- LDS = 13824 bytes
- Hot-path instruction types match CK (async K copy, MFMA GEMM1,
  V-through-LDS, buffer_store O)

**Done when:** `/amdgpu-asm-analyzer` report matches CK's structural
report within ±10% per instruction category.

### Stretch: Performance Parity

All benchmark configs within ±2% TFLOPS of CK baseline. MFMA
utilization ≥ 33%.

**Done when:** Benchmark sweep shows parity.

### Method

Build and validate each pipeline phase as a standalone kernel (Phase 0
+ Phase 1) before assembling the final kernel (Phase 2), then verify
ISA and performance (Phase 3).

## Prior Approach — Why It Failed

The prior session took a bottom-up approach: write helper functions per
instruction pattern (K async copy, MFMA GEMM1, V repack, O store), then
wire them into the pipeline. This failed because three data layout
mismatches between pipeline phases only surfaced during integration:

1. **GEMM1 V transposition** — V must be transposed in LDS for MFMA
   reads; the written layout was incompatible with the read layout.
2. **O store cross-lane packing** — `buffer_store_dwordx2` needs 4
   consecutive hdim values per lane; MFMA C-output has 1 hdim/lane.
3. **GEMM0 + Q load coupling** — interleaved GEMM0 requires
   k_sub-dependent Q load pattern; cannot wire one without the other.

549 lines of helpers were written, none could be integrated. Code was
reverted. The NaN bug in `softmax_exp` (fully-masked rows producing
`exp2f(-INF - (-INF))`) was fixed in this session.

**Current state:** 7/50 tests pass, 43 cos_only failures remain.

## Phase 0: CK Layout Understanding

Before writing any standalone kernel, map every `distributed_tensor` in
CK's FMHA pipeline and every `shuffle_tile()` transformation. This is
the research task the prior session skipped.

### Why This Comes First

The prior session's 3 layout mismatches (V transposition, O cross-lane
exchange, Q interleaved load) all correspond to `shuffle_tile()`
boundaries where one thread distribution transforms into another. Without
understanding these transformations, standalone kernels validate against
assumed layouts that may be wrong.

### Distributed Tensor Map

For each tensor in CK's pipeline, document:

- **Thread mapping**: which thread (lane, warp) owns which matrix element
- **Register layout**: which VGPR index holds which element
- **Source**: created by load, MFMA, or shuffle_tile

| Tensor | Created By | Distribution | Consumed By |
|--------|-----------|-------------|-------------|
| Q_tile | `load_tile()` from DRAM | Q load distribution (k_sub-dependent?) | GEMM0 B operand |
| K_lds | `async_load_tile_raw()` to LDS | LDS padded row layout | GEMM0 A operand via `ds_read_b128` |
| S_acc | GEMM0 C-output | MFMA C: physical `n=lane%32, m=(r/4)*8+(lane/32)*4+(r%4)` (TransposedC) | softmax (in place) |
| P_tile | softmax(S_acc), cast bf16 | **same as S_acc** (no relayout) | GEMM1 A operand directly |
| V_loaded | `load_tile()` from DRAM | V DRAM dist (K3 outer, N1 inner) | `shuffle_tile()` → V_shuffled |
| **V_shuffled** | **`shuffle_tile(V_loaded)`** — intra-thread `v_perm_b32` | **N1 outer, K3 inner** (per-thread 4×2→2×4 transpose, same lane) | LDS write (`store_tile`) |
| V_lds | LDS (written from V_shuffled) | transposed padded layout (`MakeVLdsBlockDescriptor`) | GEMM1 B operand via `ds_read_b128` |
| O_acc | GEMM1 C-output | MFMA C: same TransposedC as S_acc | Default2D epilogue (×1/rsum, cast bf16) |
| O_final | Default2D epilogue | **same C-output dist** (no reshuffle) — normalized bf16 | `buffer_store_dwordx2` |

> ✅ **RESOLVED (Phase 0.1–0.4, 2026-05-29):** CK's pipeline has exactly
> ONE `shuffle_tile()` — on **V** (intra-thread `v_perm_b32`, per-thread
> 4×2→2×4 register transpose, same lane; see `shuffle_trace.md` §2–3).
> **There is NO P shuffle** — S/P/O share the identical 32×32 TransposedC
> distribution, so P feeds GEMM1 A directly from S's C-output after
> in-place softmax + bf16 cast (`shuffle_trace.md` §4). **There is also NO
> O shuffle** — the epilogue is **Default2DEpilogue** (asm-confirmed from
> the kernel symbol; 0.1's "CShuffle" was a wrong source-guess). O_acc is
> normalized (×1/rsum), cast to bf16, and `buffer_store`d directly in the
> C-output distribution. So NONE of the three original "unknowns" is a
> cross-lane shuffle: V = intra-thread v_perm; P = direct feed; O = direct
> store. The only remaining layout details to golden-verify are the K/V
> LDS byte layouts and the exact register orders (Phases 0.6–0.11).

### Verification Method

Phase 0 uses the **full team pipeline** — not just research reading
source code. Each tensor layout hypothesis is verified by building and
running a small kernel.

Reading CK source produces a **hypothesis** ("thread L, register I
holds element [M, N]"). The hypothesis is only confirmed when a
verification kernel produces output that matches CK's actual dump
bit-exact.

**Per-tensor verification cycle:**

```
research: read CK source → write hypothesis document
research: write CK instrumented dump for this tensor (if not done)
build:    build CK instrumented dump in poyenc-ck
test:     run CK dump, save golden file
impl:     write verification kernel that arranges data per hypothesis
build:    build verification kernel in poyenc-fmha
test:     compare verification dump against CK golden — bit-exact
debug:    if mismatch → investigate, report to research
research: revise hypothesis if needed, repeat from impl step
```

A hypothesis is **confirmed** only when the verification kernel's dump
matches CK's golden dump bit-exact. Until then, it's a hypothesis —
not a fact. The tensor layout map document only contains confirmed
layouts.

### Task Breakdown

1. **Read CK source** *(research)*: trace the FMHA pipeline, identify
   every `distributed_tensor` type and `shuffle_tile()` call. Understand
   the partition index system and how it maps threadIdx.x to tensor
   elements during address computation.

2. **Write layout hypotheses** *(research)*: for each tensor, document
   the hypothesized thread-to-element and register-to-element mapping.

3. **Instrument CK to dump** *(research)*: write a CK kernel variant
   that dumps per-thread tensor data at each phase boundary.

4. **Build and run CK dumps** *(build, test)*: produce golden binary
   files for each tensor at each phase boundary.

5. **Write verification kernels** *(impl)*: for each tensor, write a
   small HIP kernel that arranges data according to the hypothesized
   layout and dumps registers/LDS to DRAM.

6. **Build and test verification kernels** *(build, test)*: compare
   against CK golden dumps. Bit-exact match confirms the hypothesis.

7. **Debug mismatches** *(debug)*: when a verification kernel's dump
   doesn't match CK's golden, investigate and report to research.

8. **Write tensor layout map** *(research)*: compile all **confirmed**
   layouts into `docs/ck-tensor-layout-map.md`.

### Key CK Source Locations

- Async K copy: `include/ck_tile/core/arch/amd_buffer_addressing.hpp:1342-1356`
- m0 helpers: `include/ck_tile/core/arch/utility.hpp:19-28`
- shuffle_tile: `include/ck_tile/core/tensor/shuffle_tile.hpp` (or similar)
- FMHA pipeline: search for `fmha_fwd_` in CK's `ck_tile` directory
- Tile distribution types: `include/ck_tile/core/tensor/tile_distribution.hpp`

### Deliverables

- `docs/ck-tensor-layout-map.md`: complete distributed tensor map with
  thread/register layouts for all pipeline tensors
- Golden binary files for each tensor at each phase boundary
- Verified `shuffle_tile()` transformation specs for P, V, and O

## New Approach — Incremental Kernel Validation

Build 6 standalone HIP kernels, one per pipeline phase. Each kernel:

- Is a separate `__global__` kernel with its own test binary
- Has a CPU reference that validates mathematical output
- Has a CK golden reference that validates register/LDS layout
- Can be independently assembled and analyzed

The standalone kernels are **prototypes to validate layout
understanding**. The final FMHA kernel is rewritten from scratch,
copy-pasting the validated inner loops with new pipeline glue.

## Kernel Ladder

Build and validate in this order. Each kernel depends on the layout
contract proven by its predecessors.

| Step | Kernel | Phase | Key Instructions |
|------|--------|-------|-----------------|
| 1 | `test_k_lds` | K async copy to LDS | `buffer_load_dword` w/ lds, m0 setup |
| 2 | `test_qk_gemm` | S = Q × K^T | `buffer_load_dwordx4` (Q), `ds_read_b128` (K), `v_mfma_f32_32x32x8bf16` |
| 3 | `test_row_max` | Row-max reduction across 64 columns | `ds_bpermute_b32`, butterfly pattern |
| 4 | `test_softmax` | Softmax exp + sum + rescale (2-tile loop) | `v_exp_f32`, `ds_bpermute_b32` |
| 5 | `test_v_lds` | V load, bf16 repack, transposed LDS write | `buffer_load_dwordx2`, `v_perm_b32`, `ds_write2_b32` |
| 6 | `test_pv_gemm` | O_acc = P × V | `v_perm_b32` (P truncate), `ds_read_b128` (V), `v_mfma_f32_32x32x8bf16` |
| 7 | `test_epilog` | O normalize, bf16 pack, store | `v_perm_b32`, `buffer_store_dwordx2` |

### Why This Order

Data movement before compute. Steps 1 and 4 validate LDS layouts before
any GEMM consumes them. This catches the layout mismatches that killed
the prior approach before any MFMA logic is involved.

## Register Layout Convention

All kernels must agree on the MFMA `v_mfma_f32_32x32x8bf16` C-output
layout. This is the contract that flows through the entire pipeline.

```
lane_id  = threadIdx.x          (0..63)
k_sub    = lane_id / 32         (0 or 1)
n_pos    = lane_id % 32         (column index)

For v_mfma_f32_32x32x8bf16 (TransposedC):
  C[i] at lane l = result[warp*32 + k_sub*16 + i, l%32]
  where i = 0..15, warp = threadIdx.x / 64

Applied to FMHA:
  s_acc_n0[i] at lane l = S[warp*32 + k_sub*16 + i, l%32]         (columns 0..31)
  s_acc_n1[i] at lane l = S[warp*32 + k_sub*16 + i, 32 + l%32]    (columns 32..63)
```

Each CPU reference and golden generator produces a register map in the
format `float[64][16][2]` — 64 lanes × 16 registers × 2 sub-tiles
(n0, n1). GPU kernels dump register state in the same format.

### Phase Boundary Contracts

Each phase has an input and output register contract. The golden
generator includes a **contract compatibility check** that verifies the
output format of phase N matches the input format expected by phase N+1.

| Phase Boundary | Contract |
|----------------|----------|
| K LDS → QK GEMM | LDS byte image: `row * 72 + col` padded layout, 2 buffers |
| QK GEMM → Softmax | S in MFMA C-output registers: `float[64][16][2]` |
| Softmax → PV GEMM | P in same register layout as S, rmax/rsum as `float[64][16]` |
| V LDS → PV GEMM | LDS byte image: transposed layout, V[hdim, seqk] ordering |
| PV GEMM → Epilog | O_acc in MFMA C-output registers: `float[64][16][2]` |

## LDS Layout Convention

K and V use the same padded LDS format with 3 buffers totaling 13824
bytes.

```
LDS address(row, col) = buf_base + (row * kPaddedRowStride + col) * 2
kPaddedRowStride = 72  (64 data elements + 8 padding for bank conflict avoidance)

buf0: bytes     0 ..  4607  (K cols 0..31 or V first half)
buf1: bytes  4608 ..  9215  (K cols 32..63 or V second half)
buf2: bytes  9216 .. 13823  (spare / V repack overlap)
```

### K async copy addressing

The `buffer_load_dword` with LDS flag auto-distributes at
`m0 + lane_id * 4`. The `m0` value encodes the target LDS row and
column within the padded layout.

### V transposed layout

V is stored transposed so that GEMM1's `ds_read_b128` reads V with
`lane%32 = hdim` (not seqk). The exact transposition formula must be
extracted from CK's source code (`amd_buffer_addressing.hpp`) and
validated with a CK instrumented LDS dump.

## Three-Level Validation

### Level 1: CPU Reference (sanity check)

Each kernel has a CPU reference function that computes the expected
output given the same inputs. Uses the existing CPU ref precision
pipeline (bf16 inputs, fp32 accumulation, standard multiply-add).

**Tolerance:** max abs error < 1e-3, cosine similarity > 0.99998.
Not bit-exact due to FMA vs separate multiply-add differences.

| Kernel | CPU Ref | Inputs | Outputs |
|--------|---------|--------|---------|
| `test_k_lds` | `ref_k_lds` | K matrix, stride, kv_offset, seqlen_k | LDS byte image (9216 bytes) — byte-exact |
| `test_qk_gemm` | `ref_qk_gemm` | Q matrix, K LDS image, warp_id | S register map `[256][32]` — tolerance |
| `test_row_max` | `ref_row_max` | S register map | rmax `[256][16]` — exact (scalar ops) |
| `test_softmax` | `ref_softmax` | S register maps (2 tiles) | P register map, rsum (per iter) — tolerance |
| `test_v_lds` | `ref_v_lds` | V matrix, stride, kv_offset, seqlen_k | LDS byte image (transposed) — byte-exact |
| `test_pv_gemm` | `ref_pv_gemm` | P (bf16-truncated), V LDS image | O_acc register map `[256][32]` — tolerance |
| `test_epilog` | `ref_epilog` | O_acc, rsum, rmax, stride_o | bf16 O matrix — tolerance |

Register dump format: `float[256][32]` — 256 threads × 32 values
(16 regs for n0, 16 regs for n1). Each thread writes at
`out[threadIdx.x * 32 + i]` for `acc_n0[i]` and
`out[threadIdx.x * 32 + 16 + i]` for `acc_n1[i]`.

### Level 2: CK Instrumented Dumps (primary gate — bit-exact)

Golden reference files produced by Phase 0 (CK Layout Understanding).
Instrumented CK kernel dumps per-thread tile data at every phase
boundary for a small test case.

**Dumps required:**

| # | After | What to dump | Format |
|---|-------|-------------|--------|
| 1 | K copy to LDS | LDS byte image | raw bytes (9216) |
| 2 | QK GEMM | S_acc per thread | `float[256][32]` |
| 3 | Row max | rmax per thread | `float[256][16]` |
| 4 | Softmax exp + sum | P per thread, rsum | `float[256][32]` + `float[256][16]` |
| 5 | P shuffle | P_shuffled per thread | `float[256][32]` |
| 6 | V write to LDS | LDS byte image (transposed) | raw bytes (9216) |
| 7 | PV GEMM | O_acc per thread | `float[256][32]` |
| 8 | O shuffle (pre-store) | O_shuffled per thread | `float[256][32]` |

Each standalone kernel test loads the corresponding golden file and
compares bit-exact. This is the primary quality gate.

The `shuffle_tile()` dumps (#5, #8) are critical — they validate the
three unknown distribution transformations (P shuffle, V shuffle, O
shuffle) that caused all prior failures.

### Level 3: Assembly Structural Diff (ISA quality gate)

After each standalone kernel passes functional verification, extract its
assembly and compare against CK's corresponding region.

**Per-kernel assembly checklist:**

| Kernel | Must contain | Must NOT contain | Expected count |
|--------|-------------|-----------------|----------------|
| `test_k_lds` | `buffer_load_dword` (w/ lds) | `buffer_load_b128`, `scratch_*` | 8 per buf |
| `test_qk_gemm` | `v_mfma_f32_32x32x8bf16`, `ds_read_b128` | `ds_bpermute`, `scratch_*` | 16 MFMA, 16 ds_read |
| `test_row_max` | `ds_bpermute_b32` | `v_mfma`, `scratch_*` | 80 bpermute |
| `test_softmax` | `v_exp_f32` | `scratch_*` | 32 exp |
| `test_v_lds` | `buffer_load_dwordx2`, `v_perm_b32`, `ds_write2_b32` | `scratch_*` | 4 loads, 8 perm, 4 writes |
| `test_pv_gemm` | `v_mfma_f32_32x32x8bf16`, `ds_read_b128`, `v_perm_b32` | `ds_bpermute`, `scratch_*` | 16 MFMA, 16 ds_read |
| `test_epilog` | `buffer_store_dwordx2`, `v_perm_b32` | `scratch_*` | 8 stores |

Uses `/amdgpu-asm-analyzer` on each standalone kernel's `.s` file.

Note: standalone kernels will have lower VGPR counts than the full
kernel (fewer simultaneously live values). Assembly diff validates
instruction selection and scheduling, not register pressure. Register
pressure is validated only in the final integrated kernel.

## Test Infrastructure

### Test binary structure

Each `test_*.cpp`:

1. Allocate inputs on host, copy to device
2. Launch standalone kernel
3. Copy results (register dump / LDS dump) to host
4. Run CPU reference → compare
5. Optionally load CK golden file → compare (`--golden <dir>` flag)

### Test cases per kernel

Two test cases per standalone kernel:

| Case | Purpose |
|------|---------|
| Full tile | seqlen_q ≥ 128, seqlen_k ≥ 64 — no boundary masking |
| Partial tile | seqlen_q = 17 or seqlen_k = 33 — validates OOB guarding |

The full 50-configuration sweep is reserved for the final integrated
kernel.

### Softmax multi-tile test

`test_softmax` runs a 2-iteration mini-loop to validate online softmax
rescaling:

- Iteration 0: establishes rmax, rsum from S tile 0
- Iteration 1: triggers rescaling (new max may differ), updates rsum,
  rescales o_acc

This directly validates the code path where the NaN bug occurred
(`exp2f(-INF - (-INF))` when a fully-masked row transitions to a valid
row on the next tile).

### Register dump format

GPU kernels dump register state to DRAM as `float[256][32]` — 256
threads × 32 values (16 regs for n0, 16 regs for n1). Each thread
writes at `out[threadIdx.x * 32 + i]` for `acc_n0[i]` and
`out[threadIdx.x * 32 + 16 + i]` for `acc_n1[i]`.

LDS kernels dump raw bytes (9216 per double-buffer).

## File Organization

```
src/
  kernels/                 # standalone validation kernels
    k_lds.hpp              # step 1: K async copy to LDS
    qk_gemm.hpp            # step 2: QK GEMM via MFMA
    row_max.hpp            # step 3: row-max butterfly reduction
    softmax.hpp            # step 4: softmax exp + sum + rescale
    v_lds.hpp              # step 5: V load/repack/transposed LDS write
    pv_gemm.hpp            # step 6: PV GEMM via MFMA
    epilog.hpp             # step 7: O normalize + bf16 pack + store
  refs/                    # CPU references (one per kernel)
    ref_k_lds.cpp / .hpp
    ref_qk_gemm.cpp / .hpp
    ref_row_max.cpp / .hpp
    ref_softmax.cpp / .hpp
    ref_v_lds.cpp / .hpp
    ref_pv_gemm.cpp / .hpp
    ref_epilog.cpp / .hpp
  kernel/                  # existing FMHA kernel (untouched until final rewrite)
    fmha_fwd_d64_device.hpp
    fmha_fwd_d64_gemm.hpp
    fmha_fwd_d64_lds.hpp
    fmha_fwd_d64_softmax.hpp
    fmha_fwd_d64_epilog.hpp
    fmha_fwd_d64_kernel.cpp
  runner/                  # existing test infrastructure (untouched)
    cpu_ref.cpp / .hpp
    gpu_ref.cpp / .hpp
    buffers.cpp / .hpp
    params.hpp
tests/
  test_k_lds.cpp           # step 1 test binary
  test_qk_gemm.cpp         # step 2 test binary
  test_row_max.cpp         # step 3 test binary
  test_softmax.cpp         # step 4 test binary
  test_v_lds.cpp           # step 5 test binary
  test_pv_gemm.cpp         # step 6 test binary
  test_epilog.cpp          # step 7 test binary
  test_fmha_fwd_d64.cpp    # existing 50-test suite (untouched)
  test_fmha_gpu_ref.cpp    # existing GPU ref test (untouched)
tools/
  ck_instrumented/         # CK instrumented dump kernel variant
docs/
  ck-tensor-layout-map.md  # Phase 0 deliverable: complete tensor distribution map
  superpowers/specs/       # design spec (this file)
```

## Overall Flow

```
Phase 0: CK Layout Understanding
  → instrument CK, dump every distributed_tensor, map shuffle_tile()
  → produce golden files + docs/ck-tensor-layout-map.md

Phase 1: Standalone Kernels (7 kernels)
  → validate against golden dumps (bit-exact) + CPU ref (tolerance)
  → assembly structural diff per kernel

Phase 2: Final Kernel Rewrite
  → copy-paste validated inner loops, new pipeline glue
  → incremental integration with 50-test suite

Phase 3: ISA Quality Gate + Performance
  → assembly analysis, benchmark sweep, PMC profiling
```

## Progression to Final Kernel

After all 7 standalone kernels pass at all 3 validation levels:

1. **Copy-paste validated inner loops** — the `for` loops that issue
   MFMAs, the LDS addressing expressions, the butterfly reductions, the
   epilog packing. These are lifted verbatim into the final kernel.

2. **Write new pipeline glue** — register lifetime management, phase
   sequencing, `sched_barrier` placement, `launch_bounds` tuning. This
   is the only new code.

3. **Incremental integration** — add one phase at a time to the final
   `fmha_fwd_d64_device.hpp`, running the 50-test suite after each
   addition:
   - K LDS + QK GEMM → verify S is correct
   - Add softmax → verify P is correct
   - V LDS + PV GEMM → verify O_acc is correct
   - Add epilog → verify final O matches

4. **Register pressure tuning** — once all phases are integrated,
   address VGPR count, spills, and scheduling with `launch_bounds`,
   `sched_barrier`, and targeted restructuring.

5. **Assembly quality gate** — run `/amdgpu-asm-analyzer` on the final
   kernel, diff against CK's structural report.

6. **Performance verification** — benchmark sweep, PMC profiling, ATT
   stall analysis.

## Existing Code

The current uncommitted changes on `isa-match-rewrite` (549 insertions +
NaN fix) are committed as a checkpoint before starting the new work.
These contain validated patterns worth copy-pasting:

- MFMA intrinsic: `__builtin_amdgcn_mfma_f32_32x32x8bf16_1k`
- SRD construction: `__builtin_amdgcn_make_buffer_rsrc()`
- Buffer loads: `__builtin_amdgcn_raw_buffer_load_b128` / `_b64`
- V repack selectors: `0x01000504` / `0x03020706`
- P truncation selector: `0x07060302`
- NaN guard in `softmax_exp` for fully-masked rows
- `__forceinline__` over `__attribute__((always_inline))`
- `__builtin_amdgcn_workitem_id_x()` over `threadIdx.x`

## Open Research Items

These are resolved by Phase 0 (CK Layout Understanding) before any
standalone kernel implementation:

| Item | Needed By | Resolved By |
|------|-----------|-------------|
| P shuffle distribution | Step 6 (`test_pv_gemm`) | Phase 0: dump P_shuffled tensor from CK |
| V shuffle distribution | Step 5 (`test_v_lds`) | Phase 0: dump V_shuffled tensor from CK |
| O shuffle distribution | Step 7 (`test_epilog`) | Phase 0: dump O_shuffled tensor from CK |
| Q load distribution | Step 2 (`test_qk_gemm`) | Phase 0: dump Q_tile tensor from CK |
| V transposed LDS layout formula | Step 5 (`test_v_lds`) | Phase 0: dump LDS after V write |
| Epilog cross-lane exchange pattern | Step 7 (`test_epilog`) | Phase 0: dump O_shuffled pre-store |

## Constraints

- Target: gfx942 (MI-300X, CDNA3)
- Compiler: hipcc (ROCm 6.x)
- Final kernel must match CK register layout: VGPR ≤ 127,
  accum_offset = 128, AGPR = 0, no spills, LDS = 13824
- Standalone kernels have no register pressure requirement
- bf16 inputs/outputs, fp32 internal accumulation
- 4 warps (256 threads), tile M=128, N=64, K=64 (head dim)

## Evaluation Criteria

### Gate 1: Phase 0 — Layout Verification *(blocks Phase 1)*

Per tensor: verification kernel dump matches CK golden dump **bit-exact**.

All 6 tensor layouts (K LDS, Q, S_acc, P_shuffled, V LDS, O_shuffled)
must be verified before any Phase 1 kernel is written.

**Failure → escalate to user.** Cannot proceed without verified layouts.

### Gate 2: Phase 1 — Standalone Kernel Correctness *(blocks Phase 2)*

Per kernel:
- CPU reference: tolerance (max abs < 1e-3, cosine > 0.99998) for
  compute kernels; byte-exact for data movement kernels
- CK golden data: **bit-exact** (primary gate)

All 7 kernels must pass correctness before Phase 2 begins.

**Failure → debug loop, then escalate.**

### Gate 3: Phase 1 — Standalone Kernel ISA *(does NOT block Phase 2)*

Per kernel: assembly checklist — correct instruction types and counts
(see per-kernel table in Three-Level Validation section). No spills,
no function calls.

This gate runs in parallel with Gate 2. Results are **recorded** by
prof and reported to lead, but a standalone kernel failing its assembly
checklist does NOT block Phase 2. It signals that the final kernel may
need different compiler hints (`sched_barrier`, `launch_bounds`) or
structural changes during Phase 2 integration.

**Failure → prof records the deviation, lead notes it for Phase 2.**

### Gate 4: Integration — Primary Goal *(must achieve)*

- `test_fmha_gpu_ref`: 48/48 PASS
- `test_fmha_fwd_d64`: **50/50 PASS**
- Build succeeds with no warnings in modified files

**This is the primary success criterion.** The team has succeeded if
this gate passes, even if secondary/stretch goals are not yet met.

**Failure → escalate to user.**

### Gate 5: ISA Structure — Secondary Goal *(target)*

Run `/amdgpu-asm-analyzer` on the final integrated kernel. Compare
against CK's structural report (`asm_analysis/scratch/pass1-report.md`).

- VGPR ≤ 127, accum_offset = 128, AGPR = 0, no spills
- LDS = 13824
- 0 function calls (`s_swappc`)
- 32 `v_mfma_f32_32x32x8bf16`
- Hot-path instruction mix matches CK (±10% per category)
- Pipeline stage boundaries (5 barriers at correct positions)
- GEMM read/MFMA interleaving pattern matches CK

**Failure → escalate to user with specific deviations.**

### Gate 6: Performance — Stretch Goal

- All benchmark configs within ±2% TFLOPS of CK baseline
- MFMA utilization ≥ 33% (CK baseline: 34.92%)
- PMC profiling if gap > 2%

**Failure → prof diagnoses stall source, escalate to user.**
