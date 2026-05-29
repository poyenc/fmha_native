# FMHA Native D64 — Incremental Kernel Validation Plan

> **For agentic workers:** This plan is executed by a hip-kernel-team
> (see `.claude/teams/fmha-native-isa-match/config.md`). Each task is
> assigned to a specific role. Tasks use checkbox (`- [ ]`) syntax for
> tracking. Launch with `/hip-kernel-team load fmha-native-isa-match`.

**Goal:** Match CK's ISA output for the D64 BF16 FMHA forward kernel by
validating each pipeline phase as a standalone kernel before assembling
the final kernel.

**Architecture:** Phase 0 maps CK's distributed tensor layouts and
produces golden reference data. Phase 1 builds 7 standalone kernels
validated against golden data. Phase 2 rewrites the final kernel using
validated inner loops. Phase 3 verifies ISA and performance match CK.

**Spec:** `docs/superpowers/specs/2026-05-29-fmha-native-d64-design.md`

---

## Prerequisites

- [ ] **P.1** *(impl)* Commit current changes on `isa-match-rewrite` as
  checkpoint (549 insertions + NaN fix, 7/50 tests pass)

```bash
git add src/kernel/fmha_fwd_d64_device.hpp \
        src/kernel/fmha_fwd_d64_epilog.hpp \
        src/kernel/fmha_fwd_d64_gemm.hpp \
        src/kernel/fmha_fwd_d64_lds.hpp \
        src/kernel/fmha_fwd_d64_softmax.hpp
git commit -m "checkpoint: NaN fix + unwired ISA helpers (7/50 pass)"
```

---

## Phase 0: CK Layout Understanding

**Roles: research + impl + build + test + debug**

Goal: map every `distributed_tensor` and `shuffle_tile()` in CK's FMHA
pipeline. Verify each layout hypothesis by building a small kernel and
comparing against CK's actual dump. Produce golden reference files and
a confirmed layout map document.

### 0.1 — Trace CK pipeline: identify all distributed tensors

- [ ] **Step 1:** Read CK kernel entry point

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`

Identify: the pipeline class used (likely `block_fmha_pipeline_qr_ks_vs_async`), the epilogue class (likely `cshuffle_epilogue`), and how they're wired together.

- [ ] **Step 2:** Read CK pipeline implementation

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`

List every `auto` variable that holds a tensor, and every `shuffle_tile()` / `store_tile()` / `load_tile()` call site. Document in `/tmp/fmha-native-isa-match/research/pipeline_trace.md`.

- [ ] **Step 3:** Read CK pipeline policy

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp`

Document: tile sizes, LDS layout parameters, MFMA instruction choice, distribution types for Q/K/V/S/P/O.

### 0.2 — Understand tile distribution encoding

CK's tile system uses partition indices to encode how tensor elements
are distributed across threads and registers. Understanding this system
is prerequisite for all subsequent tensor mapping tasks.

- [ ] **Step 1:** Read tile_distribution_encoding

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp`

Document:
- What is a partition dimension? How does it differ from a replicate dimension?
- How does the encoding map a tensor index `[i, j]` to `(thread_id, register_index)`?
- What do the template parameters mean (e.g., `Seqs<...>`, partition lengths)?

Work through one concrete example from the FMHA pipeline policy to
verify understanding.

- [ ] **Step 2:** Read tile_distribution

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/core/tensor/tile_distribution.hpp`

Document:
- How does `tile_distribution` use the encoding to compute thread-to-element mappings?
- How is the partition index computed from `threadIdx.x`? (lane_id, warp_id decomposition)
- How does the partition index translate to DRAM/LDS address offsets during `load_tile()` and `store_tile()`?

- [ ] **Step 3:** Read static_distributed_tensor

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp`

Document:
- How does the register layout (which VGPR holds which element) follow from the distribution encoding?
- Given a distribution type, how to compute: total VGPRs per thread, and the mapping `reg[i] → tensor[m, n]`?

- [ ] **Step 4:** Read FMHA pipeline policy distributions

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp`

Extract the concrete distribution types used for Q, K, V, S, P, O in
the D64 bf16 configuration. For each, write out:
- The partition dimension lengths
- The thread decomposition (which bits of threadIdx.x map to which partition)
- The register index formula

Save to `/tmp/fmha-native-isa-match/research/distribution_types.md`

- [ ] **Step 5:** Verify understanding with a worked example

Pick one distribution (e.g., the GEMM0 C-output / S_acc distribution).
Manually compute: for thread 0, 1, 31, 32, 33, 63 — which tensor
elements `[m, n]` are in registers 0..15? Compare against the MFMA ISA
spec's lane mapping. They must agree.

### 0.3 — Understand shuffle_tile implementation

- [ ] **Step 1:** Read shuffle_tile core

Read: `~/workspace/repo/rocm-libraries/projects/composablekernel/include/ck_tile/core/tensor/shuffle_tile.hpp`

Document: how it transforms input distribution → output distribution.
What parameters control the shuffle? How does it use the partition index
difference between source and target distributions to determine data
movement (LDS write/read, bpermute, or perm)?

- [ ] **Step 2:** Trace one shuffle end-to-end

Pick the P shuffle (softmax output → GEMM1 B operand). Using the source
and target distribution types from 0.2 Step 4, trace the shuffle:
- Which elements move between threads?
- Which elements stay in the same thread but change register index?
- What LDS intermediate layout does the shuffle use (if any)?
- What instructions does this lower to? (`ds_write` + `ds_read`, or `ds_bpermute`, or `v_perm`?)

### 0.4 — Write CK instrumented dump kernel *(research)*

Produces golden files that all subsequent hypothesis verification
compares against. Must be done before any tensor mapping.

- [ ] **Step 1:** Write dump utility header

Create `tools/ck_instrumented/dump_utils.hpp`:
- `dump_tensor_to_dram()`: iterates tile elements per thread, writes to
  DRAM buffer. Uses CK's tile iteration API.
- `dump_lds_to_dram()`: threads cooperatively copy LDS to DRAM.

- [ ] **Step 2:** Write instrumented FMHA kernel

Create `tools/ck_instrumented/fmha_fwd_dump.hpp`:
- Copy CK's pipeline `operator()` function
- Insert `dump_tensor_to_dram()` at 8 phase boundaries:
  K_lds, S_acc, rmax, P, P_shuffled, V_lds, O_acc, O_shuffled
- Extra kernel argument: `float* debug_buf`

- [ ] **Step 3:** Write dump runner

Create `tools/ck_instrumented/dump_runner.cpp`:
- Allocate Q, K, V with known sequential values
- Launch instrumented kernel
- Save each dump section as binary file

- [ ] **Step 4:** Write CMakeLists for dump tool

### 0.5 — Build and run CK dumps *(build, test)*

- [ ] **Step 1** *(build)*: Build dump tool in poyenc-ck

```bash
docker exec poyenc-ck bash -c "cd /root/workspace && <build command> 2>&1"
```

Save output to `/tmp/fmha-native-isa-match/build/ck_dump_build_001.txt`

- [ ] **Step 2** *(build)*: If build fails → message research with error

- [ ] **Step 3** *(test)*: Run dump on full-tile case (b=1, h=1, sq=64, sk=64, d=64)

```bash
docker exec poyenc-ck bash -c "<dump runner> --b=1 --h=1 --sq=64 --sk=64 --d=64 --out=/tmp/golden_full/ 2>&1"
docker cp poyenc-ck:/tmp/golden_full/ /tmp/fmha-native-isa-match/golden/full/
```

- [ ] **Step 4** *(test)*: Run dump on partial-tile case (b=1, h=1, sq=17, sk=33, d=64)

```bash
docker exec poyenc-ck bash -c "<dump runner> --b=1 --h=1 --sq=17 --sk=33 --d=64 --out=/tmp/golden_partial/ 2>&1"
docker cp poyenc-ck:/tmp/golden_partial/ /tmp/fmha-native-isa-match/golden/partial/
```

- [ ] **Step 5** *(test)*: Run CK end-to-end on same inputs, save final O for chain validation

- [ ] **Step 6** *(test)*: Verify instrumented kernel's epilog dump matches CK's actual O output. If mismatch → message debug.

### 0.6 — Verify & map K LDS layout

**Hypothesis → verification kernel → golden comparison**

- [ ] **Step 1** *(research)*: Read CK async K copy source, write hypothesis

Document in `/tmp/fmha-native-isa-match/research/hypothesis_k_lds.md`:
- m0 setup formula, LDS addressing, padded layout
- Expected: `LDS[buf_base + (row * 72 + col) * 2] = K[kv_offset + row, col]`

- [ ] **Step 2** *(impl)*: Write verification kernel `tools/verify/verify_k_lds.hip`

Tiny `__global__` kernel:
- Setup m0 and issue `buffer_load_dword` with lds flag per hypothesis
- `__syncthreads()`, dump LDS to DRAM

- [ ] **Step 3** *(impl)*: Write test harness `tools/verify/test_verify_k_lds.cpp`

- [ ] **Step 4** *(build)*: Build verification kernel

- [ ] **Step 5** *(test)*: Run, compare LDS dump against golden K_lds file — byte-exact

- [ ] **Step 6** *(debug)*: If mismatch → investigate m0 formula, report to research

- [ ] **Step 7** *(research)*: If confirmed → mark K LDS layout as **verified** in tensor_layouts.md

### 0.7 — Verify & map Q load distribution

- [ ] **Step 1** *(research)*: Read CK Q load, write hypothesis

Document in `/tmp/fmha-native-isa-match/research/hypothesis_q_load.md`:
- Q distribution type from policy
- Per-thread register mapping: `thread[tid].reg[i] = Q[m_row, k_col]`
- Is load k_sub-dependent (interleaved)?

- [ ] **Step 2** *(impl)*: Write verification kernel `tools/verify/verify_q_load.hip`

Load Q per hypothesis, dump register map `float[256][32]` to DRAM.

- [ ] **Step 3** *(build)*: Build

- [ ] **Step 4** *(test)*: Compare against golden S_acc input (Q registers should match Q portion of GEMM0 input). Use known Q values to verify which register holds which element.

- [ ] **Step 5** *(debug)*: If mismatch → investigate

- [ ] **Step 6** *(research)*: If confirmed → mark Q distribution as **verified**

### 0.8 — Verify & map S accumulator distribution (MFMA C-output)

- [ ] **Step 1** *(research)*: Write hypothesis based on ISA spec

Document in `/tmp/fmha-native-isa-match/research/hypothesis_s_acc.md`:
- `S_acc[i]` at lane `l` = `S[warp*32 + k_sub*16 + i, l%32]`

- [ ] **Step 2** *(impl)*: Write verification kernel `tools/verify/verify_s_acc.hip`

Pre-populate K in LDS (using verified K layout from 0.6), load Q (using
verified Q pattern from 0.7), issue 16 MFMAs, dump S_acc to DRAM.

- [ ] **Step 3** *(build)*: Build

- [ ] **Step 4** *(test)*: Compare against golden S_acc — bit-exact

- [ ] **Step 5** *(debug)*: If mismatch → investigate MFMA A/B operand assignment

- [ ] **Step 6** *(research)*: If confirmed → mark S_acc as **verified**

### 0.9 — Verify & map P shuffle distribution (CRITICAL)

- [ ] **Step 1** *(research)*: Read CK P shuffle, write hypothesis

Document in `/tmp/fmha-native-isa-match/research/hypothesis_p_shuffle.md`:
- Source distribution (same as S_acc)
- Target distribution (GEMM1 B-operand format)
- Per-thread register mapping after shuffle
- How 4 bf16 values pack into `short4`

- [ ] **Step 2** *(impl)*: Write verification kernel `tools/verify/verify_p_shuffle.hip`

Load P in S_acc layout (verified from 0.8), apply hypothesized shuffle,
dump P_shuffled registers to DRAM.

- [ ] **Step 3** *(build)*: Build

- [ ] **Step 4** *(test)*: Compare against golden P_shuffled — bit-exact

- [ ] **Step 5** *(debug)*: If mismatch → investigate shuffle implementation

- [ ] **Step 6** *(research)*: If confirmed → mark P_shuffled as **verified**

### 0.10 — Verify & map V load + shuffle + LDS distribution (CRITICAL)

- [ ] **Step 1** *(research)*: Read CK V path, write hypothesis

Document in `/tmp/fmha-native-isa-match/research/hypothesis_v_lds.md`:
- V DRAM load pattern (buffer_load_dwordx2)
- v_perm_b32 repack selectors and what they do
- ds_write2_b32 transposed LDS addressing formula
- Final LDS layout: `LDS[addr] = V[seqk, hdim]` → what addr?

- [ ] **Step 2** *(impl)*: Write verification kernel `tools/verify/verify_v_lds.hip`

Load V from DRAM, apply repack + transposed write per hypothesis,
`__syncthreads()`, dump LDS to DRAM.

- [ ] **Step 3** *(build)*: Build

- [ ] **Step 4** *(test)*: Compare LDS dump against golden V_lds — byte-exact

- [ ] **Step 5** *(debug)*: If mismatch → investigate v_perm selector or ds_write2 offset

- [ ] **Step 6** *(research)*: If confirmed → mark V LDS as **verified**

### 0.11 — Verify & map O accumulator + O shuffle distribution (CRITICAL)

- [ ] **Step 1** *(research)*: Read CK cshuffle epilogue, write hypothesis

Document in `/tmp/fmha-native-isa-match/research/hypothesis_o_shuffle.md`:
- O_acc distribution (same as S_acc, MFMA C-output)
- cshuffle transformation: which lanes exchange, what register layout results
- How post-shuffle layout enables buffer_store_dwordx2 of 4 consecutive hdim values

- [ ] **Step 2** *(impl)*: Write verification kernel `tools/verify/verify_o_shuffle.hip`

Load O_acc in MFMA C-output layout (verified from 0.8 pattern), apply
hypothesized O shuffle, dump O_shuffled registers to DRAM.

- [ ] **Step 3** *(build)*: Build

- [ ] **Step 4** *(test)*: Compare against golden O_shuffled — bit-exact

- [ ] **Step 5** *(debug)*: If mismatch → investigate cshuffle implementation

- [ ] **Step 6** *(research)*: If confirmed → mark O_shuffled as **verified**

### 0.12 — Write confirmed tensor layout map *(research)*

Only **verified** layouts (backed by bit-exact golden match) go in this
document. Unverified hypotheses stay in hypothesis files.

- [ ] **Step 1:** Compile all verified findings into `docs/ck-tensor-layout-map.md`

Contents:
- Table of all distributed tensors with **verified** thread/register
  mappings (each entry cites which golden file confirmed it)
- shuffle_tile transformation specs for P, V, O
- LDS layout formulas for K and V
- Per-phase register count (VGPRs per thread)
- Golden file inventory with paths
- Verification kernel inventory with paths

- [ ] **Step 2:** Message lead: Phase 0 complete, all layouts verified

---

## Phase 1: Standalone Kernels

7 kernels, each following the pipeline: impl → build → test → prof.
Dependencies: Phase 0 must be complete (golden files available).

### Kernel 1: test_k_lds

#### 1.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_k_lds.hpp`

Declare: `void ref_k_lds(const __hip_bfloat16* K, int stride_k, int kv_offset, int seqlen_k, uint8_t* expected_lds, int buf0_offset, int buf1_offset);`

- [ ] **Step 2:** Create `src/refs/ref_k_lds.cpp`

Implement: for each K row in [kv_offset, kv_offset+64), for each K col in [0, 64), compute the LDS byte address using padded layout `(row * 72 + col) * 2`, and write the bf16 value. Split cols 0..31 into buf0, cols 32..63 into buf1.

#### 1.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/k_lds.hpp`

Implement `__global__ void test_k_lds_kernel(...)`:
- Setup m0 for async copy addressing
- Issue `buffer_load_dword` with lds flag (inline asm) for each K element
- `__syncthreads()`
- Cooperatively copy LDS contents to DRAM output buffer

#### 1.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_k_lds.cpp`

GTest binary:
- Allocate K matrix on host with known values (sequential bf16)
- Copy to device
- Launch `test_k_lds_kernel`
- Copy LDS dump back to host
- Run `ref_k_lds` to get expected LDS image
- Compare byte-exact
- If `--golden` flag: load golden file, compare byte-exact

- [ ] **Step 2:** Full tile test case (seqlen_k=64)
- [ ] **Step 3:** Partial tile test case (seqlen_k=33)

#### 1.4 — Add to CMakeLists *(impl)*

- [ ] **Step 1:** Add `test_k_lds` target

```cmake
add_executable(test_k_lds tests/test_k_lds.cpp src/refs/ref_k_lds.cpp)
target_link_libraries(test_k_lds PRIVATE fmha_runner GTest::gtest_main)
set_source_files_properties(tests/test_k_lds.cpp PROPERTIES LANGUAGE HIP
    COMPILE_FLAGS "--offload-arch=${GPU_TARGET} --save-temps")
```

- [ ] **Step 2:** Message build: ready to compile test_k_lds

#### 1.5 — Build *(build)*

- [ ] **Step 1:** Build test_k_lds

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace/build && cmake .. -GNinja && ninja test_k_lds 2>&1"
```

Save output to `/tmp/fmha-native-isa-match/build/build_k_lds_001.txt`

- [ ] **Step 2:** If build fails → message impl with error
- [ ] **Step 3:** If build succeeds → extract assembly

```bash
docker cp poyenc-fmha:/root/workspace/build/tests/test_k_lds-hip-amdgcn-amd-amdhsa-gfx942.s /tmp/fmha-native-isa-match/build/k_lds.s
```

- [ ] **Step 4:** Message test: binary ready

#### 1.6 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref test (full tile)

```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_k_lds --gtest_filter='*FullTile*' 2>&1"
```

Save output to `/tmp/fmha-native-isa-match/test/k_lds_cpu_full_001.txt`

- [ ] **Step 2:** Run CPU ref test (partial tile)

```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_k_lds --gtest_filter='*PartialTile*' 2>&1"
```

- [ ] **Step 3:** Run golden data test (full tile)

```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_k_lds --golden=/tmp/golden/full/ --gtest_filter='*FullTile*' 2>&1"
```

- [ ] **Step 4:** Run golden data test (partial tile)
- [ ] **Step 5:** If any fail → message debug with failure output
- [ ] **Step 6:** If all pass → message prof: ready for assembly check

#### 1.7 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Read test output, kernel source, CPU ref
- [ ] **Step 2:** Identify root cause (wrong m0 calc, wrong LDS offset, wrong buf assignment)
- [ ] **Step 3:** Message impl with specific fix (file, line, what to change)

#### 1.8 — Assembly analysis *(prof)*

- [ ] **Step 1:** Analyze `k_lds.s` via Explore subagent

Check: contains `buffer_load_dword` with lds flag, does NOT contain `buffer_load_b128` or `scratch_*`. Count async loads matches expected (8 per buffer).

- [ ] **Step 2:** Message lead: step 1 assembly gate pass/fail

---

### Kernel 2: test_qk_gemm

#### 2.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_qk_gemm.hpp` / `.cpp`

Implement: given Q matrix and K LDS byte image, compute S using the MFMA lane mapping. For each thread (0..255), for each register (0..15), for each sub-tile (n0, n1): compute `S[m_row, n_col]` where `m_row = warp*32 + k_sub*16 + i`, `n_col = lane%32 (+ 32 for n1)`. Accumulate as `sum(Q[m_row, k] * K_lds[n_col, k])` for k=0..63.

Output: `float[256][32]`

#### 2.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/qk_gemm.hpp`

Implement `__global__ void test_qk_gemm_kernel(...)`:
- Load Q via `buffer_load_dwordx4` (use pattern from Phase 0 Q distribution)
- Pre-populate K in LDS (use validated K LDS layout from step 1)
- Issue 16 MFMA: `__builtin_amdgcn_mfma_f32_32x32x8bf16_1k`
- Dump S_acc registers to DRAM: `out[threadIdx.x * 32 + i] = s_acc_n0[i]`, `out[threadIdx.x * 32 + 16 + i] = s_acc_n1[i]`

#### 2.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_qk_gemm.cpp`
- [ ] **Step 2:** Add to CMakeLists.txt
- [ ] **Step 3:** Message build

#### 2.4 — Build *(build)*

- [ ] **Step 1:** Build test_qk_gemm, extract assembly
- [ ] **Step 2:** Report result → message test or impl

#### 2.5 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref (full + partial) — tolerance: max abs < 1e-3
- [ ] **Step 2:** Run golden data (full + partial) — bit-exact
- [ ] **Step 3:** Report → message prof or debug

#### 2.6 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Investigate: wrong MFMA A/B operand? wrong Q load pattern? wrong K LDS read offset?
- [ ] **Step 2:** Message impl with fix

#### 2.7 — Assembly analysis *(prof)*

- [ ] **Step 1:** Check: 16 `v_mfma_f32_32x32x8bf16`, 16 `ds_read_b128`, no `ds_bpermute`, no `scratch_*`
- [ ] **Step 2:** Message lead: gate result

---

### Kernel 3: test_row_max

#### 3.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_row_max.hpp` / `.cpp`

Implement: simulate butterfly reduction. For each thread, compute `row_max[i]` by taking max of `s_acc_n0[i]` and `s_acc_n1[i]`, then simulating 5 XOR reduction rounds across 32 lanes sharing the same k_sub.

Output: `float[256][16]`

#### 3.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/row_max.hpp`

Implement: load S register map from DRAM, run butterfly `ds_bpermute_b32` reduction, dump rmax to DRAM.

#### 3.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_row_max.cpp`, add to CMakeLists
- [ ] **Step 2:** Message build

#### 3.4 — Build *(build)*

- [ ] **Step 1:** Build, extract assembly, report

#### 3.5 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref — exact match (scalar ops, no FMA)
- [ ] **Step 2:** Run golden data — bit-exact
- [ ] **Step 3:** Report

#### 3.6 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Investigate butterfly lane indexing
- [ ] **Step 2:** Message impl

#### 3.7 — Assembly analysis *(prof)*

- [ ] **Step 1:** Check: 80 `ds_bpermute_b32`, no `v_mfma`, no `scratch_*`
- [ ] **Step 2:** Message lead

---

### Kernel 4: test_softmax

#### 4.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_softmax.hpp` / `.cpp`

Implement 2-tile mini-loop:
- Iter 0: exp2f(S - max) with -INF guard, row_sum reduction, store P/rsum
- Iter 1: rescale o_acc via exp2f(old_max - new_max), recompute max/exp/sum, update rsum

Output per iteration: `P float[256][32]`, `rsum float[256][16]`

#### 4.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/softmax.hpp`

Implement: load 2 S register maps from DRAM, run 2-iteration softmax loop (reusing validated row_max butterfly from step 3), dump P + rsum per iteration.

#### 4.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_softmax.cpp`, add to CMakeLists
- [ ] **Step 2:** Message build

#### 4.4 — Build *(build)*

- [ ] **Step 1:** Build, extract assembly, report

#### 4.5 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref — tolerance (exp2f may differ slightly)
- [ ] **Step 2:** Run golden data — bit-exact
- [ ] **Step 3:** Report

#### 4.6 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Investigate: NaN from -INF guard? wrong rescaling? wrong butterfly?
- [ ] **Step 2:** Message impl

#### 4.7 — Assembly analysis *(prof)*

- [ ] **Step 1:** Check: 32 `v_exp_f32`, no `scratch_*`
- [ ] **Step 2:** Message lead

---

### Kernel 5: test_v_lds

#### 5.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_v_lds.hpp` / `.cpp`

Implement: using the V transposed LDS layout from Phase 0 tensor map (`docs/ck-tensor-layout-map.md`), compute the expected LDS byte image after V load + repack + transposed write. This is the layout that was unknown before Phase 0.

#### 5.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/v_lds.hpp`

Implement:
- `buffer_load_dwordx2` to load V from DRAM
- `v_perm_b32` with selectors `0x01000504` / `0x03020706` for bf16 repack
- `ds_write2_b32` to write transposed V to LDS
- `__syncthreads()` then dump LDS to DRAM

#### 5.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_v_lds.cpp`, add to CMakeLists
- [ ] **Step 2:** Message build

#### 5.4 — Build *(build)*

- [ ] **Step 1:** Build, extract assembly, report

#### 5.5 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref — byte-exact
- [ ] **Step 2:** Run golden data — byte-exact
- [ ] **Step 3:** Report

#### 5.6 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Investigate: wrong v_perm selector? wrong ds_write2 offset? wrong transposition formula?
- [ ] **Step 2:** Message impl

#### 5.7 — Assembly analysis *(prof)*

- [ ] **Step 1:** Check: `buffer_load_dwordx2`, `v_perm_b32`, `ds_write2_b32`, no `scratch_*`
- [ ] **Step 2:** Message lead

---

### Kernel 6: test_pv_gemm

#### 6.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_pv_gemm.hpp` / `.cpp`

Implement: given P (in P_shuffled distribution from Phase 0) and V LDS image (transposed layout from step 5), compute O_acc using MFMA lane mapping. P is first truncated to bf16 via `v_perm_b32` with selector `0x07060302`.

Output: `float[256][32]`

#### 6.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/pv_gemm.hpp`

Implement:
- Load P from DRAM (in P_shuffled register layout)
- Truncate P to bf16 via `v_perm_b32` (selector `0x07060302`)
- Pre-populate V in LDS (using validated V LDS layout from step 5)
- Issue 16 MFMA
- Dump O_acc registers to DRAM

#### 6.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_pv_gemm.cpp`, add to CMakeLists
- [ ] **Step 2:** Message build

#### 6.4 — Build *(build)*

- [ ] **Step 1:** Build, extract assembly, report

#### 6.5 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref — tolerance
- [ ] **Step 2:** Run golden data — bit-exact
- [ ] **Step 3:** Report

#### 6.6 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Investigate: wrong P packing order? wrong V LDS read? wrong MFMA A/B?
- [ ] **Step 2:** Message impl

#### 6.7 — Assembly analysis *(prof)*

- [ ] **Step 1:** Check: 16 `v_mfma`, 16 `ds_read_b128`, `v_perm_b32`, no `ds_bpermute`, no `scratch_*`
- [ ] **Step 2:** Message lead

---

### Kernel 7: test_epilog

#### 7.1 — Write CPU reference *(impl)*

- [ ] **Step 1:** Create `src/refs/ref_epilog.hpp` / `.cpp`

Implement: using the O_shuffled distribution from Phase 0, normalize O_acc by rsum, apply the cross-lane exchange pattern (documented in tensor layout map), pack to bf16 via `v_perm_b32` (selector `0x07060302`), and produce the expected DRAM output in `buffer_store_dwordx2` layout.

#### 7.2 — Write GPU kernel *(impl)*

- [ ] **Step 1:** Create `src/kernels/epilog.hpp`

Implement:
- Load O_acc and rsum from DRAM
- Normalize: `o[i] *= 1.0f / rsum[i]`
- Apply shuffle (cross-lane exchange matching CK's O_shuffled distribution)
- Pack to bf16 via `v_perm_b32`
- Store via `buffer_store_dwordx2`

#### 7.3 — Write test harness *(impl)*

- [ ] **Step 1:** Create `tests/test_epilog.cpp`, add to CMakeLists
- [ ] **Step 2:** Message build

#### 7.4 — Build *(build)*

- [ ] **Step 1:** Build, extract assembly, report

#### 7.5 — Run tests *(test)*

- [ ] **Step 1:** Run CPU ref — tolerance
- [ ] **Step 2:** Run golden data — bit-exact
- [ ] **Step 3:** Report

#### 7.6 — Debug (on failure) *(debug)*

- [ ] **Step 1:** Investigate: wrong cross-lane exchange? wrong bf16 packing? wrong buffer_store voffset?
- [ ] **Step 2:** Message impl

#### 7.7 — Assembly analysis *(prof)*

- [ ] **Step 1:** Check: 8 `buffer_store_dwordx2`, `v_perm_b32`, no `scratch_*`
- [ ] **Step 2:** Message lead

---

## Phase 2: Final Kernel Rewrite

Dependencies: all 7 standalone kernels pass all gates.

### 8.1 — Rewrite device.hpp: K LDS + QK GEMM only *(impl)*

- [ ] **Step 1:** Copy validated K async copy inner loop from `kernels/k_lds.hpp` into new `fmha_fwd_d64_device.hpp`
- [ ] **Step 2:** Copy validated QK GEMM MFMA loop from `kernels/qk_gemm.hpp`
- [ ] **Step 3:** Add Q load using validated pattern
- [ ] **Step 4:** Wire: K copy → barrier → QK GEMM → write S to O (temporary, for testing)
- [ ] **Step 5:** Message build

### 8.2 — Build and test K+QK phase *(build, test)*

- [ ] **Step 1** *(build)*: Build test_fmha_fwd_d64
- [ ] **Step 2** *(test)*: Run test_fmha_fwd_d64 — report how many pass
- [ ] **Step 3** *(test)*: If regression from 7/50 → message debug

### 8.3 — Add softmax phase *(impl)*

- [ ] **Step 1:** Copy validated row_max butterfly from `kernels/row_max.hpp`
- [ ] **Step 2:** Copy validated softmax exp/sum/rescale from `kernels/softmax.hpp`
- [ ] **Step 3:** Wire into KV tile loop with online softmax state
- [ ] **Step 4:** Message build

### 8.4 — Build and test K+QK+softmax *(build, test)*

- [ ] **Step 1** *(build)*: Build
- [ ] **Step 2** *(test)*: Run tests, report

### 8.5 — Add V LDS + PV GEMM phase *(impl)*

- [ ] **Step 1:** Copy validated V load/repack/LDS write from `kernels/v_lds.hpp`
- [ ] **Step 2:** Copy validated PV GEMM MFMA loop from `kernels/pv_gemm.hpp`
- [ ] **Step 3:** Copy validated P shuffle/truncation pattern
- [ ] **Step 4:** Wire: V load → barrier → PV GEMM → accumulate O
- [ ] **Step 5:** Message build

### 8.6 — Build and test K+QK+softmax+PV *(build, test)*

- [ ] **Step 1** *(build)*: Build
- [ ] **Step 2** *(test)*: Run tests, report

### 8.7 — Add epilog phase *(impl)*

- [ ] **Step 1:** Copy validated O shuffle + bf16 pack + buffer_store from `kernels/epilog.hpp`
- [ ] **Step 2:** Wire: normalize O by rsum → shuffle → pack → store
- [ ] **Step 3:** Message build

### 8.8 — Build and test full kernel *(build, test)*

- [ ] **Step 1** *(build)*: Build test_fmha_fwd_d64
- [ ] **Step 2** *(test)*: Run test_fmha_fwd_d64 — target: 50/50 PASS
- [ ] **Step 3** *(test)*: Run test_fmha_gpu_ref — must remain 48/48 PASS
- [ ] **Step 4** *(test)*: If not 50/50 → message debug
- [ ] **Step 5** *(test)*: If 50/50 → message lead: Phase 2 integration gate PASS

### 8.9 — Register pressure tuning *(impl)*

- [ ] **Step 1:** Add `launch_bounds(256, 3)` to kernel entry
- [ ] **Step 2:** Add `sched_barrier(0)` at phase boundaries
- [ ] **Step 3:** Message build for rebuild + assembly extraction

### 8.10 — Verify no regression after tuning *(build, test)*

- [ ] **Step 1** *(build)*: Build, extract assembly
- [ ] **Step 2** *(test)*: Run test_fmha_fwd_d64 — must remain 50/50
- [ ] **Step 3** *(test)*: Report

---

## Phase 3: ISA Quality Gate + Performance

Dependencies: Phase 2 complete (50/50 tests pass).

### 9.1 — Full assembly analysis *(prof)*

- [ ] **Step 1:** Extract final kernel assembly

```bash
docker cp poyenc-fmha:/root/workspace/build/fmha_fwd_d64_kernel-hip-amdgcn-amd-amdhsa-gfx942.s /home/poyenc/workspace/repo/fmha_native/native_d64_kernel.s
```

- [ ] **Step 2:** Run `/amdgpu-asm-analyzer native_d64_kernel.s`
- [ ] **Step 3:** Compare against CK structural report
- [ ] **Step 4:** Report to lead:
  - VGPR count (target: ≤ 127)
  - AGPR count (target: 0)
  - Spill count (target: 0)
  - LDS size (target: 13824)
  - Function calls (target: 0)
  - MFMA count (target: 32)
  - Instruction mix comparison table

### 9.2 — Benchmark sweep *(prof)*

- [ ] **Step 1:** Run native kernel benchmark

```bash
docker exec poyenc-fmha bash -c "cd /root/workspace && bash scripts/run-benchmark.sh --d64 2>&1"
```

- [ ] **Step 2:** Run CK baseline benchmark

```bash
docker exec poyenc-ck bash -c "cd /root/workspace && bash example/ck_tile/01_fmha/script/benchmark_fwd_sp3_compare.sh --d64 2>&1"
```

- [ ] **Step 3:** Compare: all configs within ±2% TFLOPS of CK
- [ ] **Step 4:** Report to lead

### 9.3 — PMC profiling (if benchmark gap > 2%) *(prof)*

- [ ] **Step 1:** Run rocprof-compute on native kernel
- [ ] **Step 2:** Run rocprof-compute on CK kernel
- [ ] **Step 3:** Compare MFMA utilization (target: ≥ 33%)
- [ ] **Step 4:** Identify top stall source if underperforming
- [ ] **Step 5:** Report to lead

### 9.4 — Final gate report *(lead)*

- [ ] **Step 1:** Compile all gate results:
  - Integration gate: 50/50 tests
  - ISA gate: VGPR, spills, instruction mix
  - Performance gate: benchmark comparison
- [ ] **Step 2:** Report to user: all gates pass or list remaining gaps

---

## Task Summary

| Phase | Role | Task Count |
|-------|------|:----------:|
| Prerequisites | impl | 1 |
| Phase 0 | research | ~25 (hypotheses + CK dump + layout map) |
| Phase 0 | impl | ~12 (6 verification kernels + harnesses) |
| Phase 0 | build | ~8 (CK dump + 6 verify kernels) |
| Phase 0 | test | ~14 (CK dump runs + 6 verify comparisons) |
| Phase 0 | debug | ~6 (on-demand, per verification mismatch) |
| Phase 1 (×7 kernels) | impl | 28 |
| Phase 1 (×7 kernels) | build | 14 |
| Phase 1 (×7 kernels) | test | 14 |
| Phase 1 (×7 kernels) | debug | 14 (on-demand) |
| Phase 1 (×7 kernels) | prof | 14 |
| Phase 2 | impl | 12 |
| Phase 2 | build | 5 |
| Phase 2 | test | 8 |
| Phase 3 | prof | 12 |
| Phase 3 | lead | 2 |
| **Total** | | **~180** |

## Dependency Graph

```
P.1 (commit checkpoint)
  └→ 0.1-0.3 (research: read CK source, understand distributions)
       └→ 0.4-0.5 (research+build+test: CK instrumented dump → golden files)
            ├→ 0.6 (verify K LDS: research→impl→build→test→confirm)
            ├→ 0.7 (verify Q load: research→impl→build→test→confirm)
            │    └→ 0.8 (verify S_acc: needs verified K + Q)
            │         ├→ 0.9  (verify P shuffle: needs verified S_acc)
            │         └→ 0.11 (verify O shuffle: needs verified S_acc pattern)
            └→ 0.10 (verify V LDS: research→impl→build→test→confirm)
                 └→ [0.9 + 0.10 both done]
                      └→ 0.12 (write confirmed layout map)
                           └→ Phase 1 Kernels (sequential: 1→2→3→4→5→6→7)
                                └→ Phase 2 (final kernel rewrite)
                                     └→ Phase 3 (ISA + performance)
```

Phase 0 verification kernels 0.6 (K LDS) and 0.10 (V LDS) can run in
parallel since they're independent. 0.7 (Q) can also run in parallel
with 0.6/0.10. But 0.8 (S_acc) depends on both 0.6 and 0.7.

Phase 1 kernels run sequentially — impl works on one at a time to avoid
context-switching.
