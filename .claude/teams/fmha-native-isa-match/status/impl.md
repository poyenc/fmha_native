# Impl Handoff — fmha-native-isa-match (pause @ 2026-05-29)

## ⚠️ CORRECTION TO LEAD'S MENTAL MODEL
Lead's shutdown msg says "Kernel 2 hadn't started on disk / clean slate / NO files
written yet." **THAT IS STALE — test_qk_gemm IS FULLY WRITTEN, COMPILES, AND PASSES
0/8192 BIT-EXACT vs golden (full+partial).** Files are on disk (verified at pause):
- src/kernels/qk_gemm.hpp (7244 B)
- src/refs/ref_qk_gemm.hpp (1108 B) + ref_qk_gemm.cpp (765 B)
- tests/test_qk_gemm.cpp (6080 B)
- CMakeLists.txt: test_qk_gemm target added (lines 64-67), with -O3.
Successor: do NOT rewrite Kernel 2. It only needs the Lead's authoritative QA.

---

## Current task: Phase 1 Kernel 2 (test_qk_gemm) — DONE, AWAITING QA
- Compiles clean (-O3). Smoke on GPU 1: **0/8192 bit-exact (maxabs=0) vs BOTH
  cpu-ref AND golden**, full (sq64/sk64) + partial (sq17/sk33). Logs:
  /tmp/fmha-native-isa-match/impl/qk_gemm_build_002.txt, qk_gemm_smoke_O3_001.txt.
- **CORRECTED S_acc map IS APPLIED** (not the old swapped 0.2 one). Harness uses:
  m_row=(lane%32)+32*warp ; n_col=(r/8)*16+(lane/32)*8+(r%8). Confirmed 0/8192.
- **Golden S_ACC is RAW unscaled Q·Kᵀ** (NOT 0.125-scaled). CPU ref + golden compare
  both use raw sum_d Q·K. (Golden = 8× the scaled value — I verified by decode.)
- in-build asm after -O3: 16 v_mfma_f32_32x32x8, 8 ds_read_b128, 8 buffer_load..lds,
  0 flat_load, 0 function calls, 0 scratch, 0 ds_bpermute. (8 ds_read_b128 not 16:
  each b128 = 8 contiguous bf16 serving both MFMA passes of a kstep; 2 ntile×4 kstep=8.
  Flag for prof to reconcile vs gate's "16".)
- Design doc (FULL detail): /tmp/fmha-native-isa-match/impl/qk_gemm_design.md

## ★ KEY DELIVERABLE: GEMM0 SwizzleA operand wiring (derived + golden-verified 0/4096)
The accumulator-register-order trap that killed the prior attempt. Solved by MEASURING
the real HW MFMA layout on-GPU, not guessing.
- Bare HW v_mfma_f32_32x32x8bf16 acc output: acc(lane L, reg r) =
  C[m=(r/4)*8+(L/32)*4+(r%4), n=L%32]  (free-dim in GROUPS OF 4).
- Golden S_acc free-dim (n_col) is GROUPS OF 8. A bare MFMA (A=K natural) CANNOT match
  via any per-thread register reorder — per-lane seqk OWNERSHIP differs.
- BRIDGE = SwizzleA on the K(A) operand: each lane reads K seqk row = swz(lane%32),
  where swz swaps bits 2 and 3:  swz(x) = (x & ~0xC) | (((x>>2)&1)<<3) | (((x>>3)&1)<<2)
  (i.e. 4↔8, 5↔9, 6↔10, 7↔11, 20↔24, 21↔25, ...).
- With A=K(seqk=tile*32+swz(lane&31)), B=Q(seqq), 16 MFMA reproduce golden S_acc EXACTLY.
- Forward model + on-HW: 0/4096 and 0/8192. THIS is already coded in qk_gemm.hpp
  (qk_swz() + seqk0/seqk1). Reuse for GEMM1 if/when its C-output needs matching.
- MFMA layout probe saved: /tmp/mfma_probe.hip (host /tmp; container poyenc-fmha:/tmp).
- Sent to research for 0.12 tensor_layouts.md confirmation.

## Done this session
1. **Kernel 1 test_k_lds — QA-PASS, byte-exact.** Files: src/kernels/k_lds.hpp,
   src/refs/ref_k_lds.{hpp,cpp}, tests/test_k_lds.cpp, CMake target (+-O3). Stages K
   DRAM→LDS via __builtin_amdgcn_raw_ptr_buffer_load_lds (the hand-asm form did NOT
   assemble on gfx942 — use the builtin). Golden 4096/4096 full, 2112/2112 partial.
   In knowledge.md as PASS.
2. **Found bug in existing async_copy_k_slice** (src/kernel/fmha_fwd_d64_lds.hpp):
   n_base=warp*4+lane/16 is WRONG (3072/4096 mismatch); correct = (lane>>4)*4+warp
   (0/4096). Dead code now; Phase-2 must apply. In knowledge.md.
3. **Root-caused the -O0 codegen problem.** The Phase-1 test CMake COMPILE_FLAGS I
   first used had no -O → clang defaults to -O0 for HIP device TU → no unroll, LDS
   reads stay flat_load, ockl helpers not inlined, spills. FIX: added -O3 to BOTH
   test_k_lds and test_qk_gemm targets. (NOT an inlining problem — helpers already
   __forceinline__.) Correctness identical before/after; asm clean after.

## Uncommitted state
ALL Phase-1 files are uncommitted working-tree changes (k_lds + qk_gemm + CMake edits).
Both kernels compile and pass. Nothing is half-written or broken.

## Next action for successor
1. Get Lead's authoritative QA on test_qk_gemm (clean rebuild + golden). Should pass.
2. Then Kernel 3 (test_row_max): butterfly ds_bpermute reduction over S_acc. Note the
   S_acc row reduction is over the n_col dim = per-register within a lane PLUS the
   lane/32 pairing (cross-half via ds_bpermute) — see hypothesis_s_acc.md §"Implication".
3. Remember -O3 in CMake for every new Phase-1 test target (else -O0 codegen).

## Open threads
- prof asm gate says "16 ds_read_b128" but minimal correct codegen yields 8 (b128
  covers 2 passes). Reconcile the gate count — not a correctness issue.
- GEMM1 (Kernel 6 pv_gemm) will need the same kind of MFMA-order check; the SwizzleA
  tooling/probe (/tmp/mfma_probe.hip) is reusable.
- partial golden dir is sq17/sk33; full is sq64/sk64 — harness picks --golden dir per
  filter (*Full* → full/, *Partial* → partial/). Don't cross them.
