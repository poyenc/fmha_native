# Handoff: lead — 2026-05-29

## Why paused
Clean pause at the lead 40%-context checkpoint (user rule). User chose "pause team cleanly." No half-finished lead work. Resume: `/hip-kernel-team load fmha-native-isa-match`.

## State summary
Phase 0 (CK layout understanding) ~70% done. 4 of ~6 layouts GOLDEN-VERIFIED. Phase 1 (native kernels) started: Kernel 1 QA-PASS.

Authoritative knowledge: recall knowledge.md (18+ entries, confidence-labeled) at
~/.local/share/claude/recall/fmha_native/branches/isa-match-rewrite/tasks/isa-match-rewrite/
Recall status.md has the progress checklist. Read those FIRST on resume.

## GOLDEN-VERIFIED layouts (bit/byte-exact vs golden — safe to build on)
- K LDS (0.6): offset_elems(j,d)=(j%4)*136+((j/4)%4)*32+(j/16)*544+(d%32)+(d/32)*2304
- Q load (0.7): m_row=(lane%32)+32*warp, k_headdim=(r/8)*16+(lane/32)*8+(r%8). [golden corrected source: groups of 8 not 4]
- V LDS (0.10): 3-stage (buffer_load_dwordx2 → intra-thread v_perm 4×2→2×4 → ds_write2_b32); offset=base+(k/8)*576+(d/8)*72+(d%8)*8+(k%8), k=n%32
- S_acc (0.8): m=(lane%32)+32*warp, n=(r/8)*16+(lane/32)*8+(r%8) [TransposedC; golden corrected 0.2's bare-HW formula]. S/P/O share this dist.

## Other key facts (asm-confirmed)
- Epilogue = Default2D (NO O shuffle, NO cshuffle). P = register-resident (NO P shuffle). Only V is shuffled.
- ISA-match target (Phase 3 gate): 32 mfma, 8 buffer_store_dwordx2, 16 ds_read_b128, 4 ds_write2_b32, 2 ds_bpermute, 44 v_perm; VGPR 127, AGPR 0, 0 spill, LDS 13824. Build recipe + load-bearing flags in knowledge.md.
- BUG found (Phase 2 fix): existing async_copy_k_slice uses warp*4+L/16 (wrong, 3072/4096); correct = (L>>4)*4+warp. Correct version already in src/kernels/k_lds.hpp.

## Golden data
- Generated + validated (instr O == plain CK O bit-identical). 8 slots: 0=K_LDS,1=S_ACC,2=RMAX,3=P,4=V_LDS,5=O_ACC,6=O_FINAL,7=Q.
- GOTCHA: containers DON'T share /tmp. Golden lives in poyenc-ck:/tmp, staged (docker cp) to host + poyenc-fmha:/tmp/.../golden/. Any new artifact one stage consumes elsewhere must be staged via host. dump_lds md5 15675d..., dump_reg (8-slot) 568c08....
- Decode dirs: decode_{kj,kd,qm,qd,vj,vd,sm,sn} in poyenc-ck:/tmp/.../golden/.

## In-flight at pause (see members' handoffs)
- research (was 43% ctx — ROTATE on resume, near 60%): finished 0.8. Next = 0.9 (P→A feed, #19, absorbs 0.2b bit-for-bit), then 0.11 (O store), then 0.12 (final layout map docs/ck-tensor-layout-map.md). Method: two-run golden decode (set tensor=row, =col). tensor_layouts.md accumulating verified rows.
- impl (shut down): finished Kernel 1 (test_k_lds, QA-PASS) AND Kernel 2 (test_qk_gemm, QA-PASS, 0/8192 bit-exact GEMM0). [My earlier "clean slate" note was STALE — Kernel 2 was fully written + passing; corrected.] impl handoff is excellent: status/impl.md has the full SwizzleA model, -O3 fix, and the builtin gotcha. Read it.
- KEY impl findings (all in knowledge.md): GEMM0 SwizzleA bit2/3 on K(A) operand — swz(x)=(x&~0xC)|(((x>>2)&1)<<3)|(((x>>3)&1)<<2), already coded in qk_gemm.hpp (golden 0/4096); -O3 REQUIRED on every Phase-1 HIP target (default -O0 wrecks codegen); async lds copy needs __builtin_amdgcn_raw_ptr_buffer_load_lds (hand-asm doesn't assemble on gfx942).
- DEFERRED (not done before shutdown): harness --golden fix (single dir vs 2 tile fixtures — run each binary per-tile dir for now; fix in knowledge.md). EXPECT_EQ→ASSERT_EQ nit. prof to reconcile ds_read_b128 gate count 16 vs actual 8 (b128 serves 2 passes — 8 is correct/minimal).
- Both kernels' code is UNCOMMITTED working-tree changes (k_lds + qk_gemm + CMake). Nothing half-written.
- Members DID write good self-handoffs (research.md 5944B, impl.md 4947B, both in canonical repo-subdir status/). I also verified all state from disk independently.

## Workflow that's working (keep it)
- Two tracks: research decodes CK layouts vs golden (mapping); impl builds native Phase 1 kernels (reproduction). Parallel, no file conflicts (research→/tmp docs, impl→src/+tests/).
- Per-layout: research golden-decodes → records formula → impl builds native kernel using it → lead runs QA subagent (mandatory, independent) → lead runs prof asm gate (Phase 1+).
- Golden-verified-ONLY standard: never mark "verified" without bit/byte-exact golden match. Use confidence levels (golden-verified / asm-confirmed / source-derived / inferred). Golden has already caught 2 wrong source formulas (Q, S_acc) — DO NOT trust source-derived formulas for native staging.

## Standing user rules (in /tmp/.../lead/decisions_001.md)
- Lead 50% ctx → stop & ask user (changed from 40%→50% by user 2026-05-29). Teammate 60% → rotate.
- Keep ~5 tasks ahead. Doc learnings after EACH task (knowledge.md + status.md). Verified/correct content only.
- Run autonomously; surface only genuine decisions. Monitor via tmux pane (match ps --agent-name to pane pid; status bar = context%/idle).

## Next actions on resume
1. Read recall knowledge.md + status.md, both member handoffs (research.md, impl.md — both good).
2. Re-spawn team (resume.md auto-stops stale agents). Spawn fresh research + fresh impl. Optionally spawn prof for asm gates.
3. Kernel 1 + Kernel 2 already QA-PASS — do NOT re-verify or rewrite. (If paranoid, one clean rebuild of both, but they passed independent QA.)
4. Early cleanup (small, deferred from last session): apply the harness --golden per-tile fix to test_k_lds.cpp + test_qk_gemm.cpp (currently must run each binary once per tile dir); EXPECT_EQ→ASSERT_EQ nit; have prof reconcile the ds_read_b128 gate (16 expected vs 8 actual — 8 is correct).
5. Then continue two tracks: research 0.9 (P→A, absorbs 0.2b) → 0.11 (O store) → 0.12 (final layout map); impl Kernel 3 test_row_max → 4 softmax → 5 v_lds → 6 pv_gemm → 7 epilog. REMEMBER: -O3 in CMake for every new Phase-1 target; use __builtin_amdgcn_raw_ptr_buffer_load_lds for async copies.
6. Re-verify golden staging (md5) before any new test run. Golden in /tmp (host + both containers) — may not survive reboot; if gone, regenerate via tools/ck_instrumented/build.sh + dump_runner in poyenc-ck, then docker cp to host + poyenc-fmha. mfma_probe.hip (SwizzleA tool) at /tmp/mfma_probe.hip — reusable for GEMM1.
7. CONSIDER committing the Phase-1 work (k_lds + qk_gemm + CMake, all uncommitted) early — it's QA-passed and currently only in the working tree.
