# Handoff: lead — 2026-05-30

## State summary
Phase 0 COMPLETE. Phase 1 COMPLETE + COMMITTED (e4f1d85, 25d56cc).
Phase 2 design doc WRITTEN (docs/superpowers/specs/2026-05-30-phase2-fused-kernel-design.md).
Team shut down cleanly. Resume with `/hip-kernel-team load fmha-native-isa-match`.

## What was accomplished this session

### Phase 0 (completed)
- 0.9 P→A verified (no shuffle, bf16 truncation) — 4096/4096
- 0.11 O store verified (Default2D, TransposedC, no shuffle)
- 0.12 tensor_layouts.md finalized — all 8 tensors documented

### Phase 1 (completed, committed)
- Kernels 3-7 built, tested, QA-passed:
  - K3 row_max (1 ds_bpermute, not 5-round butterfly)
  - K4 softmax (exp2, bf16 trunc, 1 bpermute for rsum)
  - K5 v_lds (v_perm shuffle + ds_write2_b32)
  - K6 pv_gemm (GEMM1, no SwizzleA on V, O_acc inherits SwizzleA from P)
  - K7 epilog (normalize, SwizzleA'd headdim store, LSE)
- Test harness fixes: --golden-full/--golden-partial, output format normalized,
  ASSERT_EQ for HIP setup, epilog CPU-ref self-contained, edge case tiles
- 49/49 tests pass, 7 binaries × 7 tests each, all EXIT 0
- Independent verification: 3 spec review agents + prof ISA review + doc audit
- Commit e4f1d85 (Phase 1 kernels) + 25d56cc (ASSERT_EQ fix)

### Critical findings logged to knowledge.md
- Existing softmax butterfly is WRONG (root cause of 43/50 failures)
- O_acc inherits SwizzleA from P (epilog must use swz'd headdim)
- GEMM convention: A=LDS, B=register (not the reverse)
- GEMM1 does NOT SwizzleA V reads (unlike GEMM0's SwizzleA on K)
- bf16 uses truncation not RNE (P and O store)

### Phase 2 design (written, not started)
- 19 design decisions resolved via /grill-me interview
- 10 tasks broken down in docs/superpowers/specs/2026-05-30-phase2-fused-kernel-design.md
- Tasks 2.2-2.5 parallelizable (4 inner files), 2.6 integrates, 2.7 debug loop
- LdsSeq = {1, 2, 1, 0} for our D64 config (k0=2, k1=2, 3 buffers)
- Full CK compile flags documented and ready to apply

### Process improvements
- Verification enforcement rules added to team config
- "Done is a claim, artifact is the fact" rule added to ~/.claude/CLAUDE.md
- Caught impl's ASSERT_EQ fix that was never applied — doc audit found it

## Key files
- Phase 2 design: docs/superpowers/specs/2026-05-30-phase2-fused-kernel-design.md
- Tensor layouts: /tmp/fmha-native-isa-match/research/tensor_layouts.md
- Knowledge: ~/.local/share/claude/recall/fmha_native/branches/isa-match-rewrite/tasks/isa-match-rewrite/knowledge.md
- Status: ~/.local/share/claude/recall/fmha_native/branches/isa-match-rewrite/tasks/isa-match-rewrite/status.md

## Next actions on resume
1. Read Phase 2 design doc
2. Spawn team (impl + research, possibly prof)
3. Start with task 2.1 (CMake flags), then 2.2-2.5 in parallel
4. Task 2.6 integrates everything, 2.7 is the debug loop
5. Golden data in /tmp may not survive reboot — verify md5, regenerate if needed
