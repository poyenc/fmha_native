# Research — Handoff (2026-05-29, paused after Phase 0.8)

**Agent**: research (Researcher) | Supersedes the old bottom-up status below the line.

## Role
I read CK source, write layout hypotheses, and GOLDEN-VERIFY them against dumps. Division of labor: I own the dump tool + all layout mapping (golden-verified); impl builds the native kernel (Phase 1) from my verified formulas.

## Current state: 4 layouts GOLDEN-VERIFIED
| Layout | Phase | Status | Doc |
|---|---|---|---|
| K LDS | 0.6 | ✅ 4095/4096 | hypothesis_k_lds.md |
| Q load | 0.7 | ✅ 4096/4096 (source CORRECTED) | hypothesis_q_load.md |
| V load+shuffle+LDS | 0.10 | ✅ 2047/2048 | hypothesis_v_lds.md |
| S_acc (MFMA C) | 0.8 | ✅ 4096/4096 (0.2 CORRECTED) | hypothesis_s_acc.md |

All formulas also in `tensor_layouts.md` (the 0.12 registry, kept current). The "1 miss" in K/V is always the (0,0) zero-pad alias — benign, structurally at the predicted offset.

## Output files (host: /tmp/fmha-native-isa-match/research/)
- pipeline_trace.md (0.1), distribution_types.md (0.2 — see ⚠️ below), shuffle_trace.md (0.3)
- build_repro_verification.md (proved I can build CK's exact kernel; ISA profile = Phase 3 gate)
- dump_kernel_design.md (0.4), golden_verification.md (0.5)
- hypothesis_k_lds.md, hypothesis_q_load.md, hypothesis_v_lds.md, hypothesis_s_acc.md
- **tensor_layouts.md** — canonical registry, all 4 verified formulas. START HERE.

## Verified formulas (quick ref — full detail in docs)
- **K LDS**: offset = base + (j%4)*136 + ((j/4)%4)*32 + (j/16)*544 + (d%32) + (d/32)*2304
- **Q**: m_row=(lane%32)+32*warp ; k_headdim=(r/8)*16+(lane/32)*8+(r%8) ; 32 bf16 regs
- **V LDS**: k=n%32 ; offset=base+(k/8)*576+(d/8)*72+(d%8)*8+(k%8) ; headdim=row(72=64+8pad)
- **S_acc**: m_row=(lane%32)+32*warp ; n_col=(r/8)*16+(lane/32)*8+(r%8) ; 32 fp32 regs

## ★ KEY STRUCTURAL FINDING (drives 0.9/0.11)
S_acc, Q, P, O ALL share one register structure: M-dim=(lane%32)+32*warp, free-dim=(r/8)*16+(lane/32)*8+(r%8). TransposedC makes C(S)=A(P) — **P needs NO shuffle, just cast fp32→bf16 in place** (0.3 escalation, confirmed from C-side in 0.8). So 0.9 and 0.11 are mostly confirm-the-cast/store, not new dist derivation.

## ⚠️ Two source-formula corrections (golden caught both)
1. **Q (0.7)**: source said headdim groups of 4 (kABKPerLane); golden = groups of 8. Use verified Q formula.
2. **S_acc (0.8)**: my 0.2 worked example (n=lane%32, m=(r/4)*8+(lane/32)*4+(r%4)) is the NON-transposed MFMA layout — WRONG for this kernel. CK uses TransposedC (verified 0.8 formula). **distribution_types.md §5 is superseded for native staging** — use hypothesis_s_acc.md.
Lesson: golden-validate every layout; never ship source-only.

## Tooling (you own this)
- Dump kernel: `tools/ck_instrumented/` in the CK tree (bind-mounted in poyenc-ck). dump_utils.hpp (8 slots: K_LDS=0,S_ACC=1,RMAX=2,P=3,V_LDS=4,O_ACC=5,O_FINAL=6,Q=7), fmha_fwd_dump_pipeline.hpp (instrumented, struct renamed ...AsyncDump), dump_runner.cpp, build.sh.
- Build: `docker exec poyenc-ck /root/workspace/tools/ck_instrumented/build.sh /root/workspace/build/ck_dump_build_NNN.txt` (builds instrumented + plain).
- Run: `docker exec poyenc-ck bash -c 'HIP_VISIBLE_DEVICES=1 /root/workspace/tools/ck_instrumented/dump_runner <outdir> <Sq> <Sk> <D> <mode>'`
- Decode modes (controlled inputs): kj/kd (K), qm/qd (Q), vj/vd (V), sm/sn (S_acc). Two-run trick: set tensor=index1 then =index2; each dump slot's (val1,val2) pair = the (idx1,idx2) element it holds → decode offset/reg → element directly. Avoids periodic-input aliasing.
- Golden dirs: /tmp/fmha-native-isa-match/golden/{full,partial,*_plain,decode_*}/

## ⚠️ Environment gotchas
- **Containers don't share /tmp** (poyenc-ck, poyenc-fmha, host all separate). After any run, stage: poyenc-ck→host (docker cp), then host→poyenc-fmha (docker cp). **docker cp CANNOT do container→container** (silently makes empty dirs). Always 2-hop via host. md5-verify all 3 locations.
- poyenc-ck bind-mounts the CK tree at /root/workspace (= host .../composablekernel). Files I create in tools/ appear in the container.
- GPU: HIP_VISIBLE_DEVICES=1 (GPUs 1-7 free, GPU 0 busy).
- Build flags MUST include -DCK_TILE_FMHA_FWD_FAST_EXP2=1 + -mllvm -amdgpu-* inline flags for ISA match (in build.sh already).

## NEXT ACTION for successor: Phase 0.9 (P→GEMM1-A feed)
1. Likely Task #19 — verify P feeds GEMM1 A with NO shuffle, cast fp32→bf16 only.
2. Golden: P = dump_reg slot 3 (32 regs, bf16, dtype_tag=1), already captured in golden/full.
3. Expectation (strong, from 0.8): P's register→element map == S_acc's exactly. Compare P slot vs S_acc slot element-wise: same lane/reg → same (m,n), with P=bf16(softmax(S)). Element-ownership match → 0.9 done.
4. This ABSORBS 0.2b (P A-dist bit-for-bit = S_acc C-dist). If P slot register ownership == S_acc slot register ownership, both done.
5. Then 0.11 (O_acc slot 5 + O_final slot 6, Default2D store — O shares same dist; confirm cast+store, no shuffle; golden_verification.md already showed O_FINAL faithful), then 0.12 (finalize tensor_layouts.md).

## Open threads
- (0,0) zero-pad alias: benign verification caveat in K/V LDS. Not a kernel gap.
- S_acc≡Q≡P≡O structural finding: leverage it — 0.9/0.11 are fast confirmations, not full derivations.
- impl is building Phase 1 native kernel from these formulas in parallel — keep formulas authoritative; if further golden correction arises, flag impl immediately.

---
## (Archived) prior bottom-up session status — SUPERSEDED by top-down approach above
Old docs still valid as reference: isa-rewrite-spec.md, register-pressure-analysis.md, top-down-pipeline-analysis.md (in /tmp/.../research/). Key reusable findings: __launch_bounds__(256,3) for VGPR control; softmax scalar reduction (1 M-row/lane); sched_barrier(0) at phase boundaries; v_perm selectors 0x01000504/0x03020706; 0 AGPRs on gfx942.
