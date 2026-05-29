# Research Status — fmha-native-isa-match

**Last updated**: 2026-05-28
**Agent**: research (Researcher)

## Completed Tasks

### Task #1: ISA rewrite research spec
- **Output**: `docs/isa-rewrite-spec.md` + `/tmp/fmha-native-isa-match/research/isa-rewrite-spec.md`
- 669 lines, 6 sections: K async copy, V load/repack/store, GEMM1 MFMA, O store, register budget, Q load correction
- All claims cite CK assembly line numbers or source file:line
- **Correction applied**: MFMA intrinsic changed from `_1k` to non-1k variant (`__builtin_amdgcn_mfma_f32_32x32x8bf16`)

### Task #10: Register pressure analysis
- **Output**: `/tmp/fmha-native-isa-match/research/register-pressure-analysis.md`
- **Root cause #1**: Missing `__launch_bounds__` second argument (our: 1 arg, CK: 2 args with minBlocksPerCU=3)
- **Root cause #2**: Missing `sched_barrier(0)` at phase boundaries (CK has 10, we had few)
- CK uses no `amdgpu_num_vgpr`, `noinline`, `flatten`, or `iglp_opt` — only `__launch_bounds__` + `sched_barrier`

### Task #12: Top-down pipeline analysis
- **Output**: `/tmp/fmha-native-isa-match/research/top-down-pipeline-analysis.md`
- **Critical finding**: Softmax state arrays oversized — `float rmax[16]` + `float rsum[16]` = 32 VGPRs, CK uses ~4 VGPRs
- Root cause: In TransposedC MFMA layout, each lane's 16 C-registers span different N-columns but SAME M-row → row-max/sum reduces to 1 scalar per lane
- Full phase-by-phase pipeline comparison (CK vs ours)
- C++ pattern → assembly instruction mapping table
- VGPR waterfall analysis showing CK peak at ~127 vs our ~149

## Key Findings (Cross-Task)

1. **`__launch_bounds__(256, 3)`** is the #1 lever for VGPR control — forces compiler to target ≤168 VGPRs
2. **Softmax scalar reduction** saves ~28 VGPRs — each lane owns 1 M-row, not 16 separate rows
3. **`sched_barrier(0)`** at phase boundaries prevents live-range extension across phases
4. **CK's v_perm selectors**: `0x01000504`/`0x03020706` for V repack, `0x07060302` for fp32→bf16 truncation
5. **CK has 0 AGPRs** — on gfx942, MFMA writes directly to regular VGPRs
6. **CK Q loads 4×dwordx4** (32 bf16/thread), not 8 — k_sub=0 and k_sub=1 lanes cover complementary halves

## Research Outputs Directory

All saved to `/tmp/fmha-native-isa-match/research/`:
- `isa-rewrite-spec.md` — ISA rewrite spec (Task 1)
- `register-pressure-analysis.md` — Register pressure root cause (Task 10)
- `top-down-pipeline-analysis.md` — Top-down pipeline analysis (Task 12)

## No Remaining Tasks

All assigned research tasks completed. No pending work.
