# Handoff: lead — 2026-05-31

## Current Task
- Task #1 — Rewrite epilog_store() — buffer_store_dwordx2 via SRD
- Status: in-progress, Gate 4 VGPR FAIL (131 vs target ≤128)

## What was accomplished this session (Phase 3)

### Task 1 implementation (partial — Gates 1-3 pass, Gate 4 fails)
- epilog_store() rewritten: SRD built inside function, 8× buffer_store_dwordx2, per-store boundary guard, s_waitcnt vmcnt(0)
- Legacy epilog_store_o() deleted
- srd_o removed from device.hpp (construction + both call sites)
- Used reinterpret_cast (not __builtin_bit_cast) per known compiler bug
- Build: clean, 0 errors, 0 warnings
- Gate 2: 60/60 fused + 49/49 standalone — all pass
- Gate 3: included in 60/60 (golden bit-exact tests pass)
- Gate 4 instruction counts: PASS (buffer_store_dwordx2=8 per path, global_store=0, MFMA=32)
- **Gate 4 resource check: FAIL** — VGPR=131, occupancy=3 (target: next_free_vgpr≤128, occupancy=4)

### Root cause of VGPR 131 (investigated)
- NOT the epilog — epilog uses only v0-v39, well below threshold
- Root cause: `pack_p_subtile` in hot loop (`fmha_fwd_d64_gemm.hpp`)
- Compiler extracts 7 bf16 values simultaneously (v124-v130) via v_and_b32 before consuming with v_perm_b32
- Fix: interleave extract+consume pairs to reduce peak temps from 7 to ~2
- This saves ~5 VGPRs, bringing total to ~126 (under 128 threshold)

### ISA comparison (native vs CK)

| Metric | Native | CK | Target | Status |
|--------|--------|-----|--------|--------|
| buffer_store_dwordx2 | 16 (8×2 paths) | 8 | 8/path | PASS |
| global_store_dwordx2 | 0 | 0 | 0 | PASS |
| global_store_short | 0 | 0 | 0 | PASS |
| MFMA | 32 | 32 | 32 | PASS |
| sched_barrier | 1 | 10 | 10 (Task 2) | expected |
| VGPR | 131 | 127 | ≤128 | FAIL |
| Occupancy | 3 | 4 | 4 | FAIL |
| LDS | 13824 | 13824 | 13824 | PASS |
| Spills | 0 | 0 | 0 | PASS |

## Files Modified
- `src/kernel/fmha_fwd_d64_epilog.hpp` — full rewrite of epilog_store()
- `src/kernel/fmha_fwd_d64_device.hpp` — removed srd_o, updated call sites

## Key Files
- Build log: `/tmp/fmha-native-isa-match/impl/phase3_task1_build.txt`
- Fused test log: `/tmp/fmha-native-isa-match/lead/phase3_task1_test_fused.txt`
- Standalone test log: `/tmp/fmha-native-isa-match/lead/phase3_task1_standalone.txt`
- Native ISA analysis: `/tmp/fmha-native-isa-match/isa-analysis/native-phase3/scratch/`
- CK ISA analysis: `/tmp/fmha-native-isa-match/isa-analysis/ck-phase3/scratch/`
- Native assembly: `native_d64_kernel.s` (repo root)

## Plan Revised (2026-05-31)

VGPR gate deferred until after sched_barriers land, since barriers change
register lifetimes. New task sequence:

```
Task 1: Epilog buffer_store — Gates 1-3 only (DONE, ready to commit)
Task 2: 10 sched_barriers — Gates 1-3 only
Task 2.5: VGPR occupancy guard — full Gate 4 (match CK occupancy)
Task 3: Benchmark (<1% per workload)
Task 4: Final verification + commit
```

## Next Actions on Resume
1. Commit Task 1 (epilog rewrite — all Gates 1-3 pass, instruction counts verified)
2. Proceed to Task 2 (10 sched_barriers)
3. After Task 2, run Task 2.5 (VGPR guard — fix pack_p_subtile if needed)
4. Then Task 3 (benchmark) and Task 4 (final commit)
