# Implementer Status — 2026-05-31 (Phase 3, Task 1)

## Current State: Task 1 COMPLETE — build passes, Gate 4 VGPR FAIL (131 vs ≤128)

### Phase 3 Progress

| Task | Status | Notes |
|------|--------|-------|
| Task 1: Rewrite epilog_store() | ✅ Code complete, build clean | Gate 4: VGPR 131, target ≤127 |
| Task 2: Add 10 sched_barriers | ⏳ Pending | Blocked on Task 1 Gate 4 |
| Task 3: Benchmark native vs CK | ⏳ Pending | Blocked on Tasks 1-2 |
| Task 4: Final commit | ⏳ Pending | Blocked on Tasks 1-3 |

### Task 1 Implementation Details

**Files modified:**

1. `src/kernel/fmha_fwd_d64_epilog.hpp` — Complete rewrite (112→91 lines):
   - Deleted old `epilog_store()` (element-wise `global_store` via `o_base[row*stride+col]`)
   - Deleted legacy `epilog_store_o()`
   - New `epilog_store()`: SRD built inside function, `v_perm_b32` normalize+pack, 8× `buffer_store_dwordx2`, `s_waitcnt vmcnt(0)`
   - Dropped `o_srd` parameter; `o_base` used for SRD construction
   - voffset = `(abs_m_row * stride_o + swz(k_sub * 8)) * 2`
   - Per-store row boundary guard (`abs_m_row < seqlen_q`)

2. `src/kernel/fmha_fwd_d64_device.hpp` — 3 surgical edits:
   - Removed `auto srd_o = make_buffer_resource(o_base);` (line 70)
   - Removed `srd_o` from early-exit epilog call (line 103-104)
   - Removed `srd_o` from normal-path epilog call (line 235-236)

**Deviations from spec:**
1. Row-only boundary guard (no column check — always passes for D=64)
2. `reinterpret_cast<unsigned&>` instead of `__builtin_bit_cast` (compiler bug workaround)
3. Removed early-return guard — SRD + per-store check handles OOB

**Build:** Clean — 0 errors, 0 warnings. Output at `/tmp/fmha-native-isa-match/impl/phase3_task1_build.txt`

### Gate 4 VGPR Failure — Root Cause Analysis

**Result:** VGPR 131 (target ≤127). Epilog rewrite succeeded (buffer_store emitted, global_store eliminated), but VGPR did not drop to target.

**Root cause:** NOT the epilog. The VGPR pressure comes from `pack_p_subtile()` in `fmha_fwd_d64_gemm.hpp`. The `reinterpret_cast<const unsigned*>(&p_half)[i]` pattern (used to work around the `__builtin_bit_cast` compiler bug from Phase 2 bug #8) forces the compiler to spill the entire `v16f` to stack to take its address, inflating register pressure in the hot loop.

**Implication:** Fixing VGPR requires addressing the `pack_p_subtile` cast pattern, not further epilog changes. This is a cross-task issue that affects both Task 1's Gate 4 and Task 2's risk assessment.

### CK Assembly Reference (from Explore subagent analysis)

Key findings used during implementation:
- CK normalize+pack: interleaved `v_pk_mul_f32` + `v_perm_b32` with selector `0x07060302`
- CK stores: 8× `buffer_store_dwordx2 v[data], v1, s[4:7], 0 offen offset:{0,16,...,112}`
- CK voffset: single VGPR `v1 = 2 * (row * stride + col_base)`
- CK boundary: `v_cmp_gt_i32` row AND column checks with `s_and_saveexec_b64`
- CK SRD: `s[4:7]`, word3 = `0x20000`, num_records reconstructed per store

### Prior Phase 2 Status (preserved)

- **60/60 tests pass** (58 Full/Smoke + 2 CK Golden)
- 10 bugs fixed across Phase 2
- ISA: 32 MFMA, 16 ds_read_b128, 0 spills, LDS 13824
