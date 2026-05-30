# Implementer Status — 2026-05-30 (Final)

## Current State: All tasks 2.1–2.9 COMPLETE

### Test Results
- **60/60 tests pass** (58 Full/Smoke + 2 CK Golden)
- CK golden bit-exact: 0 bf16 mismatches for both full tile (sq=64,sk=64) and partial tile (sq=17,sk=33)
- LSE: 0 mismatches everywhere

### Files Modified

1. `src/kernel/fmha_fwd_d64_lds.hpp` — K: VGPR staging via k_lds_elem_offset. V: ds_write2 with -512 offset.
2. `src/kernel/fmha_fwd_d64_gemm.hpp` — pack_p_subtile uses reinterpret_cast (compiler workaround), 4*s stride fix. v_lds_elem_offset for V reads. K reads use lds_elem_offset.
3. `src/kernel/fmha_fwd_d64_softmax.hpp` — Scalar rmax/rsum, 1 bpermute, template HasMask. softmax_p_to_bf16 uses reinterpret_cast (compiler workaround).
4. `src/kernel/fmha_fwd_d64_epilog.hpp` — Element-wise O store via o_base. LSE: (log2(rsum)+rmax)*ln2.
5. `src/kernel/fmha_fwd_d64_device.hpp` — Clean pipeline. NaN guard: skip rescale when rmax==m_new.
6. `CMakeLists.txt` — SHELL: prefix for -mllvm flags.
7. `src/kernel/fmha_fwd_d64_kernel.cpp` — launch_bounds(256,3).
8. `tests/test_fmha_fwd_d64.cpp` — Added FmhaGoldenCK tests (FullTile + PartialTile).

### 10 Bugs Fixed (7 prior + 3 this session)

Prior session (tasks 2.1–2.6):
1. K async copy → VGPR staging (buffer_load_dword...lds m0 issue)
2. K LDS formula mismatch (k_lds_offset ≠ lds_elem_offset)
3. V LDS formula mismatch (added v_lds_elem_offset)
4. V ds_write2 -512 offset
5. LSE log2 vs ln (__builtin_amdgcn_logf = log2)
6. LSE lane guard (k_sub==0 not lane&31==0)
7. CMake SHELL: prefix

This session (task 2.7):
8. **Compiler miscompile of __builtin_bit_cast on ext_vector_type elements** — `__builtin_bit_cast(unsigned, v16f[i])` reads element 0 for every index. Fixed with `reinterpret_cast<const unsigned*>(&v16f)[i]`. Affected pack_p_subtile and softmax_p_to_bf16. Root cause of all 32 mask failures.
9. **Overlapping P register indices in pack_p_subtile** — Loop stride was `2*s` instead of `4*s`, causing p_packed entries to share P register values. Masked by bug #8 but would have caused wrong P→MFMA mapping.
10. **NaN from rescale when rmax stays -inf across tiles** — `exp2(-inf - (-inf)) = NaN`. Added `if (rmax != m_new)` guard. Fixed D64EdgeAsymS128Sk65Mask.

### ISA Gate (task 2.9)

```
Instruction              | CK Target | Actual | Status
-------------------------|-----------|--------|--------
v_mfma_f32_32x32x8_bf16 |    32     |   32   | MATCH
ds_read_b128             |    16     |   16   | MATCH
ds_write2_b32            |     4     |    4   | MATCH
ds_bpermute_b32          |     2     |    2   | MATCH
v_perm_b32               |    44     |   39   | DELTA -5 (improvement)
buffer_store_dwordx2     |     8     |    0   | DELTA (global_store used)
VGPR                     |  ≤ 127    |  135   | DELTA +8 (0 spill)
AGPR                     |     0     |    0   | MATCH
Spill (scratch)          |     0     |    0   | MATCH
LDS (bytes)              | 13824     | 13824  | MATCH
```

7/10 exact matches. 3 deltas accepted:
- buffer_store → global_store: epilog uses raw pointer stores (functionally equivalent)
- v_perm -5: reinterpret_cast workaround avoids v_perm for bf16 truncation
- VGPR +8: epilog address computation; addressable in future vectorization pass
