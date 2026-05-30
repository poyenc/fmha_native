# ISA Rewrite Spec: fmha_native D64 FMHA Kernel (gfx942)

**Target**: Match CK's D64 FMHA kernel instruction mix on gfx942 (MI-300X, CDNA3).
**Source assembly**: `ck_d64_kernel.s` (CK baseline).
**Source C++**: `ck_tile/core/arch/amd_buffer_addressing.hpp`, `ck_tile/core/arch/utility.hpp`.

Every claim cites `[asm:LINE]` for CK assembly or `[src:FILE:LINE]` for CK/fmha_native source.

---

## 1. K Async Copy Spec

### Goal
Replace VGPR-staging K copy (`buffer_load_b128` -> VGPR -> LDS write) with direct DRAM-to-LDS async copy using `buffer_load_dword ... lds` inline asm.

### Inline Asm Template

Per load (1 dword = 2 bf16):
```cpp
// Set m0 to target LDS byte offset (once per block of 4 loads)
asm volatile("s_mov_b32 m0, %0" : : "s"(lds_base) : "memory");
// [src:utility.hpp:21-24]

// Async load: DRAM -> LDS, bypasses VGPRs
char* smem;  // dummy output for compiler dependency
asm volatile("buffer_load_dword %0, %1, 0 offen offset:0 lds"
             : "=r"(smem)
             : "v"(voffset), "s"(srd)
             : "memory");
// [src:amd_buffer_addressing.hpp:1349-1354]

// Advance m0 to next LDS row
asm volatile("s_add_u32 m0, %0, m0" : : "n"(0x440) : "memory");
// [src:utility.hpp:26-29], [asm:535,543,550,675,682,689,1093,1100,1107]
```

**Operand constraints**:
| Operand | Constraint | Description |
|---------|-----------|-------------|
| `%0` (output) | `"=r"(smem)` | Dummy `char*` output. Creates compiler dependency on LDS. NOT `"=v"`. |
| `%1` (voffset) | `"v"(voffset)` | Per-lane VGPR byte offset into K buffer |
| `%2` (SRD) | `"s"(srd)` | 128-bit buffer resource descriptor (SGPR) |
| Clobber | `"memory"` | Prevents reordering across LDS accesses |

**m0 stride**: `0x440` = 1088 bytes. This is the stride between consecutive "issue groups" in the padded LDS layout. Derivation: `kNPerBlock/NPerRow * NPerRow * kKPack * sizeof(bf16)` = `(64/8) * 8 * 8 * 2` ... but actually `0x440 = 4 * 272` where 272 = LdsRowStride with alignment. The CK assembly consistently uses 0x440 at [asm:535,543,550,675,682,689,1093,1100,1107].

### Per-Thread voffset Formula

256 threads issue 4 loads each = 1024 dwords = 2048 bf16 = one 64x32 K tile.

```
warp_id  = tid / 64      (0..3)
lane_id  = tid % 64      (0..63)

// Base position within the tile
k_col    = (lane_id % 16) * 2    // K-dim position (0..30, stride 2, 2 bf16 per dword)
n_row    = warp_id * 4 + (lane_id / 16)   // seqlen_k position within warp's 16-row group

// Global byte offset for load i (i = 0..3):
//   n_pos_i = i * 16 + n_row     (advance 16 rows per load)
//   voffset_i = (kv_offset + n_pos_i) * stride_k_bytes + k_col * sizeof(bf16)
//   where stride_k_bytes = headdim * sizeof(bf16) = 64 * 2 = 128
```
[asm:513-520,530 — prologue voffset computation via v_mad_u64_u32]

Between loads, advance voffset by `16 * stride_k_bytes`:
```cpp
int voffset_stride = 16 * stride_k * sizeof(__hip_bfloat16);  // = 16 * 128 = 2048
// After each load: voffset += voffset_stride
```
[asm:538,545,552 — `v_add_u32_e32 v10, s5, v10` where s5 = stride << 5]

### m0 Init Values (LDS buffer base)

The m0 value determines which LDS buffer receives the data:

| Buffer | m0 base formula | Byte offset |
|--------|----------------|-------------|
| buf0 | `0` | 0 |
| buf1 | `0x1200` | 4608 |
| buf2 | `0x2400` | 9216 |

m0 also includes a per-warp offset: `m0 = buffer_base + warp_id * 0x110`. This positions each warp's 4 dwords at the correct LDS row. `0x110` = 272 = row stride in bytes considering alignment.

CK assembly [asm:525-528]:
```
s5 = s12 * 0x110 + 0x1200   // s12 = warp_id
s_mov_b32 m0, s5
```
[asm:663-668]: `s38 = warp_id * 0x110 + 0x2400`, `s_mov_b32 m0, s38`
[asm:1077-1086]: `s37 = warp_id * 0x110 + 0x1200`, `s_mov_b32 m0, s37`

**Per-warp m0 formula**:
```
m0_base = lds_buffer_offset + warp_id * 0x110
```
where `0x110` = 272 bytes = padded row stride. After each load, `m0 += 0x440` = 4 * 272, jumping 4 rows (one row per warp, so the next "issue" for this warp).

### Fence Pairing

After each block of 4 async loads:
```cpp
asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
```

Observed waitcnt values [asm:558,560,696]:
- `vmcnt(4)` — partial drain: wait until only 4 VMEM ops remain in flight
- `vmcnt(0)` — full drain: all VMEM complete

For our implementation: use `vmcnt(0)` after each K copy block for simplicity. Optimize later.

### Bounds Checking (Partial Tiles)

For the last K/V tile where `kv_offset + n_pos >= seqlen_k`, the async copy must be skipped or the voffset clamped. CK handles this via SRD num_records field — loads beyond the buffer return 0. The SRD `num_records` field already limits to `0xFFFFFFFF` in our code, so we need to either:
1. Clamp voffset to valid range, OR
2. Conditionally skip loads using exec mask

Current fmha_native uses the guarded path (`copy_k_to_lds_2x_guarded`) that zero-fills OOB. The inline asm approach should use SRD-based bounds: set `num_records` to the actual buffer size so OOB loads return 0 (SRD OOB behavior on gfx942 returns 0 for buffer loads).

### Load Count

- **Per main-loop iteration**: 8 async loads (2 k0 slices x 4 loads each)
- **Prologue**: 4 async loads (prefetch first k0 slice) [asm:532-554]
- **Total in hot path**: 8 per iteration [asm:672-693, 1090-1111], 4 in prolog

Current fmha_native issues loads in `copy_k_to_lds_2x_guarded` [src:fmha_fwd_d64_lds.hpp:~120-170]. Replace with two calls to a new `async_copy_k_to_lds` function, one per k0 slice.

---

## 2. V Load / Repack / Store Spec

### Goal
Add V load from DRAM via `buffer_load_dwordx2`, repack with `v_perm_b32`, store to LDS via `ds_write2_b32`. V currently bypasses LDS entirely (loaded as scalar in `gemm1_bpermute`).

### V Load: `buffer_load_dwordx2`

4 loads per thread (2 pairs of 2), each loads 2 dwords = 4 bf16.

**Thread-to-element mapping**:
```
warp_id   = tid / 64        (0..3)
lane_id   = tid % 64        (0..63)

n_hdim    = lane_id / 4     (0..15, maps to hdim/N1 dimension)
k_within  = lane_id % 4     (0..3, maps to K2 within warp's K0 slice)

// Warp W owns K0 slice [W*8 .. (W+1)*8 - 1] of seqlen_k within tile
// Per load pair (k3 = 0, 1):
v_row = kv_offset + warp_id * 8 + k_within * 2 + k3
v_col = n_hdim * 4          // 4 contiguous bf16 along hdim

voffset = v_row * stride_v * sizeof(bf16) + v_col * sizeof(bf16)
```
[asm:722-724,879-881 — buffer_load_dwordx2 v[88:89]/v[90:91]/v[92:93]/v[94:95]]

**Builtin**: `__builtin_amdgcn_raw_buffer_load_b64(srd, voffset, 0, 0)` generates `buffer_load_dwordx2`.

**SRD**: Construct V SRD via `__builtin_amdgcn_make_buffer_rsrc(v_ptr, 0, 0xFFFFFFFF, 0x00027000)`. Currently V has no SRD [src:fmha_fwd_d64_device.hpp — only srd_q and srd_k exist].

### V Repack: `v_perm_b32`

Purpose: Transpose a 2x2 bf16 matrix within register pairs so the LDS layout matches GEMM1's ds_read expectation.

**Selector constants** (from CK prolog [asm:625-626]):
```cpp
constexpr uint32_t kPermSel0 = 0x01000504;  // s16 in CK asm
constexpr uint32_t kPermSel1 = 0x03020706;  // s30 in CK asm
```

`0x01000504` selects: `{src1[byte1,byte0], src0[byte5,byte4]}` = `{src1_lo16, src0_hi16}`
`0x03020706` selects: `{src1[byte3,byte2], src0[byte7,byte6]}` = `{src1_hi16, src0_hi16_of_hi_dw}`

Wait — let me re-derive. `v_perm_b32` byte select: result byte i = src_byte[selector_byte_i], where src = {src0[7:0], src1[7:0]} concatenated as a 8-byte value (src1 is bytes 0-3, src0 is bytes 4-7).

For `0x01000504`:
- result byte 3 = src byte 0x01 = src1 byte 1
- result byte 2 = src byte 0x00 = src1 byte 0
- result byte 1 = src byte 0x05 = src0 byte 1
- result byte 0 = src byte 0x04 = src0 byte 0
- Result: `{src1_lo_bf16, src0_lo_bf16}` — takes low bf16 from each dword

For `0x03020706`:
- result byte 3 = src byte 0x03 = src1 byte 3
- result byte 2 = src byte 0x02 = src1 byte 2
- result byte 1 = src byte 0x07 = src0 byte 3
- result byte 0 = src byte 0x06 = src0 byte 2
- Result: `{src1_hi_bf16, src0_hi_bf16}` — takes high bf16 from each dword

**Pattern per pair** (2 input dwords in0, in1 → 2 output dwords out0, out1):
```cpp
// in0 = {bf16_a1, bf16_a0}, in1 = {bf16_b1, bf16_b0}
// out0 = {bf16_b0, bf16_a0}  (low bf16 from each)
// out1 = {bf16_b1, bf16_a1}  (high bf16 from each)
out0 = __builtin_amdgcn_perm(in0, in1, kPermSel0);  // src0=in0, src1=in1
out1 = __builtin_amdgcn_perm(in0, in1, kPermSel1);
```

CK assembly [asm:901-904]:
```
v_perm_b32 v97, v88, v90, s16    // out0 from pair 0
v_perm_b32 v88, v88, v90, s30    // out1 from pair 0 (overwrites in0)
v_perm_b32 v117, v89, v91, s16   // out0 from pair 1
v_perm_b32 v89, v89, v91, s30    // out1 from pair 1 (overwrites in0)
```
[asm:1066-1069]: same for 2nd pair with v92-v95.

**Note**: CK uses `v_perm_b32 dst, vsrc0, vsrc1, sSel` where vsrc0 is the FIRST argument (maps to src0 = bytes 4-7) and vsrc1 is the SECOND (maps to src1 = bytes 0-3). The `__builtin_amdgcn_perm(src0, src1, sel)` follows the same convention.

**Count**: 4 v_perm per pair x 2 pairs = 8 v_perm per iteration in main loop.

### V LDS Store: `ds_write2_b32`

2 writes per pair, 4 total per iteration.

**LDS address formula**:
```
v86 = ((lane_id >> 3) & 7) * 144 + ((lane_id >> 2) & 1) * 64 + (lane_id & 3) * 4
lds_addr = warp_id * 1152 + v86 + buffer_base
```
[asm — v86 is a precomputed per-lane constant, v91 in CK stores the final address]

Where:
- `((lane_id >> 3) & 7)` = n_group (0..7) — groups of 8 lanes
- `144` = row stride in bytes (72 bf16 elements x 2 bytes)
- `((lane_id >> 2) & 1)` = n_intra_half — sub-group selector
- `64` = half-row offset in bytes
- `(lane_id & 3)` = k_pack_pair (0..3)
- `4` = 4 bytes per dword
- `warp_id * 1152` = per-warp partition (`1152 = 8 rows * 144 bytes/row`)

**DW offsets** for ds_write2_b32 [asm:905-906,1070-1073]:
```
ds_write2_b32 lds_addr, data0, data1 offset0:128 offset1:132
ds_write2_b32 lds_addr, data2, data3 offset0:136 offset1:140
```
These are DW offsets (x4 bytes). Offset 128 = 512 bytes from lds_addr. The large offset moves to the second half of the k-dimension storage.

**Buffer destinations**: V reuses K's LDS space after K is consumed:
- k1=0: V stored in buf1 (offset 4608) — reuses K k0=0
- k1=1: V stored in buf0 (offset 0)
[src:design-spec — LdsBufferSequence <1,2,1,0>]

### Pipeline Position

Split across two points (matching CK [asm:722-724,879-881]):

| Pair | When | Fence | Repack+Store |
|------|------|-------|--------------|
| 1st (2 loads) | After barrier 2 (GEMM0 k0=0 done), before GEMM0 k0=1 | `s_waitcnt vmcnt(2)` | After GEMM0 k0=1, during causal mask |
| 2nd (2 loads) | At causal mask / softmax start | `s_waitcnt vmcnt(0)` | During softmax, before GEMM1 |

---

## 3. GEMM1 MFMA Spec

### Goal
Replace `gemm1_bpermute` (1024 bpermute + 1024 scalar FMA) with MFMA-based GEMM1, same structure as GEMM0.

### MFMA Instruction

```cpp
__builtin_amdgcn_mfma_f32_32x32x8bf16(a_v, b_p, o_acc, 0, 0, 0)
```
[asm: 32 total v_mfma_f32_32x32x8_bf16 in hot path]

### P-to-bf16 Packing (B Operand)

P lives in S_acc registers (fp32) after softmax: `s_acc_n0[16]` and `s_acc_n1[16]`, 32 VGPRs total (v[34:65] in CK).

**Conversion**: fp32 → bf16 truncation via `v_perm_b32` with selector `0x7060302`:
```cpp
constexpr uint32_t kFp32ToBf16Sel = 0x7060302;
// Pack two fp32 values (consecutive registers) into one dword of 2 bf16:
// result = v_perm_b32(fp32_hi, fp32_lo, 0x7060302)
// = {fp32_hi[byte7,byte6], fp32_lo[byte3,byte2]}
// = {bf16_trunc(fp32_hi), bf16_trunc(fp32_lo)}
```
[asm:627 — `s_mov_b32 s34, 0x7060302`; used at asm:985-986,1004-1005,1034-1035,1044-1045,1118-1119,1123-1124,1132-1133,1136-1137]

**Packing into short4** (MFMA B operand needs 4 bf16 = 2 dwords = `short4`):

Each lane holds `s_acc_n0[i]` (fp32 at N position = lane%32) and `s_acc_n1[i]` (N = 32 + lane%32). For GEMM1, K dimension = seqlen_k (N0=64), split into k1=0..1 of 32 each.

Per MFMA call, B operand needs 4 consecutive K values (bf16). Since each lane holds values at 2 N positions (n0, n1), and K maps to the N-dim of S_acc:

For k1=0 (K positions 0..31): source from `s_acc_n0` registers
For k1=1 (K positions 32..63): source from `s_acc_n1` registers

Within each k1, 4 kInner steps of 8, each step needs 8 bf16 (= short4 = 4 bf16 packed as 2 dwords). The P registers are packed similarly to how Q registers feed GEMM0:

```cpp
// For kInner step s (0..3) within k1:
//   p_bf16_lo = perm(s_acc_nX[2*s+1], s_acc_nX[2*s], kFp32ToBf16Sel)
//   p_bf16_hi = perm(s_acc_nX[2*s+3], s_acc_nX[2*s+2], kFp32ToBf16Sel)
//   b_operand = pack_short4(p_bf16_lo, p_bf16_hi)
```

This produces 4 packed bf16 operands per k1, 8 total for both k1 iterations. These map directly to the same indexing as `q_regs[8]` in GEMM0.

### P Register Index Mapping

The P-to-B mapping mirrors Q-to-B in GEMM0:
```
k1=0: p_packed[0] = pack(perm(s_acc_n0[1],s_acc_n0[0]), perm(s_acc_n0[3],s_acc_n0[2]))
      p_packed[1] = pack(perm(s_acc_n0[5],s_acc_n0[4]), perm(s_acc_n0[7],s_acc_n0[6]))
      p_packed[2] = pack(perm(s_acc_n0[9],s_acc_n0[8]), perm(s_acc_n0[11],s_acc_n0[10]))
      p_packed[3] = pack(perm(s_acc_n0[13],s_acc_n0[12]), perm(s_acc_n0[15],s_acc_n0[14]))
k1=1: p_packed[4..7] = same pattern with s_acc_n1
```

### V LDS Read (A Operand)

V is read from LDS via `ds_read_b128` (16 bytes = 8 bf16 per lane), same as K in GEMM0.

**LDS read address formula**: Same padded 5D descriptor as K:
```
Shape:    (kKPerBlock/kKPack, kNPerBlock/NPerRow, NPerRow, kKPack)
        = (4,               8,                  8,       8)
Strides:  (576,             72,                 8,       1) in bf16 elements

read_addr = buf_base + k_outer * 576*2 + n_group * 72*2 + n_inner * 8*2 + k_inner * 2
```

Per MFMA step, each lane reads 8 bf16 (one ds_read_b128). The lane-to-address mapping:
```
lane L:
  n_in_warp = L % 32
  k_in_warp = (L / 32) * 8   // 0 or 8 (two k-groups per warp)
  
  n_group = n_in_warp / 8     // 0..3
  n_inner = n_in_warp % 8     // 0..7
  
  lds_byte_addr = v_lds_base + k_outer * 1152 + n_group * 144 + n_inner * 16 + k_in_warp * 2
```

**SwizzleB**: GEMM0 uses SwizzleB with SFactor=2 for K reads. GEMM1 uses **non-SwizzleB** for V reads [CK reference Section G — "Non-SwizzleB variant"]. This means direct addressing without XOR.

### Accumulator-to-Output Mapping

O_acc has same layout as S_acc (TransposedC MFMA C-output):
```
o_acc_n0[16]: hdim columns 0..31   (v[2:17] in CK)
o_acc_n1[16]: hdim columns 32..63  (v[18:33] in CK)
```

### k-loop Structure

```
for k1 = 0, 1:
  for kInner = 0..3:  // 4 steps of 8 bf16
    a_v = ds_read_b128(v_lds_addr + k1*32*2 + kInner*8*2)  // V from LDS
    b_p = p_packed[k1*4 + kInner]                            // P from VGPRs
    // Two MFMAs per step (one per N sub-tile):
    o_acc_n0 = mfma(a_v, b_p, o_acc_n0, 0, 0, 0)
    o_acc_n1 = mfma(a_v, b_p, o_acc_n1, 0, 0, 0)
Total: 2 k1 * 4 kInner * 2 nIter = 16 MFMA
```

**O_acc rescale** (online softmax correction): 8x `v_pk_mul_f32` interleaved with k1=0 MFMAs in MFMA co-execution windows [asm:987-1029].

---

## 4. O Store Spec

### Goal
Replace element-wise `__hip_bfloat16*` pointer writes with `buffer_store_dwordx2` via SRD.

### MFMA C-output to bf16 Conversion

Same selector as P packing: `0x7060302`
```cpp
constexpr uint32_t kFp32ToBf16Sel = 0x7060302;

// For each pair of consecutive o_acc registers:
// bf16_packed = __builtin_amdgcn_perm(o_acc[i+1], o_acc[i], kFp32ToBf16Sel);
// This truncates 2 fp32 -> 2 bf16 packed in 1 dword
```

Normalize first: `o_acc[i] *= (1.0f / row_sum)` before conversion.

CK epilog conversion [asm:1179-1211]:
```
v_pk_mul_f32 v[22:23], v[36:37], v[NN:NN+1] op_sel_hi:[0,1]  // scale
v_perm_b32 vDst, v23, v22, s0                                   // s0 = 0x7060302
```

### SRD Construction for O

```cpp
auto srd_o = __builtin_amdgcn_make_buffer_rsrc(
    o_ptr + batch_offset + head_offset,  // base pointer
    0,                                    // stride (0 for raw)
    0xFFFFFFFF,                          // num_records
    0x00027000                           // flags (same as Q/K SRDs)
);
```
[asm: SRD in s[4:7], partially constructed inline in epilog]

### Per-Store voffset Formula

Each thread computes one voffset for all 8 stores (the 8 stores differ only by immediate offset):

```
warp_id = tid / 64
lane_id = tid % 64

k_sub   = lane_id / 32      // 0 or 1
n_pos   = lane_id % 32      // hdim position within 32-wide half

// M position base: warp's 32-row block + k_sub's 4-row sub-block
m_base  = (m_tile_idx * 128) + warp_id * 32 + k_sub * 4

// voffset = m_base * stride_o * sizeof(bf16) + n_pos * sizeof(bf16)
voffset = m_base * stride_o * 2 + n_pos * 2
```

### The 8 Store Offsets

Each store writes 2 dwords = 4 bf16. The 8 stores cover the full 128-row x 64-col O tile:

```
For m0_iter = 0..3 (4 groups of 8 M-rows):
  For n_iter = 0..1 (2 halves of 64 hdim):
    store_idx = m0_iter * 2 + n_iter
    byte_offset = m0_iter * 8 * stride_o * sizeof(bf16) + n_iter * 32 * sizeof(bf16)
```

CK assembly offsets [asm:1281,1301,1321,1341,1361,1381,1401,1421]:
```
offset: 0, 16, 32, 48, 64, 80, 96, 112 (bytes)
```

These are **immediate byte offsets** added to voffset. The stride between them is 16 bytes. With stride_o=64 bf16 elements = 128 bytes per row, 16 bytes = 8 bf16 = spacing along hdim or rows depending on layout.

**Interpretation**: Each offset steps by 16 bytes along the M dimension. With BHSD layout (stride_o = headdim = 64), each row = 128 bytes. 16 bytes = 8 bf16 along hdim. But CK uses 8 stores x 16-byte stride = 128 bytes total = one full row. This means each store writes 4 bf16 contiguous along hdim, and the 8 stores tile across the row:

```
Store i (0..7):
  addr = voffset + i * 16
  data = 4 bf16 packed as 2 dwords
  Covers hdim positions: i*4 .. i*4+3  (but only for this thread's M row)
```

Wait — this doesn't match. Each MFMA output register `o_acc[r]` holds data at M-row = `warp*32 + (r/4)*4 + k_sub*4 + (r%4)` and N-col = `n_pos (+ n_iter*32)`. 4 consecutive registers (r%4 = 0..3) are at the same N column but consecutive M rows. So the 2 dwords per store pack 4 consecutive M-row values at the same N position.

**Corrected interpretation**: The 8 stores iterate:
- `m0_iter` = 0..3: selects groups of 4 M-rows within the warp's partition
- `n_iter` = 0..1: selects first half (n_pos) or second half (32 + n_pos) of hdim

```
Store (m0_iter, n_iter):
  m = warp*32 + m0_iter*8 + k_sub*4   // base of 4 consecutive M rows
  n = n_iter*32 + n_pos
  offset = m0_iter * stride_o * 8 * 2 + n_iter * 32 * 2
         = m0_iter * 16 + n_iter * 64   ... no, CK offsets are 0,16,32,...112 linear

  Actually: offset = store_idx * 16 where store_idx = 0..7
  This means the stores pack data along a contiguous 128-byte region.
  voffset already encodes the base row+col position.
```

The simplest correct implementation: use the same offset pattern as CK (0,16,32,...,112) and encode the row/col base in voffset. The 4 bf16 per store are 4 consecutive M-row values at the same N position.

### Conditional Exec Mask

Each store is guarded by `s_and_saveexec_b64` [asm:1267,1287,...,1407]:
```cpp
// Mask out threads whose M-row exceeds seqlen_q
// exec = exec & (m_base + m0_iter*8 + k_sub*4 + 3 < seqlen_q)
```

The mask prevents OOB writes when seqlen_q is not a multiple of 128.

### Builtin

```cpp
__builtin_amdgcn_raw_buffer_store_b64(data, srd_o, voffset, offset, 0);
```
generates `buffer_store_dwordx2`.

---

## 5. Register Budget

### Target
- **VGPRs**: <= 127 (occupancy = 4 waves/SIMD)
- **Accum VGPRs**: 128 (arch VGPRs for MFMA accumulators)
- **accum_offset**: 128
- **Spills**: 0

### VGPR Allocation Plan

| Category | VGPRs | Range (CK ref) | Notes |
|----------|-------|-----------------|-------|
| O_acc n0 | 16 | v[2:17] | MFMA accumulators, hdim 0..31 |
| O_acc n1 | 16 | v[18:33] | MFMA accumulators, hdim 32..63 |
| S_acc n1 / P | 16 | v[34:49] | Reused for P bf16 after softmax |
| S_acc n0 / P | 16 | v[50:65] | Reused for P bf16 after softmax |
| Q tile | 16 | v[66:81] | Loaded once, kept resident |
| V voffset | 2 | v[82:83] | Base addresses for V loads |
| K voffset + misc | 4 | v[84:87] | K address computation |
| V load destinations | 8 | v[88:95] | buffer_load_dwordx2 targets |
| V repack temps | 4 | v[96:99] or v[97,117,...] | Repack output (partially overwrite V loads) |
| Softmax temps | 4 | v[100:103] | max, sum, scale, exp_scale |
| LSE | 1 | v[101] | Loop-carried log-sum-exp |
| Address misc | 12 | v[104:115] | Loop counter, tile offsets, SRD words |
| Loop max | 1 | v[116] | Loop-carried max value |
| ds_read temps | 4 | v[118:121] | ds_read_b128 destinations for MFMA A |
| Misc | 6 | v[122:127] | Remaining |
| **Total** | **126** | | Within 127 limit |

### Reuse Strategy

1. **S_acc → P reuse**: After softmax, S_acc VGPRs (v[34:65]) are converted in-place to bf16 P operands. No additional storage needed.
2. **V load → repack**: V load destinations (v[88:95]) are partially overwritten during repack. 2 extra temps needed for repack outputs.
3. **ds_read temps recycled**: The 4 VGPRs for ds_read_b128 are reused between GEMM0 and GEMM1.
4. **Q is permanent**: 16 VGPRs (v[66:81]) held for entire kernel lifetime — used every main-loop iteration for GEMM0 B operand.

---

## 6. Q Load Correction

### Problem

Current code loads 8x `buffer_load_b128` = 64 bf16 per thread = 32 VGPRs [src:fmha_fwd_d64_device.hpp — `v4i q_regs[8]`]. This is **double** what CK loads.

CK loads 4x `buffer_load_dwordx4` = 32 bf16 per thread = 16 VGPRs [asm:288,331,376,419].

### Root Cause

The MFMA 32x32x8 B operand needs `short4` = 4 bf16 per lane per call. Each lane contributes values at **one** N position (lane%32) and **two** K sub-positions (k_sub=0,1 from lane/32). With headdim=64 split into 2 k0 iterations of 32, each k0 has 4 kInner steps of 8. Each step needs one short4 = 4 bf16.

Per k0: 4 short4 = 16 bf16 per lane. For 2 k0: 32 bf16 per lane = 16 VGPRs.

The current code loads 64 bf16 because it uses 8 loads at different k_group offsets. This should be 4 loads.

### Correct Q Load Pattern

4x `buffer_load_dwordx4` (each = 4 dwords = 8 bf16):

```cpp
v4i q_regs[4];  // 16 VGPRs total (was 8 = 32 VGPRs)

// Lane L: row = L % 32, k_sub = L / 32
// Q matrix: M x K = seqlen_q x headdim = M0 x 64
// Each lane reads one row of Q at its M position:
//   m_pos = m_tile_offset + warp_id * 32 + (L / 32) * 16 + ... (MFMA A-input mapping)
//   Actually: m_pos comes from the MFMA B-input mapping (TransposedC)

// voffset = m_row * stride_q * sizeof(bf16) + k_base * sizeof(bf16)
// where m_row = q_tile_offset + warp_id * 32 + ...
// k_base iterates as 0, 16, 32, 48 (4 groups of 16 bytes = 8 bf16)
```

CK Q load voffset [asm:254]: `v10 = v2 << 1` where v2 encodes the per-lane byte offset.

CK offsets [asm:288,331,376,419]: `offset:0`, `offset:32`, `offset:64`, `offset:96`
- Each offset = 32 bytes = 16 bf16 = 4 dwords stride along K dimension.
- 4 loads x 32 bytes = 128 bytes = 64 bf16 = full headdim row.

Wait — this is still 64 bf16 total across the row, but each lane only reads at its row position. 4 loads x 8 bf16 = 32 bf16 per lane. The discrepancy is that our current code has `q_regs[8]` (8 loads) while CK has 4 loads.

### Q-to-GEMM0-MFMA Mapping

For MFMA B operand (Q is B due to TransposedC):

```
Lane L:
  k_sub = L / 32         // 0 or 1
  row = L % 32            // M position within warp's 32-row partition

  // B operand = short4 = 4 consecutive K values
  // For kInner step s (0..3):
  //   bf16[0..3] = Q[row, k0*32 + s*8 + k_sub*4 .. k0*32 + s*8 + k_sub*4 + 3]

  // From q_regs[4]:
  //   q_regs[0] = Q[row, 0..7]    (k0=0, s=0: k_sub=0 uses [0:3], k_sub=1 uses [4:7])
  //   q_regs[1] = Q[row, 8..15]   (k0=0, s=1: same split)
  //   q_regs[2] = Q[row, 16..23]  (k0=0, s=2 and k0=0, s=3)
  //   q_regs[3] = Q[row, 24..31]  (... wait, this needs correction)
```

Actually, with 4 loads at offsets 0,32,64,96 bytes = 0,16,32,48 bf16:
```
q_regs[0] = Q[row, 0..7]     (8 bf16, offsets 0-15 bytes, load at offset:0)
q_regs[1] = Q[row, 8..15]    (8 bf16, load at offset:16 from the voffset... )
```

Hmm — CK uses offsets 0,32,64,96 which are **byte offsets**: 0, 32, 64, 96. Each buffer_load_dwordx4 loads 16 bytes. So:
- Load 0: bytes [voffset+0 .. voffset+15] = bf16 positions 0..7
- Load 1: bytes [voffset+32 .. voffset+47] = bf16 positions 16..23
- Load 2: bytes [voffset+64 .. voffset+79] = bf16 positions 32..39
- Load 3: bytes [voffset+96 .. voffset+111] = bf16 positions 48..55

Gap! Positions 8..15, 24..31, 40..47, 56..63 are NOT loaded. Each lane only loads 32 of 64 bf16 along headdim. The missing positions are at `k_sub=1` — covered by the lane at `L+32` (or `L-32`).

**Key insight**: Due to TransposedC, lane L and lane L+32 are at the **same** M row but different k_sub. Lane L (k_sub=0) loads K positions {0..7, 16..23, 32..39, 48..55} and lane L+32 (k_sub=1) loads {8..15, 24..31, 40..47, 56..63}. Together they cover the full 64 bf16.

### Correct voffset Formula

```
lane_id  = tid % 64
k_sub    = lane_id / 32    // 0 or 1
row_in_warp = lane_id % 32

m_row    = q_tile_offset + warp_id * 32 + row_in_warp
k_base   = k_sub * 8       // k_sub=0 starts at K=0, k_sub=1 starts at K=8

voffset  = m_row * stride_q * sizeof(bf16) + k_base * sizeof(bf16)
         = m_row * stride_q * 2 + k_sub * 16
```

Then the 4 loads use immediate offsets 0, 32, 64, 96 to step by 16 bf16 = 32 bytes along K.

### Action Required

In `fmha_fwd_d64_device.hpp`:
1. Change `v4i q_regs[8]` to `v4i q_regs[4]`
2. Change 8 buffer_load_b128 calls to 4
3. Update voffset computation to include k_sub offset
4. Update `gemm0_k0` to index `q_regs[0..3]` instead of `q_regs[0..7]`
5. Saves 16 VGPRs

---

## Appendix: Key Constants

| Constant | Value | Hex | Derivation |
|----------|-------|-----|------------|
| m0 stride | 1088 | 0x440 | 4 warps * 272 bytes/warp-row |
| LDS row stride | 144 bytes | 0x90 | 72 bf16 elements * 2 |
| LDS buf0 | 0 | 0x0 | |
| LDS buf1 | 4608 | 0x1200 | 2304 elements * 2 |
| LDS buf2 | 9216 | 0x2400 | 4608 elements * 2 |
| Single buffer | 2304 elements | — | 4 * 8 * 8 * ~9 (padded) |
| kKPack | 8 | | bf16 elements per LDS pack |
| kPaddedRowStride | 72 elements | | 64 + 8 padding |
| V perm sel0 | 0x01000504 | | low bf16 from each dword |
| V perm sel1 | 0x03020706 | | high bf16 from each dword |
| fp32-to-bf16 sel | 0x07060302 | | truncate fp32 pair to bf16 pair |
| Warp V partition | 1152 bytes | 0x480 | 8 rows * 144 bytes/row |
| SRD flags | 0x00027000 | | gfx942 buffer resource flags |
| O store offsets | 0,16,32,48,64,80,96,112 | | 8 stores x 16 bytes stride |

## Appendix: Assembly Line Number Index

| Feature | CK Assembly Lines |
|---------|-------------------|
| Q load (prolog) | 288, 331, 376, 419 |
| K async copy (prolog) | 532-554 |
| K async copy (main loop) | 672-693 |
| K async copy (prefetch) | 1090-1111 |
| m0 set | 528, 668, 1086 |
| m0 increment | 535,543,550 / 675,682,689 / 1093,1100,1107 |
| V load pair 1 | 722-724 |
| V load pair 2 | 879-881 |
| V perm repack 1 | 901-904 |
| V perm repack 2 | 1066-1069 |
| V ds_write2 1 | 905-906 |
| V ds_write2 2 | 1070-1073 |
| GEMM0 MFMAs | 700-743 |
| GEMM1 MFMAs k1=0 | 987-1029 |
| GEMM1 MFMAs k1=1 | 1036-1146 |
| Softmax | 882-1064 |
| O store (epilog) | 1267-1421 |
| Perm selector init | 625 (s16), 626 (s30), 627 (s34) |
| fp32→bf16 conversion | 1179-1211 |
| Exec mask for O | 1267,1287,...,1407 |
| Barriers | 698, 719, 984, 1082, 1116 |
