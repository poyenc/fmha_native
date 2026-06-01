#pragma once
#include <cstdint>

// ===== REFERENCE ORACLE for the K-LDS component (src/components/k_lds.hpp) ===
// The CPU truth-image for the staged K tile: it writes each in-range K element
// to the exact LDS slot the GPU's swizzled copy should land it in, using the
// same offset formula. The test then byte-compares the GPU's dumped LDS region
// against this image. Test-only; never compiled into src/fused/.
// ============================================================================
//
// CPU reference for the K-LDS staging layout (Phase 1 Kernel 1).
//
// Produces the expected LDS byte image (bf16) after K is staged from DRAM
// into LDS in CK's GOLDEN-VERIFIED layout:
//
//   offset_elems(j,d) = (j%4)*136 + ((j/4)%4)*32 + (j/16)*544
//                     + (d%32)   + (d/32)*2304
//   byte_offset       = 2 * offset_elems    (single buffer, base 0)
//
// j = seqlen_k row within the 64-row tile, d = headdim column [0,64).
// Rows with (kv_offset + j) >= seqlen_k are zero-filled (CK OOB behavior).
//
// K input is row-major [seqlen_k, D] bf16 (raw uint16). The reference reads
// K[(kv_offset + j) * stride_k + d] for in-range rows.
//
// Verified region span: max offset_elems = 6775 for a full 64x64 tile, so the
// expected image must hold at least kLdsRegionElems (8192) bf16 elements to be
// directly comparable against the golden dump's per-buffer K region.

constexpr int kKLdsRegionElems = 8192; // matches golden dump slot stride

// LDS element offset for K element (j seqlen, d headdim), single buffer base 0.
inline int k_lds_offset_elems(int j, int d) {
    return (j % 4) * 136
         + ((j / 4) % 4) * 32
         + (j / 16) * 544
         + (d % 32)
         + (d / 32) * 2304;
}

// Fill `expected_lds` (uint16 bf16 image, kKLdsRegionElems entries) with the
// staged K tile. `K` is the raw bf16 (uint16) K matrix, row-major, row stride
// `stride_k` elements. `kv_offset` is the seqlen_k base row of this tile.
// Out-of-range rows (>= seqlen_k) are left as zero.
void ref_k_lds(const uint16_t* K,
               int stride_k,
               int kv_offset,
               int seqlen_k,
               uint16_t* expected_lds);
