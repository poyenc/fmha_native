#pragma once
#include <cstdint>

// CPU reference for V-LDS staging layout (Phase 1 Kernel 5).
//
// Produces the expected LDS byte image (bf16) after V is loaded from DRAM,
// repacked (4x2 -> 2x4 transpose via v_perm_b32), and written to LDS.
//
// GOLDEN-VERIFIED V LDS store formula (from Phase 0.10):
//   k = n % 32  (seqlen_k position within the 32-row chunk)
//   offset_elems(n, d) = buffer_base
//                      + (k / 8) * 576
//                      + (d / 8) * 72
//                      + (d % 8) * 8
//                      + (k % 8)
//   byte = 2 * offset_elems
//   buffer_base(chunk=0) = 2304. chunk = n / 32 selects buffer.
//
// headdim = ROW axis (72 stride = 64 data + 8 kKPack pad)
// seqlen_k = innermost contiguous (k%8 stride 1)
// SingleVSize = (kK1/kKPack)*(kN1/NPerRow)*(PixelsPerRow+kKPack) = 4*8*72 = 2304
//
// V input: row-major [seqlen_k, D=64] bf16.

constexpr int kVLdsRegionElems = 8192; // matches golden dump slot stride

// V LDS element offset for V element (n=seqlen_k, d=headdim), buffer base 0.
// Caller adds buffer_base for the correct chunk (n/32).
inline int v_lds_offset_elems(int n, int d) {
    int k = n % 32;
    return (k / 8) * 576
         + (d / 8) * 72
         + (d % 8) * 8
         + (k % 8);
}

constexpr int kVLdsBufBase = 2304; // buffer_base for chunk 0

// Fill `expected_lds` (uint16 bf16 image, kVLdsRegionElems entries) with the
// staged V tile. `V` is raw bf16 row-major [seqlen_k, D=64], stride_v elements.
// kv_offset = seqlen_k base row of this tile.
void ref_v_lds(const uint16_t* V,
               int stride_v,
               int kv_offset,
               int seqlen_k,
               uint16_t* expected_lds);
