// FMHA test configurations — D64 parameterized cases.
//
// kAllFull is the master problem-size matrix shared by BOTH full-kernel suites
// (test_fmha_fwd_d64.cpp and test_fmha_gpu_ref.cpp) via INSTANTIATE_TEST_SUITE_P
// + test_name(). Add a row here and both suites pick it up automatically.
//
// Each entry is a TestCase (see test_params.hpp): {name, batch, heads, seq,
// hdim, kv_heads, kv_seq, mask, lse, opt, varlen_seqs}.  Covers smoke tests,
// feature combos (GQA, MQA, causal mask, LSE, asymmetric seq), edge cases
// (small/odd seq lengths), and varlen batching. kv_heads/kv_seq = 0 means
// "derive from q" (handled by make_params). The trailing brace list, when
// present, gives per-batch sequence lengths for varlen cases.

#pragma once
#include "test_params.hpp"
#include <vector>

// clang-format off
inline const std::vector<TestCase> kAllFull = {
    // === D64 Smoke ===
    {"D64Smoke1", 2, 2, 512, 64, 0, 0, 0, 0, 0},
    {"D64Smoke2", 1, 1, 256, 64, 0, 0, 0, 0, 0},
    {"D64Smoke3", 2, 4, 1024, 64, 0, 0, 0, 0, 0},
    {"D64Smoke4", 1, 8, 2048, 64, 0, 0, 0, 0, 0},
    // === D64 Feature Combos ===
    {"D64MaskLse", 2, 4, 1024, 64, 0, 0, 1, 1, 0},
    {"D64GqaMask", 2, 8, 1024, 64, 2, 0, 1, 0, 0},
    {"D64MqaMaskLse", 2, 8, 1024, 64, 1, 0, 1, 1, 0},
    {"D64AsymSqLtSkvMaskLse", 1, 8, 512, 64, 0, 1024, 1, 1, 0},
    {"D64AsymSqGtSkvMask", 1, 4, 1024, 64, 0, 512, 1, 0, 0},
    {"D64KitchenSink", 2, 8, 1024, 64, 2, 2048, 1, 1, 0},
    // === D64 Edge Cases ===
    {"D64EdgeS300MaskLse", 1, 2, 300, 64, 0, 0, 1, 1, 0},
    {"D64EdgeS300GqaMask", 1, 4, 300, 64, 2, 0, 1, 0, 0},
    {"D64EdgeS255", 1, 2, 255, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS255MaskLse", 1, 2, 255, 64, 0, 0, 1, 1, 0},
    {"D64EdgeS33MaskLse", 1, 2, 33, 64, 0, 0, 1, 1, 0},
    {"D64EdgeS33MqaMask", 1, 2, 33, 64, 1, 0, 1, 0, 0},
    {"D64EdgeS17MaskLse", 1, 1, 17, 64, 0, 0, 1, 1, 0},
    {"D64EdgeS17GqaMask", 1, 4, 17, 64, 2, 0, 1, 0, 0},
    {"D64EdgeS16", 1, 1, 16, 64, 0, 0, 0, 0, 0},
    {"D64EdgeScale", 1, 1, 8192, 64, 0, 0, 1, 1, 0},
    // === D64 Varlen ===
    {"D64VarlenMask", 4, 4, 1024, 64, 0, 0, 1, 0, 0, {512, 256, 1024, 128}},
    {"D64VarlenGqaMask", 2, 8, 512, 64, 2, 0, 1, 0, 0, {512, 256}},
    {"D64VarlenEdgeMaskLse", 2, 2, 300, 64, 0, 0, 1, 1, 0, {300, 33}},
    {"D64VarlenMqaMaskLse", 4, 8, 1024, 64, 1, 0, 1, 1, 0, {1024, 512, 256, 128}},
    // === D64 Feature Variants ===
    {"D64MaskOnly", 2, 4, 1024, 64, 0, 0, 1, 0, 0},
    {"D64LseOnly", 2, 4, 1024, 64, 0, 0, 0, 1, 0},
    {"D64GqaAsymMaskLse", 1, 8, 512, 64, 2, 1024, 1, 1, 0},
    {"D64MqaAsymMask", 1, 8, 1024, 64, 1, 512, 1, 0, 0},
    // === D64 Boundary/Misc ===
    {"D64EdgeS300", 1, 2, 300, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS256", 1, 2, 256, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS33", 1, 2, 33, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS17", 1, 1, 17, 64, 0, 0, 0, 0, 0},
    {"D64EdgeKvseq0", 1, 1, 256, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS1", 1, 1, 1, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS2Mask", 1, 2, 2, 64, 0, 0, 1, 0, 0},
    {"D64EdgeS31", 1, 2, 31, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS32", 1, 2, 32, 64, 0, 0, 1, 0, 0},
    {"D64EdgeS64MaskLse", 1, 4, 64, 64, 0, 0, 1, 1, 0},
    {"D64EdgeS257Mask", 1, 2, 257, 64, 0, 0, 1, 0, 0},
    {"D64H3Mask", 2, 3, 512, 64, 0, 0, 1, 0, 0},
    {"D64H6GqaMask", 1, 6, 1024, 64, 3, 0, 1, 0, 0},
    {"D64GqaRatio8MaskLse", 1, 16, 512, 64, 2, 0, 1, 1, 0},
    {"D64GqaNomask", 2, 8, 1024, 64, 2, 0, 0, 0, 0},
    {"D64VarlenNomask", 2, 4, 512, 64, 0, 0, 0, 0, 0, {512, 256}},
    {"D64EdgeKvseq1", 1, 2, 256, 64, 0, 1, 0, 0, 0},
    {"D64AsymLongContext", 1, 4, 64, 64, 0, 4096, 1, 1, 0},
    {"D64VarlenTinyMaskLse", 4, 2, 512, 64, 0, 0, 1, 1, 0, {1, 512, 32, 256}},
    {"D64B8Mask", 8, 4, 512, 64, 0, 0, 1, 0, 0},
    // === D64 Tile Boundary ===
    {"D64EdgeS128", 1, 2, 128, 64, 0, 0, 0, 0, 0},
    {"D64EdgeS128MaskLse", 1, 4, 128, 64, 0, 0, 1, 1, 0},
    {"D64EdgeS65MaskLse", 1, 2, 65, 64, 0, 0, 1, 1, 0},
    {"D64EdgeAsymS128Sk65Mask", 1, 2, 128, 64, 0, 65, 1, 0, 0},
    // === D64 Feature Combos (gaps) ===
    {"D64GqaLseNomask", 2, 8, 512, 64, 2, 0, 0, 1, 0},
    {"D64VarlenGqaMaskLse", 2, 8, 512, 64, 2, 0, 1, 1, 0, {512, 256}},
    {"D64VarlenLseNomask", 2, 4, 512, 64, 0, 0, 0, 1, 0, {512, 256}},
    {"D64VarlenAsymMask", 2, 4, 256, 64, 0, 512, 1, 0, 0, {256, 128}},
    // === D64 Causal full-tile-skip coverage (softmax_mask HasMask fast path) ===
    // These stress the below-diagonal full-tile skip added to softmax_mask:
    // tiles fully below the diagonal must skip masking, the diagonal/boundary
    // tile must still mask. kM0=128, kN0=64.
    // Deep square: M-tile 5 (base 640) skips kv tiles 0..576, masks tile 640.
    // kv_offset=576 hits kv_offset+kN0=640 == base+1=641 minus one (the +1 edge).
    {"D64CausalDeepSquare", 1, 2, 768, 64, 0, 0, 1, 0, 0},
    {"D64CausalDeepSquareLse", 1, 2, 768, 64, 0, 0, 1, 1, 0},
    // Ragged tail (768+64=832 = 6*128+64): half-empty last M-tile, like S=40000.
    {"D64CausalRaggedTail", 1, 2, 832, 64, 0, 0, 1, 0, 0},
    // Deep ragged multi-tile (1100 = 8*128+76): many below-diag tiles + ragged.
    {"D64CausalDeepRagged", 1, 2, 1100, 64, 0, 0, 1, 0, 0},
    // Tile-aligned shift (mask_shift = Sk-Sq = 128 = kM0): diagonal on M-tile edge.
    {"D64CausalShift128", 1, 2, 256, 64, 0, 384, 1, 0, 0},
    // N-tile-aligned shift (mask_shift = 64 = kN0): diagonal on N-tile edge.
    {"D64CausalShift64", 1, 2, 256, 64, 0, 320, 1, 0, 0},
    // GQA + deep ragged causal: exercises skip path under head grouping.
    {"D64CausalGqaDeepRagged", 1, 8, 900, 64, 2, 0, 1, 0, 0},
};
// clang-format on
