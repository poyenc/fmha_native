// FMHA test configurations — D64 parameterized cases.
//
// Each entry specifies {name, batch, heads, seq, hdim, kv_heads, kv_seq,
// mask, lse, opt, varlen_seqs}.  Covers smoke tests, feature combos
// (GQA, MQA, causal mask, LSE, asymmetric seq), edge cases (small/odd
// seq lengths), and varlen batching.

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
};
// clang-format on
