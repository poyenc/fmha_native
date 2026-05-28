// Shared test infrastructure for FMHA Google Test binaries.
//
// TestCase defines a parameterized test configuration (problem dimensions
// and feature flags).  make_params() converts a TestCase into a fully
// normalized FmhaParams with derived fields populated.  test_name()
// provides CamelCase names for gtest output.

#pragma once
#include "runner/fmha_params.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Parameterized test configuration.  Fields mirror the YAML manifest
// Fields use 0 to mean "use default / derive".
struct TestCase {
    const char* name;          // CamelCase test name for gtest output
    int b, h, s, d;            // batch, q_heads, seq_len, head_dim
    int kv_heads = 0;          // 0 = same as q_heads
    int kv_seq   = 0;          // 0 = same as seq_len
    int mask     = 0;          // causal mask enable
    int lse      = 0;          // log-sum-exp output enable
    int opt      = 0;          // mask optimization strategy
    std::vector<int> varlen_seqs = {};  // per-batch seq lengths (empty = fixed)
};

// Convert a TestCase into a fully populated FmhaParams.
// Derives kv_heads, gqa, kv_seq_len, and varlen batch/seq from defaults.
inline FmhaParams make_params(const TestCase& tc) {
    FmhaParams p;
    p.batch      = tc.b;
    p.q_heads    = tc.h;
    p.kv_heads   = tc.kv_heads;
    p.gqa        = 1;
    p.seq_len    = tc.s;
    p.kv_seq_len = tc.kv_seq;
    p.head_dim   = tc.d;
    p.hdim_padded = 0;
    p.mask       = tc.mask;
    p.lse        = tc.lse;
    p.opt        = tc.opt;
    p.varlen_seqs = tc.varlen_seqs;

    if (p.kv_heads == 0) p.kv_heads = p.q_heads;
    if (p.q_heads != p.kv_heads) {
        p.gqa = p.q_heads / p.kv_heads;
    }
    if (!p.varlen_seqs.empty()) {
        p.batch = (int)p.varlen_seqs.size();
        int max_s = 0;
        for (int s : p.varlen_seqs) if (s > max_s) max_s = s;
        p.seq_len = max_s;
        if (p.kv_seq_len == 0) p.kv_seq_len = max_s;
    }
    if (p.kv_seq_len == 0) p.kv_seq_len = p.seq_len;
    if (p.hdim_padded == 0) p.hdim_padded = p.head_dim;

    return p;
}

// Return the TestCase name for gtest parameterized test output.
inline std::string test_name(const testing::TestParamInfo<TestCase>& info) {
    return info.param.name;
}
