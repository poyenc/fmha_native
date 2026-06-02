// CPU reference FMHA and verification — see cpu_ref.hpp for API documentation.

#include "cpu_ref.hpp"
#include "fmha_params.hpp"
#include "buffers.hpp"
#include "bf16_utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

void cpu_ref_split(const FmhaParams& p, const FmhaBuffers& bufs,
                   int b, int hq, int i, int kv_start, int kv_end,
                   float* o_row, float* lse_out) {
    const int Sq   = p.seq_len;
    const int Skv  = p.kv_seq_len;
    const int Dlog = p.head_dim;
    const int Dpad = p.hdim_dispatch();
    const bool varlen_mode = !p.varlen_seqs.empty();
    const uint32_t total_sl = bufs.total_seqlen;
    const float scalar = p.scalar();

    const uint16_t* h_Q = bufs.h_Q.data();
    const uint16_t* h_K = bufs.h_K.data();
    const uint16_t* h_V = bufs.h_V.data();

    // Per-batch (varlen) or global (dense) sequence lengths.
    const int Sq_b  = varlen_mode ? p.varlen_seqs[b] : Sq;
    const int Skv_b = varlen_mode ? p.varlen_seqs[b] : Skv;

    // Recompute the same Q/K/V base offsets cpu_ref_verify uses for this row.
    // (Recomputing per call is intentional — see cpu_ref.hpp / the split-K
    // plan; the cost is negligible next to the O(Skv*Dlog) inner work.)
    const int hkv = hq / p.gqa;
    size_t Q_off, K_base, V_base;
    if (varlen_mode) {
        // Reconstruct the cumulative varlen offset for batch b.
        uint32_t vl_off = 0;
        for (int bb = 0; bb < b; bb++) vl_off += p.varlen_seqs[bb];
        Q_off  = ((size_t)hq  * total_sl + vl_off + i) * Dpad;
        K_base = ((size_t)hkv * total_sl + vl_off)     * Dpad;
        V_base = K_base;
    } else {
        Q_off  = (((size_t)b * p.q_heads  + hq ) * Sq  + i) * Dpad;
        K_base = ((size_t)b * p.kv_heads + hkv) * Skv * Dpad;
        V_base = ((size_t)b * p.kv_heads + hkv) * Skv * Dpad;
    }

    // Clamp the requested range to this batch's valid KV span.
    if (kv_start < 0)      kv_start = 0;
    if (kv_end   > Skv_b)  kv_end   = Skv_b;

    // Empty range: nothing to attend to -> zero output, -inf LSE.
    if (kv_start >= kv_end) {
        for (int d = 0; d < Dlog; d++) o_row[d] = 0.0f;
        if (lse_out) *lse_out = -INFINITY;
        return;
    }

    // Scores S = (Q . K^T) * scalar over the sub-range, with causal mask.
    // scalar is the PLAIN 1/sqrt(head_dim); softmax below is natural-e (expf).
    // The base-2 (log2(e)) trick lives only in the GPU kernel; both produce
    // the same probabilities.
    std::vector<float> S(kv_end - kv_start);
    for (int j = kv_start; j < kv_end; j++) {
        float s = 0;
        for (int d = 0; d < Dlog; d++) {
            float q = bf16_to_float(h_Q[Q_off + d]);
            float k = bf16_to_float(h_K[K_base + (size_t)j * Dpad + d]);
            s += q * k;
        }
        float sj = s * scalar;
        // Right-aligned causal mask (FA-2 / CK convention): query row i may
        // attend key j only up to i + (Skv_b - Sq_b); the (Skv_b - Sq_b) shift
        // aligns the diagonal when k and q lengths differ.  Masked positions
        // are set to -inf so exp() -> 0.
        if (p.mask && j > i + (Skv_b - Sq_b)) sj = -INFINITY;
        S[j - kv_start] = sj;
    }

    const int n = kv_end - kv_start;

    // Numerically-stable softmax over the sub-range: subtract local max.
    // local_max is the max of the ALREADY-SCALED S (see header A1 note).
    float local_max = S[0];
    for (int j = 1; j < n; j++) if (S[j] > local_max) local_max = S[j];

    if (local_max == -INFINITY) {
        // Entire sub-range masked out (can happen with causal masking):
        // zero output and -inf LSE, matching the empty-range convention.
        for (int d = 0; d < Dlog; d++) o_row[d] = 0.0f;
        if (lse_out) *lse_out = -INFINITY;
        return;
    }

    std::vector<float> P(n);
    float local_sum = 0;
    for (int j = 0; j < n; j++) {
        P[j] = (std::isinf(S[j]) && S[j] < 0) ? 0.0f : expf(S[j] - local_max);
        local_sum += P[j];
    }
    float inv_sum = 1.0f / local_sum;
    for (int j = 0; j < n; j++) {
        P[j] *= inv_sum;
        // Emulate the kernel data path: P is truncated to BF16 and widened
        // back before the P.V GEMM, so the reference sees the same rounding
        // as the shader.
        P[j] = bf16_to_float(float_to_bf16(P[j]));
    }

    // P.V accumulation over the sub-range -> fp32 output (normalized by THIS
    // range's sum, since P was already divided by local_sum above).
    for (int d = 0; d < Dlog; d++) {
        float o = 0;
        for (int j = kv_start; j < kv_end; j++) {
            float v = bf16_to_float(h_V[V_base + (size_t)j * Dpad + d]);
            o += P[j - kv_start] * v;
        }
        o_row[d] = o;
    }

    // Natural-log LSE for this range.  NO extra `* scalar` here — the scale is
    // already baked into local_max (A1: double-applying the scale is a known
    // bug).  Matches gpu_ref.cpp (logf(sum_exp)+max_s) and op_epilog.hpp.
    if (lse_out) *lse_out = logf(local_sum) + local_max;
}

CpuRefResult cpu_ref_verify(const FmhaParams& p, const FmhaBuffers& bufs) {
    const int B    = p.batch;
    const int Hq   = p.q_heads;
    const int Sq   = p.seq_len;
    const int Skv  = p.kv_seq_len;
    const int Dlog = p.head_dim;
    const int Dpad = p.hdim_dispatch();
    const bool varlen_mode = !p.varlen_seqs.empty();
    const uint32_t total_sl = bufs.total_seqlen;

    // Q/K/V scoring now lives in cpu_ref_split; this routine only reads the
    // kernel output (h_O) for comparison.
    const float tol_abs = 0.001f;
    const float tol_rel = 0.05f;
    const double tol_cos = 0.99995;

    const size_t total_rows = (size_t)B * Hq * Sq;
    printf("\n=== Verify (CPU reference, all %zu rows x %d dims) ===\n",
           total_rows, Dlog);

    const uint16_t* h_O = bufs.h_O.data();

    double max_abs = 0, max_rel = 0, min_cos = 1.0, sum_cos = 0;
    size_t mismatch = 0, rel_counted = 0, total_elems = 0, nonfinite = 0;
    size_t cos_rows = 0, cos_fail = 0;
    int cos_wb = -1, cos_wh = -1, cos_wi = -1;
    double cos_worst = 1.0;
    int worst_b = -1, worst_h = -1, worst_i = -1, worst_d = -1;
    float worst_ref = 0, worst_kern = 0;

    std::vector<float> row0_ref(Dlog, 0), row0_kern(Dlog, 0);

    const bool dump_worst_rows = []() { const char* e = getenv("WORST_ROWS"); return e && e[0]=='1'; }();
    std::vector<double> per_row_max(dump_worst_rows ? (size_t)B * Hq * Sq : 0, 0.0);

    std::vector<int> probe_qs;
    if (const char* e = getenv("PROBE_DUMP_ROWS")) {
        const char *s = e;
        while (*s) { probe_qs.push_back(atoi(s)); while (*s && *s != ',') ++s; if (*s) ++s; }
    }
    std::vector<std::vector<float>> probe_ref(probe_qs.size(), std::vector<float>(Dlog, 0));
    std::vector<std::vector<float>> probe_kern(probe_qs.size(), std::vector<float>(Dlog, 0));

    auto get_sq = [&](int b) -> int {
        return varlen_mode ? p.varlen_seqs[b] : Sq;
    };
    auto get_skv = [&](int b) -> int {
        return varlen_mode ? p.varlen_seqs[b] : Skv;
    };
    std::vector<uint32_t> vl_offsets;
    if (varlen_mode) {
        vl_offsets.resize(B + 1, 0);
        for (int i = 0; i < B; i++)
            vl_offsets[i + 1] = vl_offsets[i] + p.varlen_seqs[i];
    }

    #pragma omp parallel
    {
        std::vector<float> o_row(Dlog);
        double t_max_abs = 0, t_max_rel = 0, t_min_cos = 1.0, t_sum_cos = 0;
        size_t t_mismatch = 0, t_rel_counted = 0, t_total = 0, t_nonfinite = 0;
        size_t t_cos_rows = 0, t_cos_fail = 0;
        int t_wb = -1, t_wh = -1, t_wi = -1, t_wd = -1;
        float t_wref = 0, t_wkern = 0;
        int t_cwb = -1, t_cwh = -1, t_cwi = -1;
        double t_cos_worst = 1.0;

        #pragma omp for collapse(2) schedule(dynamic, 16)
        for (int b = 0; b < B; b++)
        for (int hq = 0; hq < Hq; hq++)
        for (int i = 0; i < get_sq(b); i++) {
            const int Skv_b = get_skv(b);
            // O offset for reading the kernel output (mirrors the Q offset).
            size_t O_off;
            if (varlen_mode) {
                O_off = ((size_t)hq * total_sl + vl_offsets[b] + i) * Dpad;
            } else {
                O_off = (((size_t)b * Hq + hq) * Sq + i) * Dpad;
            }

            // Per-row attention compute over the FULL KV range [0, Skv_b).
            // The score -> softmax -> P.V math now lives in cpu_ref_split; this
            // call reproduces exactly the inlined reference that preceded the
            // split-K refactor (lse_out=nullptr: LSE is not needed here).
            cpu_ref_split(p, bufs, b, hq, i, 0, Skv_b, o_row.data(), nullptr);

            const bool save_row0 = (b == 0 && hq == 0 && i == 0);
            double dot_rk = 0, norm_r = 0, norm_k = 0;
            for (int d = 0; d < Dlog; d++) {
                float o_ref = o_row[d];
                float o_kern = bf16_to_float(h_O[O_off + d]);
                dot_rk += (double)o_ref * o_kern;
                norm_r += (double)o_ref * o_ref;
                norm_k += (double)o_kern * o_kern;
                float abs_err = fabsf(o_ref - o_kern);
                if (!std::isfinite(abs_err)) {
                    abs_err = INFINITY;
                    t_nonfinite++;
                }
                if (abs_err > t_max_abs) {
                    t_max_abs = abs_err;
                    t_wb = b; t_wh = hq; t_wi = i; t_wd = d;
                    t_wref = o_ref; t_wkern = o_kern;
                }
                if (fabsf(o_ref) >= 0.01f) {
                    float rel = abs_err / fabsf(o_ref);
                    if (rel > t_max_rel) t_max_rel = rel;
                    t_rel_counted++;
                }
                if (abs_err > tol_abs) t_mismatch++;
                t_total++;
                if (save_row0) {
                    row0_ref[d]  = o_ref;
                    row0_kern[d] = o_kern;
                }
                if (dump_worst_rows) {
                    size_t row_idx = ((size_t)b * Hq + hq) * Sq + i;
                    if (abs_err > per_row_max[row_idx]) per_row_max[row_idx] = abs_err;
                }
                if (b == 0 && hq == 0) {
                    for (size_t pi = 0; pi < probe_qs.size(); ++pi) {
                        if (i == probe_qs[pi]) {
                            probe_ref[pi][d]  = o_ref;
                            probe_kern[pi][d] = o_kern;
                        }
                    }
                }
            }
            double denom = sqrt(norm_r) * sqrt(norm_k);
            if (denom > 0) {
                double cs = dot_rk / denom;
                t_sum_cos += cs;
                t_cos_rows++;
                if (cs < tol_cos) t_cos_fail++;
                if (cs < t_cos_worst) {
                    t_cos_worst = cs;
                    t_cwb = b; t_cwh = hq; t_cwi = i;
                }
            }
        }

        #pragma omp critical
        {
            if (t_max_abs > max_abs) {
                max_abs   = t_max_abs;
                worst_b   = t_wb;   worst_h    = t_wh;
                worst_i   = t_wi;   worst_d    = t_wd;
                worst_ref = t_wref; worst_kern = t_wkern;
            }
            if (t_max_rel > max_rel) max_rel = t_max_rel;
            if (t_cos_worst < cos_worst) {
                cos_worst = t_cos_worst;
                cos_wb = t_cwb; cos_wh = t_cwh; cos_wi = t_cwi;
            }
            sum_cos     += t_sum_cos;
            cos_rows    += t_cos_rows;
            cos_fail    += t_cos_fail;
            mismatch    += t_mismatch;
            rel_counted += t_rel_counted;
            total_elems += t_total;
            nonfinite   += t_nonfinite;
        }
    }

    double mean_cos = cos_rows > 0 ? sum_cos / cos_rows : 0;
    printf("Checked %zu elems  (tol_abs=%g  tol_rel=%g  tol_cos=%.5f)\n", total_elems, tol_abs, tol_rel, tol_cos);
    printf("max abs err: %.6f   max rel err: %.6f (%zu/%zu elems with |ref|>=tol)   mismatches: %zu/%zu   nonfinite: %zu/%zu\n",
           max_abs, max_rel, rel_counted, total_elems, mismatch, total_elems, nonfinite, total_elems);
    printf("cosine sim:  min=%.8f  mean=%.8f  fail=%zu/%zu rows\n",
           cos_worst, mean_cos, cos_fail, cos_rows);
    if (worst_b >= 0)
        printf("Worst abs at (b=%d, h=%d, q=%d, d=%d): kernel=%+.5f  ref=%+.5f\n",
               worst_b, worst_h, worst_i, worst_d, worst_kern, worst_ref);
    if (cos_wb >= 0)
        printf("Worst cos at (b=%d, h=%d, q=%d): %.8f\n",
               cos_wb, cos_wh, cos_wi, cos_worst);

    printf("Row 0 (b=0,h=0,q=0) first 8 dims [kernel | ref | abs_err]:\n");
    for (int d = 0; d < 8 && d < Dlog; d++)
        printf("  d=%d  %+.5f | %+.5f | %.5f\n",
               d, row0_kern[d], row0_ref[d], fabsf(row0_kern[d] - row0_ref[d]));

    bool rel_fail = (rel_counted > 0 && max_rel > tol_rel);
    bool pass = (mismatch == 0 && nonfinite == 0 && cos_fail == 0 && !rel_fail);
    if (pass)
        printf("OK: kernel matches CPU reference within tolerance\n");
    else {
        printf("FAIL:");
        if (mismatch > 0 || nonfinite > 0)
            printf(" %zu elements exceed abs tolerance", mismatch);
        if (nonfinite > 0)
            printf(" (%zu NaN/Inf)", nonfinite);
        if (rel_fail)
            printf(" max rel err %.4f exceeds threshold %.4f", max_rel, tol_rel);
        if (cos_fail > 0)
            printf(" %zu rows below cosine threshold", cos_fail);
        printf("\n");
    }

    for (size_t pi = 0; pi < probe_qs.size(); ++pi) {
        int q = probe_qs[pi];
        printf("PROBE_DUMP (b=0, h=0, q=%d) -- d  | kernel    | ref       | abs_err\n", q);
        for (int d = 0; d < Dlog; ++d) {
            float ae = fabsf(probe_kern[pi][d] - probe_ref[pi][d]);
            printf("  d=%2d  %+10.5f | %+10.5f | %.5f\n", d, probe_kern[pi][d], probe_ref[pi][d], ae);
        }
    }

    if (dump_worst_rows && !per_row_max.empty()) {
        std::vector<size_t> idx(per_row_max.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(20, idx.size()),
                          idx.end(), [&](size_t a, size_t bb){ return per_row_max[a] > per_row_max[bb]; });
        printf("Top-20 worst rows by max abs err:\n");
        const size_t topk = std::min<size_t>(20, idx.size());
        for (size_t k = 0; k < topk; ++k) {
            size_t r = idx[k];
            int b_ = r / (Hq * Sq);
            int h_ = (r / Sq) % Hq;
            int q_ = r % Sq;
            int wave_ = q_ / 32;
            printf("  rank=%2zu  (b=%d, h=%d, q=%3d, wave=%d)  max abs err=%.6f\n",
                   k, b_, h_, q_, wave_, per_row_max[r]);
        }
    }

    CpuRefResult result;
    result.pass = pass;
    result.max_abs = max_abs;
    result.max_rel = max_rel;
    result.min_cos = cos_worst;
    result.mean_cos = mean_cos;
    result.mismatch = mismatch;
    result.nonfinite = nonfinite;
    result.total_elems = total_elems;
    result.cos_fail = cos_fail;
    result.cos_rows = cos_rows;
    result.worst_b = worst_b;
    result.worst_h = worst_h;
    result.worst_i = worst_i;
    result.worst_d = worst_d;
    result.worst_ref = worst_ref;
    result.worst_kern = worst_kern;
    result.cos_wb = cos_wb;
    result.cos_wh = cos_wh;
    result.cos_wi = cos_wi;
    return result;
}
