#include "runner/params.hpp"
#include "fused/pipeline.hpp"
#include "fused/op_combine.hpp"   // split-K combine device worker

// ================================================================
// kernel.cpp — the four __global__ launch entry points
// ================================================================
//
// ROLE IN THE PIPELINE
//   These are the only host-launchable symbols of the kernel. Each is a thin
//   shell: it allocates the per-block LDS scratch, decodes blockIdx into the
//   (batch, head, m_tile) coordinates of the tile this block owns, and tail-calls
//   the templated per-block device function fmha_fwd_d64_device<HasMask,IsVarlen>
//   in pipeline.hpp (which is the entire forward pass for one M-tile).
//
// THE FOUR ENTRIES (the 2x2 product of two compile-time switches)
//                       dense (batch)            varlen (group)
//     no-mask  msk0   fmha_fwd_d64_bf16_msk0   fmha_fwd_d64_bf16_msk0_varlen
//     causal   msk1   fmha_fwd_d64_bf16_msk1   fmha_fwd_d64_bf16_msk1_varlen
//   HasMask  (msk0/msk1)  -> false = boundary mask only; true = causal+boundary.
//   IsVarlen (.._varlen)  -> false = dense batch tensors with batch strides;
//                            true  = group/varlen, per-sequence offsets via
//                                    seqstart_q/seqstart_k (no batch stride).
//   Templating both lets every dead branch fold away, so each entry is a fully
//   specialized kernel with no runtime mode dispatch in the hot loop.
//
// GRID MAPPING  dim3 grid(q_heads, m_tiles, batch):
//     blockIdx.x = head_idx   (0 .. nhead_q-1)
//     blockIdx.y = m_tile     (0 .. ceil(seqlen_q/kM0)-1)  [see reversal below]
//     blockIdx.z = batch_idx  (0 .. batch-1)
//   One block computes one kM0(=128)-row M-tile of one (batch, head). Block size
//   is kBlockSize(=256) = 4 warps; the 4 warps tile the 128 query rows (32 each).
//
// CAUSAL M-TILE REVERSAL (msk1 / msk1_varlen ONLY)
//   Causal work per block is LINEAR in the m_tile index: M-tile i processes only
//   the KV-tiles up to its diagonal (~i+1 of them), so low tiles are light and
//   high tiles are heavy. If blocks launched in natural order (m_tile=blockIdx.y),
//   the heaviest tiles would all land in the FINAL block-wave with no light work
//   left to overlap them — a load-imbalance tail. Reversing the index
//   (m_tile = gridDim.y-1-blockIdx.y) launches the HEAVY high-index tiles FIRST,
//   so by the time the wave front reaches the light tiles they backfill the gaps
//   left by finished heavy blocks. This mirrors CK and is the single biggest
//   causal win: MEASURED +7-16% (at one config, occupancy 387->464, kernel cycles
//   787k->646k). Root cause is block scheduling, NOT mask compute.
//   The no-mask entries keep NATURAL order (work is uniform across tiles, so there
//   is no imbalance to fix) — see the contrast at each entry below.
//
// __launch_bounds__(kBlockSize, 4)
//   First arg = max threads per block (kBlockSize=256). Second arg = minimum
//   waves(blocks)-per-CU the compiler must keep schedulable; it caps per-thread
//   VGPR/LDS use so >=4 blocks of this kernel can be resident on a CU at once
//   (an occupancy floor that lets memory latency hide behind other blocks).

// No-mask, dense batch. Natural m_tile order: work is uniform across M-tiles
// (every block walks the full KV length), so there is no load imbalance to fix.
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk0(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    const int m_tile_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<false, false>(params, lds, batch_idx, head_idx, m_tile_idx);
}

// Causal, dense batch. Reverses the m_tile index to launch heavy tiles first
// (see file header: causal load-balance, measured +7-16%).
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk1(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    // Causal load balance: tile i does i+1 KV-tiles, so launch the heavy
    // high-index tiles first (matches CK). Natural order strands all heavy
    // tiles in the final block-wave with no light work to overlap -> tail.
    const int m_tile_idx = gridDim.y - 1 - blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<true, false>(params, lds, batch_idx, head_idx, m_tile_idx);
}

// No-mask, varlen/group mode. Natural m_tile order (uniform work, as in msk0);
// IsVarlen=true switches the device fn to seqstart-based offsets.
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk0_varlen(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    const int m_tile_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<false, true>(params, lds, batch_idx, head_idx, m_tile_idx);
}

// Causal, varlen/group mode. Same heavy-first reversal as msk1, with varlen
// offsets.
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk1_varlen(FmhaFwdParams params) {
    __shared__ char lds[kLdsBytes];
    const int head_idx  = blockIdx.x;
    // Causal load balance: launch heavy high-index tiles first (see msk1).
    const int m_tile_idx = gridDim.y - 1 - blockIdx.y;
    const int batch_idx = blockIdx.z;
    fmha_fwd_d64_device<true, true>(params, lds, batch_idx, head_idx, m_tile_idx);
}

// ================================================================
// Split-K COMBINE entry (fmha_fwd_d64_bf16_combine)
// ================================================================
//   The second pass of split-K: reweights the G per-split fp32 partials into the
//   single global-softmax output and stores the final BF16 O. Unlike the four
//   forward entries above, this needs NO LDS pipeline — it is a cheap row-wise
//   reduction. The grid mirrors the forward kernel: dim3(nhead_q, m_tiles, batch),
//   so the same blockIdx decode applies (h=x, m_tile=y, b=z). See op_combine.hpp.
__global__ void __launch_bounds__(kBlockSize)
fmha_fwd_d64_bf16_combine(FmhaFwdCombineParams params) {
    combine_split(params, blockIdx.z, blockIdx.x, blockIdx.y);
}

// ================================================================
// Split-K PRODUCER entries (fmha_fwd_d64_bf16_msk{0,1}_split)
// ================================================================
//   The FIRST pass of split-K. These run the SAME per-block forward pass as the
//   four entries above (fmha_fwd_d64_device, the IsSplit=true specialization), but
//   each block walks ONLY a disjoint sub-range of the KV axis — its "split" — and,
//   instead of bf16-truncating O straight to the final tensor, writes a NORMALIZED
//   fp32 partial output O_g + a per-row natural-log LSE_g into the split-major
//   "scratch" staging buffer. The combine entry above then folds the G partials
//   into the final O. (See FmhaFwdSplitParams in params.hpp + epilog_store_split.)
//
//   Setting IsSplit=true (the third template arg) switches fmha_fwd_d64_device to:
//     * narrow the per-M-tile KV-tile range to this split's slice
//       [split_idx*T, min((split_idx+1)*T, full)) with T = ceil(full/num_splits);
//     * route the epilogue through epilog_store_split (wide fp32 scratch writes,
//       NOT the bf16 epilog_store), including the EMPTY-split sentinel (LSE=-inf,
//       O=0) for a slice past the tile count or wholly in the masked-future region.
//   The `if constexpr (IsSplit)` guards in pipeline.hpp keep the four non-split
//   entries byte-identical (their IsSplit=false instances never see this code), and
//   instantiating these two real globals is what forces the IsSplit=true body to
//   compile — so the standalone Task-5 forced instantiation is now redundant and
//   has been removed (these globals subsume it).
//
//   SPLIT GRID  dim3 grid(q_heads, m_tiles, batch*num_splits):
//     blockIdx.x = head_idx
//     blockIdx.y = m_tile  (natural for msk0; reversed for msk1 — see below)
//     blockIdx.z = batch*num_splits, packing BOTH batch and split on one axis:
//                    b         = blockIdx.z / num_splits
//                    split_idx = blockIdx.z % num_splits
//   The device fn recovers B = gridDim.z / num_splits for the scratch base, so the
//   z-axis MUST be sized batch*num_splits and num_splits MUST be passed through.

// No-mask split producer. Natural m_tile order (uniform work across M-tiles, as in
// the non-split msk0). z-axis decodes to (batch, split); IsSplit=true writes the
// fp32 partial plane for split_idx into scratch.
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk0_split(FmhaFwdSplitParams sp) {
    __shared__ char lds[kLdsBytes];
    const int head_idx   = blockIdx.x;
    const int m_tile_idx = blockIdx.y;                  // natural order (no mask)
    const int b          = blockIdx.z / sp.num_splits;  // batch lives in z / G
    const int split_idx  = blockIdx.z % sp.num_splits;  // split lives in z % G
    fmha_fwd_d64_device<false, false, true>(
        sp.base, lds, b, head_idx, m_tile_idx,
        sp.scratch_o, sp.scratch_lse, sp.num_splits, split_idx);
}

// Causal split producer. Keeps the IDENTICAL heavy-first M-tile reversal as the
// non-split msk1 entry (m_tile = gridDim.y-1-blockIdx.y) so the causal load-balance
// win carries over to split-K. The partial is keyed by the ABSOLUTE query row, so
// the reversal only changes WHICH block runs WHEN, not where each row's partial
// lands in scratch — the test's per-(b,h,row) comparison is reversal-agnostic.
__global__ void __launch_bounds__(kBlockSize, 4)
fmha_fwd_d64_bf16_msk1_split(FmhaFwdSplitParams sp) {
    __shared__ char lds[kLdsBytes];
    const int head_idx   = blockIdx.x;
    // Causal load balance: launch heavy high-index tiles first (see msk1 header).
    const int m_tile_idx = gridDim.y - 1 - blockIdx.y;
    const int b          = blockIdx.z / sp.num_splits;  // batch lives in z / G
    const int split_idx  = blockIdx.z % sp.num_splits;  // split lives in z % G
    fmha_fwd_d64_device<true, false, true>(
        sp.base, lds, b, head_idx, m_tile_idx,
        sp.scratch_o, sp.scratch_lse, sp.num_splits, split_idx);
}
