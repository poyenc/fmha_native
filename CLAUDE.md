# CLAUDE.md — fmha_native project index

Quick orientation for an agent (or engineer) working in this repo. For build/test details see
`README.md`; for internals see `docs/ARCHITECTURE.md`.

## Project

Hand-written HIP **D64 BF16 fused multi-head attention (FMHA) forward kernel** for AMD
**gfx942 (MI300X, CDNA3)**. Goal: **ISA- and performance-parity with Composable Kernel (CK)**.
Forward pass only; head dim 64; BF16 in / FP32 accumulate; dense + varlen; optional causal
mask; optional LSE.

## Layout

- `src/fused/` — the production kernel: `kernel.cpp` (4 `__global__` entries: dense/varlen ×
  nomask/causal), `pipeline.hpp` (per-block forward pass), `op_lds.hpp` / `op_gemm.hpp` /
  `op_softmax.hpp` / `op_epilog.hpp` (device-op helpers).
- `src/components/` — 7 standalone building-block kernels, **test-only**, NOT included by
  `src/fused/`.
- `src/components_ref/` — CPU/GPU reference oracles for the components.
- `src/runner/` — shared infra: `params.hpp` (`FmhaFwdParams` + tile constants), `buffers`,
  `cpu_ref`, `gpu_ref`, `bf16_utils`.
- `tests/` — GoogleTest: full-kernel suite, gpu-ref suite, 7 component-vs-golden suites.
- `scripts/` — `run-gates.sh`, `run-benchmark.sh`, `common.sh`.
- `docs/ARCHITECTURE.md` — deep dive (pipeline, layouts, perf, CK methodology).
- `asm_compare/` — CK reference assembly for ISA diffing.

## Build / test / benchmark

Builds and tests run on gfx942 via ROCm; development used a container (`docker exec <ctr>
bash -c "cd /root/workspace && ..."`) with the repo bind-mounted.

- Build: `cmake -B build -DGPU_TARGET=gfx942 && cmake --build build -j`
- Gate (build + 67 fused + 49 standalone tests + extract `.s`): `scripts/run-gates.sh`
  → expect `[  PASSED  ] 67 tests.` + seven `[  PASSED  ] 7 tests.` + `ALL REQUESTED GATES PASSED`.
- Benchmark: `scripts/run-benchmark.sh [--mask 0|1]` (8-config sweep; TFLOPS is avg-based).
- Single run: `./build/bench_fmha_fwd -b 1 -h 16 -s 8192 -d 64 --mask 1`.

## Conventions & gotchas

- **Release / `-O3` is mandatory.** The kernel depends on full unrolling/inlining. `CMakeLists.txt`
  sets `CMAKE_BUILD_TYPE=Release` by default; do not build debug/`-O0` (it miscompiles).
- **Load-bearing `-mllvm` flags** on the kernel target (`CMakeLists.txt`): `-amdgpu-early-inline-all=true`,
  `-amdgpu-function-calls=false`, `--lsr-drop-solution=1`, `-enable-post-misched=0`, plus
  `--save-temps` (emits `build/kernel-hip-amdgcn-amd-amdhsa-gfx942.s`),
  `-fgpu-flush-denormals-to-zero`, `-fbracket-depth=1024`. Changing these changes codegen — do
  not remove without re-checking the gate and the asm.
- **Component tests need golden dumps** at `/tmp/fmha-native-isa-match/golden/{full,partial}`;
  the gate fails if absent.
- **Containers don't share `/tmp`** — stage golden/data per container.
- **This repo is intentionally heavily commented** (teaching-grade, for handover). This OVERRIDES
  the usual "minimal comments" default — do NOT strip the explanatory comments in `src/`.
- **base-2 softmax**: `params.scale` is pre-multiplied by `log2(e)`; `__builtin_amdgcn_exp2f` →
  `v_exp_f32` (2^x) and `__builtin_amdgcn_logf` → `v_log_f32` (LOG2, not natural log). bf16 cast
  is truncation (`u >> 16`), not round-to-nearest.
- **Causal perf**: the masked kernel entries reverse the M-tile (`gridDim.y-1-blockIdx.y`) for
  load balance; the OR-scan in `softmax_mask` is a verified perf non-issue. See
  `docs/ARCHITECTURE.md` §5/§7 before touching causal performance.

## Pointers

- `README.md` — build / test / benchmark, platform, layout.
- `docs/ARCHITECTURE.md` — pipeline walkthrough, data layouts/swizzles, performance status,
  CK comparison methodology, and verified dead-ends.
