# fmha_native

A hand-written HIP implementation of a **D64 (head-dim 64) BF16 fused multi-head attention
(FMHA) forward kernel** for AMD **gfx942 (MI300X, CDNA3)**. The project goal is **ISA- and
performance-parity with Composable Kernel (CK)** for the same problem — the kernel is built
bottom-up so its generated assembly and runtime match CK's equivalent D64 FMHA forward kernel.

Supports dense and variable-length batching, optional causal masking, and optional log-sum-exp
(LSE) output. Forward pass only; head dim fixed at 64; BF16 in, FP32 accumulate.

For the internals (pipeline, register layouts, performance status, CK comparison methodology),
see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For a quick orientation, see `CLAUDE.md`.

## Platform & toolchain

- **GPU:** gfx942 (MI300X, CDNA3). The kernel is specialized for this target.
- **Toolchain (verified):** ROCm 7.2.2, HIP 7.2, AMD clang 22, CMake ≥ 3.21, C++17.
- A **ROCm 7.2 container is recommended.** Development used a container with this repo
  bind-mounted (e.g. at `/root/workspace`); any host with a matching ROCm install also works.
  Commands below show both the plain form and the container form.

## Repository layout

| Path | Role |
|------|------|
| `src/fused/` | The production kernel: `kernel.cpp` (4 `__global__` entries), `pipeline.hpp` (per-block forward pass), and `op_lds/op_gemm/op_softmax/op_epilog.hpp` device-op helpers. |
| `src/components/` | 7 standalone building-block kernels (one per pipeline stage). **Test-only**, not used by `src/fused/`. |
| `src/components_ref/` | CPU/GPU reference oracles paired with the components. |
| `src/runner/` | Shared infra: parameter structs, device buffers, CPU reference, naive GPU reference, bf16 utils. |
| `tests/` | GoogleTest suites: full-kernel vs GPU ref, GPU ref vs CPU ref, and 7 component-vs-golden suites. |
| `scripts/` | `run-gates.sh` (build + tests + asm extract), `run-benchmark.sh` (perf sweep), `common.sh`. |
| `docs/` | `ARCHITECTURE.md` (deep dive). |
| `asm_compare/` | CK reference assembly for ISA diffing. |

## Build

```bash
cmake -B build -DGPU_TARGET=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`CMAKE_BUILD_TYPE=Release` is the default and is **mandatory** — the kernel relies on full
unrolling/inlining and a set of `-mllvm` codegen flags (see `CMakeLists.txt`); a debug/`-O0`
build will not produce correct code. `GPU_TARGET` defaults to `gfx942`.

Container form:
```bash
docker exec <container> bash -c "cd /root/workspace && cmake --build build -j\$(nproc)"
```

Main targets: `fmha_kernel` (the kernel static lib), `bench_fmha_fwd` (benchmark),
`test_fmha_fwd_d64` (full-kernel test), and the 7 component test binaries.

## Test

The canonical check is the gate runner, which builds, runs all tests, and extracts the kernel
assembly:

```bash
# inside the ROCm container (repo bind-mounted at /root/workspace):
docker exec <container> bash -c "/root/workspace/scripts/run-gates.sh"
```

A passing run reports `[  PASSED  ] 67 tests.` (full-kernel suite) plus seven
`[  PASSED  ] 7 tests.` lines (the component suites), ending with
`ALL REQUESTED GATES PASSED (g1,g2,g3)`.

The gate stages are: **G1** build, **G2** tests, **G3** extract the kernel `.s` to
`native_d64_kernel.s`. The component suites (G2) compare against **golden dumps** expected at
`/tmp/fmha-native-isa-match/golden/{full,partial}`; the gate fails if those are missing.

Run a single binary directly, e.g. the causal cases of the full-kernel suite:
```bash
./build/test_fmha_fwd_d64 --gtest_filter='*Causal*'
```

## Performance

Measured on **MI300X (gfx942)**, d64 bf16, via `scripts/run-benchmark.sh` (6 runs per config,
first dropped), compared against Composable Kernel (CK) `tile_example_fmha_fwd` built and run
back-to-back on the same GPU. TFLOPS is average-based on both sides (FLOP conventions match
within <0.1%), so `native/CK` is an apples-to-apples throughput ratio (higher is better).

Non-causal (`--mask 0`):

| B | H | S | native | CK | native/CK |
|---|---|------|------|------|------|
| 8 | 16 | 1024  | 291.7 | 318.4 | 91.6% |
| 4 | 16 | 2048  | 340.1 | 356.1 | 95.5% |
| 2 | 16 | 4096  | 367.9 | 378.4 | 97.2% |
| 1 | 8  | 8192  | 327.1 | 353.7 | 92.5% |
| 1 | 16 | 8192  | 381.1 | 396.1 | 96.2% |
| 1 | 8  | 16384 | 398.1 | 415.1 | 95.9% |
| 1 | 4  | 32768 | 397.2 | 412.6 | 96.3% |
| 1 | 2  | 40000 | 327.9 | 345.3 | 95.0% |

Causal (`--mask 1`, ×0.5 FLOP convention):

| B | H | S | native | CK | native/CK |
|---|---|------|------|------|------|
| 8 | 16 | 1024  | 206.4 | 228.0 | 90.5% |
| 4 | 16 | 2048  | 263.3 | 289.7 | 90.9% |
| 2 | 16 | 4096  | 297.8 | 312.8 | 95.2% |
| 1 | 8  | 8192  | 248.6 | 253.3 | 98.2% |
| 1 | 16 | 8192  | 352.2 | 366.7 | 96.1% |
| 1 | 8  | 16384 | 370.2 | 378.8 | 97.7% |
| 1 | 4  | 32768 | 387.1 | 397.8 | 97.3% |
| 1 | 2  | 40000 | 299.7 | 241.7 | 124.0% |

TFLOPS are in TFLOP/s. The `B1 H2 S40000` causal config is ragged/low-occupancy (CK regresses
there) — treat it as an outlier. Per-file roles and the same tables live in
[`src/fused/README.md`](src/fused/README.md); see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
§5 for why causal closed to ~97% (the M-tile load-balance fix).

## Benchmark

```bash
./scripts/run-benchmark.sh            # mask=0 (non-causal) sweep
./scripts/run-benchmark.sh --mask 1   # causal sweep
```

This sweeps 8 (batch, heads, seqlen) configs (6 runs each, dropping the first) and prints a
table of `Avg(ms)`, `Min(ms)`, and `TFLOPS` per config. TFLOPS is derived from the **average**
time. The single-shot binary takes individual configs:
```bash
./build/bench_fmha_fwd -b 1 -h 16 -s 8192 -d 64 --mask 1 --warmup 5 --iters 20
```
Run on a free GPU (set `HIP_VISIBLE_DEVICES`) for stable numbers.
