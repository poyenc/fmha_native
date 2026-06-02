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

Measured on **MI300X (gfx942)**, d64 bf16, via `scripts/run-benchmark.sh` (native) and
`example/ck_tile/01_fmha/script/benchmark_fwd_sp3_compare.sh` (CK), run consecutively on the
same idle GPU. Both use the identical 6-run framework: 6 runs per config, drop the first,
**average** runs 2–6 (each run is itself an average over 20 iterations). TFLOPS is average-based
on both sides (FLOP conventions match within <0.1%), so `native/CK` is an apples-to-apples
throughput ratio (higher is better).

Non-causal (`--mask 0`):

| B | H | S | native | CK | native/CK |
|---|---|------|------|------|------|
| 8 | 16 | 1024  | 302.0 | 313.1 | 96.5% |
| 4 | 16 | 2048  | 348.4 | 350.3 | 99.5% |
| 2 | 16 | 4096  | 371.6 | 374.5 | 99.2% |
| 1 | 8  | 8192  | 335.9 | 346.3 | 97.0% |
| 1 | 16 | 8192  | 387.9 | 392.5 | 98.8% |
| 1 | 8  | 16384 | 406.5 | 412.5 | 98.6% |
| 1 | 4  | 32768 | 404.9 | 411.1 | 98.5% |
| 1 | 2  | 40000 | 335.1 | 342.5 | 97.8% |

Causal (`--mask 1`, ×0.5 FLOP convention):

| B | H | S | native | CK | native/CK |
|---|---|------|------|------|------|
| 8 | 16 | 1024  | 207.8 | 223.8 | 92.9% |
| 4 | 16 | 2048  | 265.6 | 286.1 | 92.8% |
| 2 | 16 | 4096  | 305.8 | 305.4 | 100.1% |
| 1 | 8  | 8192  | 256.2 | 250.1 | 102.4% |
| 1 | 16 | 8192  | 358.1 | 361.5 | 99.1% |
| 1 | 8  | 16384 | 378.9 | 377.2 | 100.4% |
| 1 | 4  | 32768 | 395.8 | 393.8 | 100.5% |
| 1 | 2  | 40000 | 308.2 | 243.5 | 126.5% |

TFLOPS are in TFLOP/s. The `B1 H2 S40000` causal config is ragged/low-occupancy (CK regresses
there) — treat it as an outlier. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5 for why
causal closed to ~97% (the M-tile load-balance fix).

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
