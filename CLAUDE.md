# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A topic-driven, self-paced CUDA course for ML kernels and low-latency designs.
Target hardware is **RTX 4090 (Ada, `sm_89`)** with CUDA 12.x. Each module is a
self-contained directory `NN-topic/` containing teaching material plus exercises.

## Module layout (the convention every module follows)

```
NN-topic/
  README.md       # concepts + the "why" — read this first
  Makefile        # builds the binaries below
  starter.cu      # scaffold with TODOs — the user's work happens here
  solution.cu     # reference implementation (do not just copy into starter)
  bench.cu        # timing harness; varies parameters
```

Modules with extra structure:
- **`05-reductions`** — also has `scan_starter.cu` / `scan_solution.cu` /
  `scan_bench.cu` (a separate exercise on scans alongside reductions).
- **`06-gemm`** — `kernels.cuh` (v0..v6 ladder), `gemm.h` (verify helper).
  v6 is the **full Boehm warp-tiled** kernel (WSUBM/WSUBN sub-tiling); v5
  splits into 5a (vectorized) and 5b (transposed As).
- **`07-tensor-cores`** — `kernels.cuh`, `gemm_tc.h`. Three sub-kernels:
  `gemm_v0_wmma` (evolves from M06 v6 — same launcher / tile params, inner
  FMA loop replaced with WMMA fragments), `gemm_v1_wmma_swizzled` (adds
  shared-memory swizzling), `gemm_v2_mma_sync` (raw mma.sync PTX, hits ~76%
  of cuBLAS hgemm vs ~40% for WMMA).
- **`09-fused-epilogues`** — `kernels.cuh` covers softmax, online softmax,
  Welford LayerNorm, LN+residual, FP16 LN, GEMM+bias+GELU (post-pass and
  fused).
- **`10-flash-attention`** — `kernels.cuh` covers two ladders side by side.
  *Optimization ladder*: `flash_attention` (10.0, one thread per Q row, FP32),
  `flash_attention_warp` (10.1, bigger blocks for better K-tile amortization),
  `flash_attention_wmma` (10.2, FP16 inputs / FP32 acc / WMMA inner matmul),
  `flash_attention_async_wmma` (10.3, adds 2-stage cp.async double-buffering on
  top of 10.2 — measured 28 TF/s at N=8192 single-head, 4.1× over 10.0).
  *Shape variants on the simple base*: MHA, causal+tile-skip, KV-cache, GQA.
- **`11-low-latency`** — multiple demo files: `bench.cu`, `events_demo.cu`,
  `persistent_demo.cu`, `ring_buffer.cu`. Module README is structured in
  sub-sections (§4 doorbell → §5 ring → §6 megakernel → §7 Green Contexts
  → §8 GPUDirect).
- **`12-capstone`** — Project A scaffold in `kernels.cuh` / `starter.cu`,
  Project B scaffold in `inference_pipeline.cu`. **`PROJECT-E.md`** is
  a post-course Mamba megakernel project (TODO, not implemented).
- **`13-ptx-appendix`** — committed `.cu` / `.ptx` / `.sass` triples for
  reading material: `vector_add`, `cache_hints`, `cpasync_inline`,
  `mma_sync_example`, `ldmatrix_example`. Plus `clock_microbench.cu` and
  `bar_sync_example.cu` (no committed PTX — runtime-only).

**`viz/`** is a top-level directory with 12 self-contained interactive HTML
visualizations (vanilla JS + SVG, no build step) covering thread-memory
mapping, bank conflicts, warp shuffles, the GEMM tile hierarchy, WMMA
fragment layouts, the cp.async pipeline timeline, online softmax,
FlashAttention tile streaming, persistent doorbell, CUDA-Graph replay, and
PTX↔SASS alignment. See `viz/README.md`.

`common/` is shared by every module:

- `common/cuda_utils.h` — `CUDA_CHECK` / `CUDA_CHECK_LAST` macros and a
  `GpuTimer` class (CUDA-event wall clock). New kernels and benches should use
  these rather than rolling their own.
- `common/bench.h` — `bench_min_ms(iters, fn)` returns the **minimum** observed
  GPU time across `iters` runs after one warm-up. Min (not mean) is the
  convention here; it filters out OS/driver jitter and reports what the
  hardware can do.

## Build / run

```bash
make                                # build every module from the repo root
make clean                          # clean every module
make -C 01-execution-model          # build one module
cd 01-execution-model && make       # equivalent
./bench                             # run the bench in that module
./starter                           # run the user's WIP solution
./solution                          # run the reference
```

There is no test runner — correctness checks are inline in each `starter.cu` /
`bench.cu` (typically a host-side recompute compared to the device result).
"Run a single test" means running the relevant binary directly.

### nvcc invocation (every module uses this — keep it consistent when adding modules)

```
nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo -I../common
```

- `sm_89` is fixed (RTX 4090 / Ada). Don't lower it without reason — several
  modules use Ada-specific features (`cp.async`, 4th-gen Tensor Cores).
- `-lineinfo` is required so Nsight Compute / `cuobjdump` can map back to source.
- `-I../common` is how modules pick up the shared headers above.
- Module `06-gemm` additionally links `-lcublas` (used as the speed-of-light reference).

### PTX / SASS (module 13 only)

`13-ptx-appendix` adds two extra Make targets:

```bash
make ptx     # produce .ptx for each .cu via `nvcc --ptx`
make sass    # produce .sass via `nvcc --cubin` then `cuobjdump --dump-sass`
```

Existing `.ptx` and `.sass` files are checked in deliberately as reading
material; regenerate them when the corresponding `.cu` changes.

## Profiling (the point of every module's "profiler checklist" section)

Every module's README ends with a profiler checklist. The expected tools are:

```bash
ncu --set full ./solution                # full per-kernel profile (Nsight Compute)
nsys profile --stats=true ./solution     # timeline + summary (Nsight Systems)
```

Both ship with the CUDA toolkit. The course's pedagogical stance is that
reading the profiler matters more than passing a correctness check — when
helping the user, surface profiler counters (DRAM throughput, achieved
occupancy, stall reasons, bank conflicts) rather than just "it works."

## Working on exercises

- `starter.cu` is where the user solves TODOs. **Do not auto-fill the TODOs** by
  copying from `solution.cu` unless the user explicitly asks you to — the
  exercises are the point. When asked for hints, give the smallest useful
  pointer rather than the full answer.
- `solution.cu` is a reference, not a target to beat.
- `bench.cu` is meant to be modified for parameter sweeps (block size, tile
  size, etc.); changing it is normal.

## Curriculum order (modules build on each other)

`01-execution-model` → `02-memory-coalescing` → `03-shared-memory` →
`04-profiling` → `05-reductions` → `06-gemm` (7 incremental versions, 6.0 → 6.6)
→ `07-tensor-cores` → `08-async-copy` → `09-fused-epilogues` →
`10-flash-attention` → `11-low-latency` → `12-capstone`. `13-ptx-appendix`
is a *reference*, linked from M07/M08/M11; not meant to be read end-to-end.

Cross-module load-bearing dependencies:
- M03 → M07: M03 forward-refs swizzling to M07 §3 "Swizzled shared memory."
- M05 → M09 → M10: M05 introduces online softmax primitives in
  `solution.cu`; M09 and M10 both use the same `(m, s)` recurrence.
- M06 v6 → M07: M07's WMMA kernel reuses v6's launcher + warp tiling.
- M11 → M13: M11 forward-refs `__threadfence` / `bar.sync` PTX form to M13.
- M12 → many: capstone projects ask the learner to combine 3-5 prior modules.

When adding new content or kernels, prefer fitting into the existing module
that owns the topic rather than creating a new module. Renumbering is
expensive — every cross-reference breaks. Use **subsection ladders** within
a module instead (M11.0 → M11.5; M06 6.0 → 6.6).

## Wave 2 follow-ups

`WAVE2-TODO.md` at the repo root captures deferred items from the Wave 1
overhaul (megakernel framing, Green Contexts, Mamba TODOs, online-softmax
shared-header lift, etc.). Many of these landed during Wave 2 but the file
remains as the historical work order.

## Recording perf numbers

`BENCH-RESULTS.md` at the repo root is where measured perf numbers live —
each module's bench prints a few key throughput / latency metrics, and the
file collects them into one place with the run setup (driver, OS, host)
documented at the top. Update it when you re-bench on new hardware or after
substantive kernel changes; it's the historical perf ledger for the repo.
