# learn-cuda

A topic-driven CUDA course aimed at high-performance ML kernels and low-latency designs.
Target hardware: **RTX 4090** (Ada, `sm_89`), CUDA 12.x.

## How to take this course

### Mandatory order

Modules **01–12 are sequential**. Each one assumes the previous ones; skipping
forward reliably wastes more time than it saves. **Module 13 is a *reference***
— forward-linked from M07 (`mma.sync`, `ldmatrix`), M08 (`cp.async` PTX), and
M11 (`__threadfence` / fence semantics). Read its sections when those modules
point you there; you don't need to read it end-to-end.

### Time budget per module

These are estimates for working through the README (concepts), solving the
exercises in `starter.cu`, running the bench, and skimming the profiler output.
They assume you read the README before touching code and don't re-derive
material the previous module already established.

| #  | Topic                              | Hours | Notes |
|----|------------------------------------|-------|-------|
| 01 | Execution model                    | ~2    | Vector add + block-size sweep + launch-overhead bench. |
| 02 | Memory coalescing                  | ~3    | AoS-vs-SoA + L2 sweep + sector counters. |
| 03 | Shared memory & tiling             | ~3    | Transpose ladder; 32×8-with-4-rows is now main, not stretch. |
| 04 | Profiling                          | ~3    | Two `ncu` exercises (single + multi-bottleneck) plus `nsys` and compute-sanitizer. |
| 05 | Reductions, scans, warp shuffles   | ~5    | Reductions v0/v1/v2 + Hillis-Steele + Blelloch + segmented + online softmax (required). |
| 06 | GEMM journey (6.0 → 6.6)           | ~10   | Eight kernels; v6 is full Boehm warp-tiling (WSUBM/WSUBN). The longest single module. |
| 07 | Tensor Cores                       | ~5    | WMMA v0 (evolved from M06 v6) + v1 swizzled + v2 raw `mma.sync`. |
| 08 | Async copy & pipelining            | ~4    | Legacy `__pipeline_*` + modern `cuda::pipeline` + `mbarrier`. |
| 09 | Fused epilogues                    | ~6    | Softmax + online softmax + Welford LN + LN+residual + GEMM+bias+GELU + FP16 LN. |
| 10 | FlashAttention                     | ~6    | FA-1 + multi-head + causal+tile-skip + KV-cache + GQA. |
| 11 | Low-latency patterns               | ~7    | Streams + events + graphs + persistent + ring + megakernel + Green Contexts + GPUDirect explainer. |
| 12 | Capstone                           | 5–30  | Project A 5–7 h, Project B 15–25 h, Project C 5–8 h, Project D 5–10 h. Project E is post-course. |
| 13 | PTX appendix (reference)           | ~2    | Skim end-to-end if you want, otherwise read on demand from M07/M08/M11 forward-refs. |

**Core path** = modules 01–11 + Project A + skim of 13: **~60–65 hours**.

**Extended path** = core + Projects B, C, and D from M12: **~85–115 hours**.

**Project E** (Mamba inference megakernel, post-course) is a multi-week
project. See [`12-capstone/PROJECT-E.md`](12-capstone/PROJECT-E.md).

The core path is the recommended budget. The extended path is for users who
want to land production-shaped systems (multi-stage CUDA-Graph pipeline,
megakernel, batched GEMM); it doesn't unlock anything pedagogically that the
core path doesn't already teach.

### How to use `solution.cu`

`solution.cu` is a **reference**, not a target to read first. The exercises
are the point of the course. The intended workflow:

1. Read the module's README. Internalize the concepts before opening code.
2. Open `starter.cu`. Solve the TODOs in order.
3. Run `./starter` and `./bench`. Confirm correctness and look at the numbers.
4. *Then* read `solution.cu` if you want to compare structures, or if you got
   stuck and the hints below didn't help.

Reading `solution.cu` first short-circuits the value of the exercise. The
gap between "I understand this code" and "I can write this code" is exactly
what the course is teaching you to close.

### What to do when stuck

In rough order of cost:

1. **Re-read the relevant section of the README.** Especially the "why"
   sections (every module has one). Most stuck-ness is "I lost track of *why*
   this step is here."
2. **Check [`STRETCH-ANSWERS.md`](STRETCH-ANSWERS.md)** — one-paragraph hints
   for every "Stretch" exercise across modules. Hints, not full answers; the
   point is still for you to do the work.
3. **Run `./bench` and look at the numbers.** "What do I expect this counter
   to be?" is often more useful than reading more prose.
4. **Profile with `ncu`.** Module 4 introduces the workflow and the counter
   glossary. Subsequent modules' "Profiler checklist" sections cite specific
   counters worth checking. The profiler is the truth.
5. **Read the corresponding section of `solution.cu`.** Last resort. Compare
   structures, then go back and re-derive.

### Visualizations

Twelve self-contained interactive HTML pages in [`viz/`](viz/) cover the
concepts that genuinely benefit from a picture (thread-to-memory mapping,
bank conflicts, the GEMM tile hierarchy, online softmax recurrence, FA tile
streaming, persistent doorbell state machine, etc.). No build step — open
the file in a browser. See [`viz/README.md`](viz/README.md) for the index
and which module each one supports.

### Recording results

[`BENCH-RESULTS.md`](BENCH-RESULTS.md) at the repo root is the place to log
the numbers you measure on your hardware. Each module's bench prints a few
key throughput / latency numbers — the file is a checklist of which ones to
record. Useful both for sanity-checking that you reproduced expected
behavior and for comparing future code changes against a known baseline.

---

## How to use a module

Each module is a self-contained directory:

```
NN-topic/
  README.md      # concepts + the "why"
  Makefile       # builds everything below
  starter.cu     # scaffold with TODOs — your work happens here
  solution.cu    # reference implementation
  bench.cu       # timing harness, varies parameters
```

Workflow:

```bash
cd 01-execution-model
make
./bench           # see baseline numbers
# edit starter.cu, solve the TODOs
./starter         # check correctness + speed
```

Every module ends with a **profiler checklist** — run Nsight Compute on your binary and look at specific counters. The point isn't to pass a test, it's to read the profiler.

## Curriculum

### Weekend 1 — Foundations
| # | Topic | What you'll be able to do |
|---|---|---|
| 01 | Execution model | Launch kernels, reason about warps/blocks/grids, measure both queued and per-launch-sync overhead |
| 02 | Memory hierarchy & coalescing | Hit a high fraction of DRAM peak; spot uncoalesced loads; recognize the AoS-vs-SoA trap |
| 03 | Shared memory & tiling | Use `__shared__` correctly (static + dynamic); avoid bank conflicts; forward-ref swizzling for 16-byte loads |
| 04 | Profiling | Drive Nsight Compute, Nsight Systems, and `compute-sanitizer`; read occupancy and stall reasons; counter glossary cited by every later module |

### Weekend 2 — Patterns and the GEMM journey
| # | Topic | What you'll be able to do |
|---|---|---|
| 05 | Reductions, **scans**, warp shuffles | Reductions v0/v1/v2; Hillis-Steele + Blelloch scans; segmented (per-row) reductions; online-softmax `(m, s)` recurrence |
| 06 | **GEMM journey** (6.0 → 6.6) | FP32 matmul from naive to ~80% of cuBLAS over 8 kernels. v6 is full Boehm warp tiling (WSUBM/WSUBN sub-tiling) |
| 07 | Tensor Cores | WMMA v0 evolved from M06 v6, v1 with swizzled shared memory, v2 raw `mma.sync` PTX with documented fragment layout |
| 08 | Async copy & pipelining | Legacy `__pipeline_memcpy_async` + modern `cuda::pipeline` + `mbarrier` for warp-specialized producer/consumer |

### Weekend 3 — ML kernels and low-latency
| # | Topic | What you'll be able to do |
|---|---|---|
| 09 | Fused epilogues | Stable softmax, online softmax, Welford LayerNorm, LN+residual, FP16 LN, **GEMM+bias+GELU** (post-pass and fused-in-MMA) |
| 10 | FlashAttention | FA-1 forward + multi-head + causal+tile-skip + KV-cache + GQA — all in working kernels, not capstone TODOs |
| 11 | Low-latency patterns | Streams, events, graphs, persistent kernels, ring buffer, megakernel, Green Contexts (CUDA 12.4+), GPUDirect RDMA explainer |
| 12 | Capstone (pick) | A: multi-head causal FA. B: low-latency CUDA-Graph inference pipeline. C: strided batched GEMM. D: megakernel variant of B. E (post-course): Mamba inference |

### Reference
| # | Topic |
|---|---|
| 13 | PTX / SASS appendix — `vector_add`, `cache_hints`, `cp.async`, `mma.sync`, `ldmatrix`, `bar.sync`, `clock64`. Forward-linked from M07/M08/M11. |

## Prerequisites

- Solid C/C++ (pointers, memory layout, build systems)
- `nvcc` 12.x and a `sm_89`-capable GPU (RTX 4090, RTX 6000 Ada, L40, H100 also fine)
- For profiling: `ncu` (Nsight Compute) and `nsys` (Nsight Systems) — both ship with the CUDA toolkit
- `compute-sanitizer` (also bundled with the toolkit) — used from M04 onward

## Build

```bash
make            # build all modules currently scaffolded
make clean
```
