# Module 8 — Async copy and software pipelining

**Goal:** by the end of this module you should be able to (a) use `cp.async`
(via `__pipeline_memcpy_async` *or* the modern `cuda::pipeline` /
`cuda::memcpy_async`) to overlap shared-memory loads with compute,
(b) write a double-buffered GEMM main loop, (c) reach for `mbarrier` when
block-wide `__syncthreads` is too coarse, and (d) explain why pipelining is
the single biggest remaining lever after Module 6's tiling and Module 7's
Tensor Cores.

This module bridges the Module 7 → cuBLAS gap. We start at the sync-WMMA
baseline and climb toward cuBLAS by hiding shared-memory load latency behind
the MMA work that needs no data we haven't already loaded.

> **Forward-ref:** the underlying PTX form is documented in Module 13
> [§ cp.async PTX form](../13-ptx-appendix/README.md#cpasync-ptx-form),
> which shows the exact instruction both APIs emit and the difference
> between the `.cg` (cache-global, skip L1) form the wrappers default to
> and the `.ca` (cache-all) form you can request via inline asm.

---

## 1. The problem

Module 7's main loop looks like:

```
for kBlock in [0, K, BK):
    cooperative load A tile -> As (synchronous)
    cooperative load B tile -> Bs (synchronous)
    __syncthreads()
    for kk in [0, BK, WMMA_K):  # WMMA work on the just-loaded tiles
        load_matrix_sync, mma_sync, ...
    __syncthreads()
```

Each iteration's structure is:

```
[ load A,B ]  [ MMA work ]  [ load A,B ]  [ MMA work ]  ...
   bound on        bound on
   DRAM/L2         Tensor Core
```

The two halves don't overlap — every iteration waits for its loads before
issuing any MMAs, even though the *next* iteration's loads have nothing to do
with the *current* iteration's MMAs. That's a serialization the hardware
shouldn't need.

## 2. `cp.async` — load straight into shared memory, async

Starting with Ampere, the SM has dedicated load units for **`cp.async`**: a
memory instruction that copies global → shared **without** going through
registers, and **without** stalling the issuing warp. The warp issues it,
keeps executing, and checks for completion later.

There are two C++ APIs to drive `cp.async`. They're functionally equivalent
on Ada but stylistically different.

### 2a. The legacy (`<cuda_pipeline.h>`) API

```cpp
#include <cuda_pipeline.h>

__pipeline_memcpy_async(smem_dst_ptr, gmem_src_ptr, /*bytes=*/16);
__pipeline_commit();                 // group prior async copies into one batch
__pipeline_wait_prior(N);            // wait until at most N batches remain in flight
```

`__pipeline_memcpy_async` accepts 4, 8, or 16-byte transfers. The 16-byte form
maps to `cp.async.cg.shared.global [smem], [gmem], 16` and is what production
code uses.

The mental model:

- A warp issues many `__pipeline_memcpy_async` calls — they're queued, not executed.
- One `__pipeline_commit()` per "batch" that should be waited on as a unit.
- `__pipeline_wait_prior(N)` waits until *all but the most recent N* batches
  have drained.

### 2b. The modern (`<cuda/pipeline>`) API

```cpp
#include <cuda/pipeline>
#include <cooperative_groups.h>

__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
auto pipe = cuda::make_pipeline(cooperative_groups::this_thread_block(), &pss);

pipe.producer_acquire();             // wait for a stage slot to be free
cuda::memcpy_async(dst, src,
                   cuda::aligned_size_t<16>(sizeof(int4)), pipe);
pipe.producer_commit();

pipe.consumer_wait();                // wait for next consumed stage to be ready
// ... use the data ...
pipe.consumer_release();
```

This is the form CUTLASS and FlashAttention idiom around. It's typed,
composes cleanly with `cuda::barrier`, and on Hopper is the prerequisite for
TMA (`cp.async.bulk.tensor`). On Ada it emits the *same* `cp.async`
instruction the legacy form does — see Module 13 [§ cp.async PTX form](../13-ptx-appendix/README.md#cpasync-ptx-form)
for the disassembly.

### Commit-group depth limit

Both APIs maintain an in-flight commit-group queue per warp. The hardware
limit is **64 in-flight commit groups per warp**. Practically:

- `STAGES <= 4` for production GEMM is far below the limit, so it's a
  non-issue here.
- Persistent kernels that issue `cp.async` from many independent loops
  (e.g. K/V stages of FlashAttention) can hit it; the symptom is a silent
  back-pressure stall when `__pipeline_commit` blocks waiting for slots.
- CUTLASS-style multi-stage pipelines (5-8 stages on Hopper) approach the
  limit and use `mbarrier` for finer accounting.

## 3. Software pipelining: double buffering

The pattern is:

```
+------+-----------------------------------------------------------+
| iter | what happens                                              |
+------+-----------------------------------------------------------+
| -1   | (prologue) issue load for tile 0 -> buffer[0], commit     |
|  0   | issue load for tile 1 -> buffer[1], commit                |
|      | wait_prior(1)                  // tile 0 ready            |
|      | __syncthreads                                             |
|      | mma on buffer[0]                                          |
|      | __syncthreads                  // see "the WAR fence" §3a |
|  1   | issue load for tile 2 -> buffer[0], commit                |
|      | wait_prior(1)                  // tile 1 ready            |
|      | __syncthreads                                             |
|      | mma on buffer[1]                                          |
|  ... | ...                                                       |
+------+-----------------------------------------------------------+
```

Two shared-memory buffers; the next iteration's load is in flight while the
current iteration's MMA work runs. The MMA work hides the load latency.

That's the simplest "2-stage" pipeline. With more stages (3, 4, ...) you
cover more load latency at the cost of more shared memory and tracking
complexity.

### 3a. The WAR fence

There's a subtle correctness fence at the *end* of each iteration's MMA
work. Iteration `i+1` begins by issuing async copies into `load_stage`,
which under STAGES=2 is the same buffer iteration `i` just MMA-read. Without
a `__syncthreads()` between the MMA reads and the next iteration's
`cp.async` writes, the cp.async stores can race the MMA reads in
slow-warps. The kernel needs that fence even though `wait_prior` at the top
of the next iteration provides a different guarantee (the *load* is done,
not that the *previous read* is done).

(With STAGES >= 3 the buffers don't alias across consecutive iterations,
but the fence is still needed for the memory-model handshake; CUTLASS
formalizes this with mbarrier-based handshakes — see §5.)

## 4. The skeleton

```cpp
constexpr int STAGES = 2;

__shared__ __half As[STAGES][BM * BK];
__shared__ __half Bs[STAGES][BK * BN];

// Prologue: issue STAGES-1 stages worth of loads
for (int s = 0; s < STAGES - 1; ++s) {
    issue_async_loads_for_stage(s, kBlock = s * BK);
    __pipeline_commit();
}

int compute_stage = 0;
int load_stage    = STAGES - 1;

for (int kBlock = 0; kBlock < K; kBlock += BK) {
    int next_kBlock = kBlock + (STAGES - 1) * BK;

    if (next_kBlock < K) {
        issue_async_loads_for_stage(load_stage, next_kBlock);
    }
    __pipeline_commit();                 // empty commit at the tail is fine

    __pipeline_wait_prior(STAGES - 1);   // keep most recent STAGES-1 in flight
    __syncthreads();

    do_mma_on_stage(compute_stage);

    __syncthreads();                     // WAR fence (§3a)
    compute_stage = (compute_stage + 1) % STAGES;
    load_stage    = (load_stage    + 1) % STAGES;
}
```

The data dependency chain: a load committed at iteration `i` becomes ready by
iteration `i + STAGES - 1`. With `STAGES = 2`, that means "next iteration's
compute sees this iteration's load."

## 5. mbarrier: arrive-wait at finer granularity than `__syncthreads`

`__syncthreads()` is a block-wide barrier — every thread in the block waits
for every other thread. That's overkill when only *some* warps need to
synchronize, e.g. one producer warp filling shared memory while two
consumer warps drain it.

The PTX `mbarrier` (memory barrier) primitive offers an arrive-wait
semantic: each thread *arrives* at the barrier (incrementing a count); the
barrier *flips phases* every time the expected count is reached; threads
*wait* on the current phase until it flips.

In CUDA C++ this is exposed via `<cuda/barrier>`:

```cpp
#include <cuda/barrier>

__shared__ cuda::barrier<cuda::thread_scope_block> bar;
if (threadIdx.x == 0) init(&bar, blockDim.x);
__syncthreads();   // make sure init is visible

// per-thread:
auto token = bar.arrive();          // increment, get a phase token
bar.wait(std::move(token));         // wait until expected count reached
```

For this module's GEMM the per-block scope of `__syncthreads` is fine, but
two patterns benefit from `mbarrier`:

1. **Producer/consumer warps.** Designate 1-2 warps as "loader warps" doing
   only `cp.async` + commit; designate the rest as "math warps" doing only
   MMA. Pair them through an mbarrier per stage. This is the CUTLASS
   "warp-specialized GEMM" pattern.

2. **Async-copy completion tracking** (the modern way to do `wait_prior`).
   `cuda::barrier::arrive_and_wait_async` understands `cp.async` completion
   without needing a separate `__pipeline_wait_prior`. On Ampere/Ada this
   is the same instruction underneath; on Hopper the mbarrier path is
   *required* for TMA (TMA writes its completion phase directly to an
   mbarrier slot).

A sketch (not in the kernels for this module, but useful to read):

```cpp
__shared__ cuda::barrier<cuda::thread_scope_block> stage_ready[STAGES];

// loader warp issues cp.async, then arrives:
cuda::memcpy_async(dst, src, size, stage_ready[stage]);
stage_ready[stage].arrive();   // bound to cp.async completion

// math warp waits:
auto t = stage_ready[stage].arrive();
stage_ready[stage].wait(std::move(t));   // unblocks when cp.async is done
```

The CUDA C++ Programming Guide § "Asynchronous Barrier" has the canonical
shape; `gemm_v_modern` in `kernels.cuh` uses the simpler (non-mbarrier)
`cuda::pipeline` flavor as the entry point.

## 6. Why this is the *biggest* remaining lever

Module 6 closed the *bytes* gap (better tiling → less DRAM traffic). Module
7 closed the *math throughput* gap (Tensor Cores). What remains after those
two is the *latency* gap: even with the right number of bytes and the
fastest math units, back-to-back load-then-compute serializes the two
pipelines.

`cp.async` lets the load pipeline run ahead. On the Ada SM, that's exactly
what cuBLAS / CUTLASS do under the hood — the production GEMMs you measured
against in Modules 6 and 7 are pipelined with 3–5 stages and have been since
the Ampere release.

## 7. Where it stops helping

- **You've already saturated math throughput.** If MMA is fully busy on
  every cycle, hiding load latency doesn't get you more FLOPs.
- **Shared-memory pressure.** More stages = more shared memory per block =
  lower occupancy. Beyond 3–4 stages you're usually trading occupancy for
  very small latency wins.
- **Tiny K.** If `K` only requires a couple of `kBlock` iterations, the
  pipeline startup cost dominates and you're better off without it.

## 8. Beyond Ada — TMA and Hopper

> **One-paragraph forward-ref.** This module's `cp.async` evolves on
> Hopper into `cp.async.bulk.tensor`, the **Tensor Memory Accelerator**
> (TMA). TMA replaces per-element address math with a single descriptor
> (shape, strides, layout, swizzle pattern); one instruction copies an
> entire multi-D tile, and completion is signaled to an `mbarrier` slot
> directly — no more per-warp commit-group tracking. Combined with `wgmma`
> (warp-group async MMA), it lets a single warp-group issue tile-load +
> tile-mma async, with the hardware tracking completion. On Ada you stay
> with `cp.async` + WMMA / `mma.sync` — there's no TMA on `sm_89`.

---

## Exercises

> Open `starter.cu` and complete the TODOs.

1. **Single-buffered async.** Replace the synchronous loads in Module 7 with
   `__pipeline_memcpy_async` + commit + `wait_prior(0)` (wait for everything
   before you compute). This is functionally equivalent to the sync version
   but exercises the cp.async machinery — get this working before adding
   the second buffer.
2. **Double-buffered async.** Add the second shared buffer, the prologue,
   and the `wait_prior(1)` pattern. Verify that the result still matches
   cuBLAS within tolerance. Don't forget the WAR fence at the end of the
   loop body (§3a) — without it, the kernel may pass at small K and fail
   at large K.
3. **(Stretch) 3-stage pipeline.** Increase `STAGES` to 3, add another
   buffer, and measure. On Ada you should see modest gains over 2 stages;
   with smaller `BK` you may need 3–4 stages to fully hide latency.

Each version should pass the cuBLAS verification.

### Stretch

- **Modern API rewrite.** Take your working `__pipeline_memcpy_async`
  kernel and rewrite it using `cuda::pipeline` + `cuda::memcpy_async`. The
  reference is `gemm_v_modern` in `kernels.cuh`. Inspect the PTX of both
  with `nvcc --ptx`; they should emit the same `cp.async.cg.shared.global`
  instruction.
- **mbarrier-based stage tracking.** Replace `__pipeline_wait_prior` with
  per-stage `cuda::barrier` arrive/wait; have one warp produce while the
  other three consume. This is the warp-specialized form CUTLASS uses;
  on Ada it's a stylistic exercise, on Hopper it's a TMA prerequisite.
- **Compare against cuBLAS at varying problem sizes.** What happens to the
  gap as `K` shrinks? At small `K`, cuBLAS's pipeline startup overhead
  becomes visible. The bench's scaling study makes this concrete.
- **Profile the difference.** `ncu` will show *Issue Active* (warp issued
  an instruction this cycle) climbing as you add stages — that's the
  metric of "compute and load are overlapping".

---

## Profiler checklist

```bash
make
ncu --set full ./bench
```

Look at:

- **Compute throughput (SM)** — should *rise*; you're using the math units
  more of the time.
- **Issue Active** (or "Issue Slot Utilization") — should also rise; warps
  spend fewer cycles waiting on memory.
- **`smsp__warps_eligible.avg.per_cycle_active`** — average eligible warps
  per cycle. Pipelining reduces serialization, increasing this.
- **Stall reasons → "Long Scoreboard"** — should *decrease*. That's the
  wait-on-shared-memory state we're explicitly hiding.
- **`smsp__inst_executed_pipe_lsu.sum` (LSU instructions)** — `cp.async` ops
  show up here, distinct from regular LDG counters.

## Key takeaways

- `cp.async` is a load instruction that doesn't stall the issuing warp. It
  lets the load pipeline run ahead of compute.
- Two C++ APIs drive it: legacy `__pipeline_memcpy_async` /
  `__pipeline_commit` / `__pipeline_wait_prior`, and modern
  `cuda::pipeline` / `cuda::memcpy_async`. They emit the same PTX. Modern
  is preferred in new code and required for Hopper TMA.
- Double buffering: while iteration N's MMAs run, iteration N+1's loads
  are in flight. Two shared buffers are enough to start; 3–4 stages can
  squeeze out more.
- Each warp can have at most ~64 commit groups in flight — far above what
  this module needs but worth knowing for FA-style kernels.
- `mbarrier` (`<cuda/barrier>`) lets producer/consumer warps synchronize
  at finer granularity than `__syncthreads`. Production CUTLASS uses this
  pattern for warp specialization.
- Beware the WAR fence at the end of each pipelined iteration body — it's
  the bug that passes at small K and fails at large K (§3a).
- This is the last big "structural" lever for a single-kernel GEMM. Beyond
  this is autotuning, fancy fragment scheduling, and Tensor Memory
  Accelerator (Hopper).
