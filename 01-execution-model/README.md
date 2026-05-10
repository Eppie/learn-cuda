# Module 1 — Execution model & your first kernel

**Goal:** by the end of this module you should be able to launch a kernel, reason about
how threads map to hardware, and explain *why* a kernel takes the time it takes — even if
that explanation is only "it's bandwidth-bound at X% of peak."

---

## 1. Why GPUs win

A CPU optimizes for **single-thread latency**: deep pipelines, big caches, sophisticated
branch predictors. A GPU optimizes for **aggregate throughput**: many simple cores, modest
caches, and the assumption that there are always more threads waiting if some stall.

When you load from DRAM on a CPU, the core stalls and out-of-order execution tries to
keep busy. When you load from DRAM on a GPU, the warp stalls and the scheduler **switches
to a different warp** that's ready to run. Latency hiding via parallelism is the central
trick.

This is why "fast CUDA" almost always means: keep enough work in flight to hide latency,
and don't waste memory bandwidth.

## 2. The hardware (RTX 4090, `sm_89`)

```
GPU
├── 128 SMs (Streaming Multiprocessors)
│   ├── 4 warp schedulers per SM, each with its own register-file partition
│   ├── 65,536 32-bit registers per SM   = 256 KB total
│   │     split as 4 × 16,384 regs = 4 × 64 KB partitions (one per scheduler)
│   ├── 128 KB unified L1 / shared memory (configurable split; up to 100 KB shared)
│   └── Tensor Cores (4th gen)
├── 72 MB L2 cache
└── 24 GB GDDR6X — ~1008 GB/s peak DRAM bandwidth
```

A common point of confusion: the 256 KB register file is *per SM*, not per partition.
Each of the four schedulers owns a 64 KB slice (16,384 × 4 B), and a warp issued by that
scheduler can only see registers in its own slice. A thread can use up to 255 registers,
but the more you use, the fewer warps fit per SM, so occupancy drops. We'll see this in
the bench numbers.

Your code never targets an SM directly. You launch a **grid of blocks**; the runtime
schedules whole blocks onto SMs. Within a block, threads are grouped into **warps of 32**,
which execute in lockstep (SIMT — single instruction, multiple thread).

## 3. The programming model

```
Grid       (1D / 2D / 3D)
└── Block  (1D / 2D / 3D, max 1024 threads on most GPUs)
    └── Warp (32 threads, hardware-defined)
        └── Thread (one lane in the warp)
```

A kernel sees its position with built-in variables:

| variable      | meaning                              |
|---------------|--------------------------------------|
| `threadIdx.x` | this thread's index inside its block |
| `blockIdx.x`  | this block's index in the grid       |
| `blockDim.x`  | block size                           |
| `gridDim.x`   | grid size                            |

A typical 1D global index:

```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

A kernel launch looks like this:

```cpp
constexpr int N    = 1 << 24;
constexpr int BLK  = 256;
const     int GRID = (N + BLK - 1) / BLK;

vector_add<<<GRID, BLK>>>(d_a, d_b, d_c, N);
```

The `<<<grid, block>>>` syntax is sugar for `cudaLaunchKernel`; under the hood it queues
work on the default stream and returns immediately. The kernel runs **asynchronously** —
the next host line executes while the GPU is still working. That's why benchmarking GPU
code requires CUDA events or an explicit `cudaDeviceSynchronize()`.

## 4. Occupancy (intro)

**Occupancy** = active warps per SM ÷ maximum supported warps per SM. It's a coarse proxy
for "do you have enough parallelism to hide latency?" You raise it by giving the SM more
threads (bigger grid, or more threads per block). You lower it by using too many registers
per thread or too much shared memory per block — both are limited per-SM resources, so
heavy use of either caps the number of resident blocks.

We won't tune occupancy by hand yet, but watch the block-size sweep in `bench.cu`: you'll
see occupancy effects show up as throughput differences.

## 5. Warp divergence (preview)

The 32 threads of a warp share *one* program counter. So this code:

```cpp
if (threadIdx.x < 16) {
    out[gid] = expensive_a();
} else {
    out[gid] = expensive_b();
}
```

does **not** run `expensive_a` on 16 lanes and `expensive_b` on the other 16 in parallel.
The hardware *masks off* the lanes that aren't taking the current branch and walks both
paths sequentially. Wall-clock cost ≈ cost of `expensive_a` + cost of `expensive_b`. That's
**warp divergence**, and it can silently halve (or worse) your throughput.

The fix is usually to align branches to warp boundaries — `if (warp_id < K)` is free,
`if (lane_id < 16)` is expensive — or to push the conditional into a data-dependent
computation that all lanes can run uniformly. We'll look at concrete examples in
[Module 2 §5](../02-memory-coalescing/README.md) (where divergence interacts with memory)
and again in [Module 5](../05-reductions/README.md) (warp-level reductions are *designed*
to avoid divergence).

For now, internalize: if the same source line runs different code on different lanes,
you're paying for both paths.

## 6. Streams and synchronization (preview)

Every kernel launch goes onto a **stream** — by default the *default stream* (stream 0).
Operations on the same stream are serialized; operations on different streams can run
concurrently if the hardware has room.

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

kernel_a<<<g, b, 0, s1>>>(...);   // queued on s1
kernel_b<<<g, b, 0, s2>>>(...);   // queued on s2 — may run concurrently with kernel_a

cudaDeviceSynchronize();          // wait for all streams to drain
// alternatively: cudaStreamSynchronize(s1) waits only on s1
```

Three things to know now:

1. The `<<<...>>>` launch returns immediately; the kernel runs whenever the stream and
   hardware are ready. Always sync (events, `cudaDeviceSynchronize`, or
   `cudaStreamSynchronize`) before you read GPU output on the host.
2. The default stream has special semantics: it implicitly synchronizes with all other
   streams. Real concurrency requires non-default streams.
3. You can't time async work with CPU clocks; use `cudaEvent_t` (see `common/cuda_utils.h`'s
   `GpuTimer`).

We'll come back to streams, events, and CUDA Graphs as a serious tool in
[Module 11](../11-low-latency/README.md), where we use them to drive launch overhead toward
zero. For now, just know they exist.

## 7. Launch overhead — and why HFT cares

Every kernel launch has a fixed cost (~5–10 µs on a fast PCIe machine, sometimes higher on
WSL because each launch crosses a virtualization boundary). For ML training, where each
kernel runs for milliseconds, this is invisible. For **low-latency inference or
HFT-adjacent workloads**, where you might want a kernel to respond to an input in
microseconds, launch overhead can dominate.

Two distinct things to measure (both are in `bench.cu`):

- **Queue throughput.** Fire N empty launches back-to-back, then sync once at the end.
  Total time / N is dominated by the *queueing rate* — how fast the driver can stuff
  launches into the stream. This is what asynchronous loops naturally see.
- **Per-launch latency.** Fire one launch, sync, repeat. Each iteration includes the
  full host→GPU→host round-trip. This is what a synchronous "do one thing, wait for
  it" workload sees — and it's the number that matters for low-latency request/response
  patterns.

You'll typically see the per-launch-sync number be ~5–20 µs higher than the queued
number. That gap is the latency you'd save by switching to CUDA Graphs (Module 11).

---

## Exercises

> Open `starter.cu` and complete the TODOs.

**1. Implement vector add.** `c[i] = a[i] + b[i]` for `N = 1 << 24` floats (~16 M).
Verify against the host result.

**2. Sweep block size.** Use `bench.cu` to time the kernel at block sizes 32, 64, 128,
256, 512, 1024. Compute achieved DRAM bandwidth as `bytes_moved / time`. Vector add reads
2 floats and writes 1 per element — 12 bytes/element.

**3. Measure launch overhead.** `bench.cu` reports both **queued** and **per-launch
synced** timings for an empty kernel. Note the ratio between them and compare both to the
time it takes to *do* the vector add.

### Stretch

**4.** Try `c[i] = a[i] * x + b[i]` (saxpy). Same bandwidth as vector add? Why?

> *Hint:* Look at bytes moved per element vs FLOPs per element. Saxpy reads 2 floats,
> writes 1, does 2 FLOPs (one multiply + one add) — vector add reads 2, writes 1, does
> 1 FLOP. The bytes-per-element are identical (12), so for a memory-bound kernel the
> bandwidth should be identical too. If saxpy is faster on your machine, you're seeing
> the FMA (fused multiply-add) free instruction — the GPU can fold `a*x + b` into a
> single FFMA, reducing instruction-issue pressure but not byte traffic.

**5.** What happens at block size **48**? Why does the runtime accept it but it likely
underperforms?

> *Hint:* Warps are 32 lanes wide and indivisible. A block of 48 threads runs as 2 warps
> (64 lanes), with 16 lanes of the second warp permanently masked-off. You launch 50 %
> more warps than you need useful work for, so achieved throughput drops by roughly
> that fraction. The general rule: make `blockDim.x` a multiple of 32 unless you have a
> specific reason not to.

---

## Profiler checklist

Build and run the profiler over your solution:

```bash
make
ncu --set basic ./solution     # speed-of-light per-kernel summary
nsys profile --stats=true ./solution   # timeline + summary
```

Look at:

- **DRAM throughput** (% of peak) — for vector add this should be high; if it's < 70%
  on a 4090, something is off.
- **Achieved occupancy** — vector add is so simple it should be near max.
- **Memory chart** in `ncu` — confirm loads are coalesced (no replays).
- **Stall reasons** — at this stage "Long Scoreboard" (waiting on DRAM) is what you
  *want* to see for a memory-bound kernel.

The full counter dictionary lives in [Module 4 — Profiler counters introduced in this
module](../04-profiling/README.md#profiler-counters-introduced-in-this-module). For this
module, the counters worth watching are `dram__throughput.avg.pct_of_peak_sustained_elapsed`
(DRAM peak %) and `smsp__warps_active.avg.pct_of_peak_sustained_active` (achieved
occupancy).

## Key takeaways

- Threads are grouped into warps; warps into blocks; blocks into a grid.
- Warps share a program counter — divergent branches serialize across the warp.
- Kernel launches are async and live on streams; benchmark with CUDA events.
- Most simple kernels are bandwidth-bound. Your job is to hit DRAM peak.
- Launch overhead is real and it matters whenever individual kernels are short. There
  are *two* numbers worth measuring: queued throughput and per-launch latency.
