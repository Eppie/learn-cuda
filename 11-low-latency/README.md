# Module 11 — Low-latency patterns

**Goal:** by the end of this module you should be able to (a) reduce kernel-launch
overhead with CUDA Graphs, (b) build a persistent-kernel doorbell loop that gets
host→GPU→host round trips down to a few microseconds, (c) use CUDA events to
synchronize across streams, (d) use pinned host memory and `cudaMemPrefetchAsync`
for fast async transfers, (e) build a multi-producer ring-buffer dispatched by a
persistent kernel, and (f) reason about when each pattern fits.

This is the module that addresses your secondary interest in HFT-style designs.
ML-throughput thinking dominated Modules 6–10; here we flip to *latency*.

> Forward-ref: the PTX form of the fences and atomics used in this module
> (`__threadfence`, `__threadfence_system`, `atomicAdd`, `__nanosleep`) is
> covered in **Appendix 13 §M11-anchor: low-latency primitives** — open
> `13-ptx-appendix/README.md` after this module if you want to see exactly
> what the compiler emits.

---

## 1. The launch-cost ladder

For a typical ML kernel that runs in milliseconds, kernel launch overhead (~5 µs
on a fast machine) is invisible. For a kernel that needs to respond to an
external event in **single-digit microseconds**, that 5 µs is the dominant cost.

The progression in this module is not "ways to make launches fast" — it's
**ways to eliminate launches entirely**. The ladder:

| Step | Mechanism | What it does | Section |
|---|---|---|---|
| 0 | `cudaLaunchKernel` baseline | One syscall per kernel; ~5 µs overhead each | (M01 covers this) |
| 1 | **Reduce** per-launch overhead | CUDA Graphs replay an N-kernel DAG with one driver call | §2 |
| 2 | **Eliminate** per-request launches | Persistent kernel watches a doorbell or ring; one launch per *lifetime*, not per request | §4–5 |
| 3 | **Eliminate** per-pipeline launches | Megakernel runs all phases (GEMM → softmax → GEMM) in one kernel; phase dispatched internally | §6 |
| 4 | **Partition** SMs to hardware-isolate the hot path | Green Contexts dedicate N SMs to the persistent / megakernel; rest run batch | §7 |
| 5 | **Bypass** the host CPU entirely | GPUDirect RDMA: NIC writes the doorbell directly | §8 |

Steps 1 and 5 reduce or remove costs at the *edges* of the kernel; steps 2–4
operate on what runs *inside* the GPU. Together they take a typical
"queue+launch+copy+wait" pattern from ~50 µs end-to-end down to single-digit
µs (§5) or sub-µs (§8 with the right NIC).

The supporting tools — CUDA events (§3), pinned memory (§9), streams (§10) —
are the connective tissue. They aren't the latency story; they're the things
without which the latency story doesn't compile.

## 2. CUDA Graphs

A CUDA Graph is a pre-recorded DAG of CUDA work (kernels, memcpys, host
callbacks) that you submit with one API call. Once instantiated, you launch it
many times with effectively **zero per-launch overhead** — the driver knows
exactly what's coming and skips the per-kernel argument parsing, validation,
and dispatch logic.

There are two ways to build a graph:

```cpp
// 1. Capture an existing stream sequence.
cudaStream_t s;
cudaStreamCreate(&s);

cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
kernel1<<<...>>>();
kernel2<<<...>>>();
kernel3<<<...>>>();
cudaGraph_t g;
cudaStreamEndCapture(s, &g);

cudaGraphExec_t gx;
cudaGraphInstantiate(&gx, g, nullptr, nullptr, 0);

// 2. Replay (cheap).
for (int i = 0; i < N; ++i) cudaGraphLaunch(gx, s);
```

Capture-and-instantiate is heavy (tens of microseconds, sometimes more). Replay
is *fast* — typically less than 1 µs added per replay vs. ~5 µs per individual
launch. For N=3 kernels at 1 ms each, that's an O(percent) win on throughput, but
for short kernels it's huge.

Production note: graphs are great when the same DAG runs many times with the
same shape. They're awkward when shapes change every step (you'd need a new
graph). LLM serving frameworks like vLLM use them aggressively for the
"forward pass once" hot path.

## 3. CUDA events: the bridge between streams and graphs

A CUDA **event** is a lightweight marker you can record on one stream and
either (a) wait on from the host, (b) wait on from another stream, or (c) use
to compute elapsed time between two markers. Events cost a few hundred
nanoseconds to record and are the canonical way to express fine-grained
inter-stream dependencies.

The three core operations:

```cpp
cudaEvent_t e;
cudaEventCreate(&e);
// — or —
cudaEventCreateWithFlags(&e, cudaEventDisableTiming);  // ~30% cheaper record

// Record on stream A:
cudaEventRecord(e, sA);

// (a) Host waits:
cudaEventSynchronize(e);

// (b) Another stream waits — non-blocking on the host. The runtime inserts a
//     wait node into sB's queue; sB's next operation will block until e fires.
cudaStreamWaitEvent(sB, e, 0);

// (c) Time-elapsed between two events (requires both NOT have DisableTiming):
float ms; cudaEventElapsedTime(&ms, e_start, e_stop);
```

Mental model:

| Tool | Use when |
|---|---|
| `cudaEventSynchronize` / `cudaStreamSynchronize` | host needs the result *now* |
| `cudaStreamWaitEvent` | one stream depends on another; don't bounce through host |
| CUDA Graphs | static DAG that repeats |
| Persistent kernel + doorbell | sub-10 µs request/response, fixed shape |

`cudaStreamWaitEvent` is the building block CUDA Graph capture uses internally
when you record kernels across multiple streams. If you can describe your
dependencies as a static DAG, prefer a graph; events are the dynamic version.

See **`events_demo.cu`** for a working example with all three modes.

## 4. Persistent kernels with a doorbell

For the *truly* latency-critical case (microsecond response), don't launch a new
kernel per request — keep one running and signal it.

```cpp
__global__ void persistent_worker(volatile int* doorbell,
                                  volatile int* response,
                                  ...workload args...) {
    while (true) {
        int cmd;
        do {
            cmd = *doorbell;
            if (cmd == 0) __nanosleep(200);   // back off, save SM power
        } while (cmd == 0);
        if (cmd == -1) break;                 // shutdown

        // do work...

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            __threadfence();        // make output visible BEFORE we signal
            *response = 1;
            *doorbell = 0;
        }
    }
}
```

The doorbell and response are in **host-mapped pinned memory**:

```cpp
int* dbell_host;
cudaHostAlloc(&dbell_host, sizeof(int), cudaHostAllocMapped);

int* dbell_dev;
cudaHostGetDevicePointer(&dbell_dev, dbell_host, 0);

persistent_worker<<<grid, block>>>(dbell_dev, ...);

// To trigger work — note the order!
*resp_host  = 0;     // clear response FIRST (see ordering note below)
*dbell_host = 1;     // then arm the doorbell
while (*resp_host == 0) ;
```

### 4.1 Measured round-trip on this machine

Numbers from `./persistent_demo` on the course's reference setup
(RTX 4090, PCIe Gen4 x16, x86 host, WSL2, CUDA 12.6):

```
Persistent-kernel round-trip: 2.04 us  (5000-iter mean, idle GPU, no other CUDA workload)
```

That clean idle number is the *floor*. Under contention (other CUDA workloads
on the same GPU, busy host scheduler, virtualization overhead), the same
binary will measure 5–15 µs — same kernel-side primitives, same protocol,
just sitting through more PCIe / OS jitter. That spread is the rule for
host-mapped pinned-memory signalling on consumer hardware: PCIe latency,
x86 power-state transitions (C-states, P-states), and WSL2's added
context-switch cost all show up in the noise floor.

Treat the ~2 µs number as a best-case figure: tight host-side spin, response
in host-mapped pinned memory, no real workload in the kernel body. With
actual work in the kernel body, a non-spinning host, server-class memory
isolation, NUMA hop costs, or virtualization (WSL2, K8s), you'll see 5–15 µs
on consumer x86 + PCIe. Bare-metal Linux, server CPU, GPU pinned to the
right NUMA node, IRQ shielding all matter.

Numbers you'll see in HFT marketing (sub-microsecond) are **not** this pattern —
they're NIC→GPU GPUDirect RDMA where the network card DMAs straight into GPU
memory and a persistent kernel polls it; the host CPU is out of the loop
entirely. See §8 below.

### 4.2 Ordering and the volatile-fence-`__nanosleep` trio

Three things to watch out for:

- **`volatile`**. Without it, the compiler caches the doorbell read. The kernel
  loop becomes infinite.
- **`__threadfence()`** before signaling. Otherwise the host might see the
  doorbell go to 0 (work done) but the actual output writes haven't reached
  DRAM yet. (For host-mapped memory specifically, use
  `__threadfence_system()`, which orders against the host CPU's view; see
  §5 / `ring_buffer.cu`.)
- **`__nanosleep(N)`**. Pure spinning saturates the SM at full clock and dumps
  watts as heat for nothing. `__nanosleep` (sm_70+) gates the clock briefly;
  100–1000 ns is a good range. Set N=0 if you absolutely cannot afford the
  jitter.
- **Spin-loops are expensive on the SM.** A persistent kernel ties up an SM
  *forever*. Only do this for the small number of latency-critical streams.

**Host-side ordering — why does `*resp_host = 0; *dbell_host = 1;` work?**
Two plain stores, no fence. On x86 this is fine because TSO (total-store-
ordering) guarantees that stores by a single core become visible to other
agents in program order. The GPU, polling host-mapped pinned memory over
PCIe's coherent fabric, observes `resp = 0` before `dbell = 1`. We needed
the *clear* to happen first so the host doesn't see a stale "done" from the
previous round-trip after arming the new request.

On a weakly-ordered CPU (AArch64, POWER) this is a bug: insert a
`std::atomic_thread_fence(std::memory_order_release)` between the two
stores, and prefer `std::atomic` with explicit release-semantics on the
doorbell. Don't port this pattern off x86 without auditing for that.

> Forward-ref: see appendix 13 for the PTX form of `__threadfence` /
> `__threadfence_system` (`membar.gl` / `membar.sys`) and the SASS
> equivalents.

## 5. The kernel queue: SPSC ring buffer + persistent consumer

The doorbell is one bit. A real low-latency dispatcher is a **ring buffer**:
multiple host threads enqueue work items concurrently into a fixed-capacity
ring, and the persistent kernel consumes them one at a time.

This is closer to what an HFT shop's risk engine or a streaming trade-signal
pipeline looks like in shape: many independent inputs (market-data feed
threads, order-book updaters), one hot consumer that must respond in single
microseconds.

See **`ring_buffer.cu`** for a complete working implementation. Highlights:

- **Layout.** Ring of `Slot{seq, x, y, out}` in host-mapped pinned memory.
  `seq` is the per-slot state machine (`EMPTY → READY → DONE → EMPTY`).
  Producers write `x, y` then *release* with `seq = READY`; the consumer
  *acquires* by spinning on `seq == READY`, executes, and posts
  `seq = DONE` after a system-scope fence.
- **Why `__threadfence_system()` and not `__threadfence()`.** On the GPU side,
  `__threadfence` orders against other GPU threads only. Host-mapped pinned
  memory is read by the host CPU, not other GPU threads — we need
  `membar.sys` semantics so the host sees the result write before the seq
  flip. The slowest fence in CUDA, but unavoidable for host-coherent
  signaling.
- **MPSC on the host side.** Multiple producer threads contend for slots
  via `std::atomic.fetch_add(head)` to grab a sequence number, then
  publish in sequence-number order. The consumer is single-threaded
  (one thread of one warp) for simplicity; making the consumer multi-warp
  is a stretch exercise.
- **Backpressure.** A producer reserves slot `idx_seq & MASK`; if that
  physical slot's previous occupant hasn't been consumed, the producer
  yields. A real system would size CAP for the worst-case burst.
- **`__nanosleep` on the consumer hot loop.** Same rationale as §4.2.

Run it: `./ring_buffer` — 4 host threads × 100 items, prints PASS if all
results match `x + y`.

## 6. Megakernel: eliminate launches across the entire pipeline

The doorbell + ring buffer pattern eliminates the per-request launch cost. But
look at what an inference pipeline still does *between* requests:

```
host → ring → kernel_gemm → kernel_softmax → kernel_gemm_out → ring → host
```

Each of those three kernels is its own launch. Even with a persistent dispatcher
in front of them, the dispatcher launches `kernel_gemm`, waits, launches
`kernel_softmax`, etc. Every transition is a kernel boundary: results spill to
global, the next kernel reads them back, the scheduler does its work between.

A **megakernel** collapses all of that into one kernel that switches "phases"
internally based on a work-item descriptor:

```cuda
enum Phase { GEMM, SOFTMAX, GEMM_OUT, STOP };

struct WorkItem { Phase phase; void* args; };

__global__ void megakernel(volatile WorkQueue* wq, /* shared state ptrs */ ...) {
    while (true) {
        WorkItem w = wq->dequeue();          // spin/yield
        if (w.phase == STOP) break;

        switch (w.phase) {
            case GEMM:        gemm_phase(w.args);     break;
            case SOFTMAX:     softmax_phase(w.args);  break;
            case GEMM_OUT:    gemm_out_phase(w.args); break;
        }
        __threadfence_system();
    }
}
```

The host (or another kernel, or a producer warp inside the megakernel) enqueues
work items: one for the GEMM, one for the softmax, one for the output GEMM.
The megakernel processes them in order, never returning to the host scheduler.

### 6.1 Pros
- **Zero kernel-boundary cost.** No SM teardown/setup between phases.
- **Cross-phase state in registers/shared.** A "GEMM result then softmax" pair
  can stage the GEMM output in shared memory for the softmax to read — no
  global round-trip between phases.
- **Predictable scheduling.** You decide phase order; the runtime doesn't pick.
- **Composes with Green Contexts (§7).** Pin the megakernel to N SMs and it
  never competes with batch work.

### 6.2 Cons
- **Worst-case sizing.** Shared memory and registers are sized for the fattest
  phase. If `gemm_phase` needs 96 KB shared and `softmax_phase` needs 4 KB, you
  pay 96 KB for both. Same for register pressure.
- **Poor utilization on unbalanced phases.** If GEMM takes 100 µs and softmax
  takes 1 µs, the SMs sit half-idle during softmax (launch geometry was chosen
  for GEMM).
- **No library composition.** You can't use cuBLAS or cuDNN inside a megakernel
  — those expect their own launches. You're writing every phase by hand.
- **Instruction-cache pressure.** Big switch statement, lots of code paths
  resident; you can blow the I-cache and stall.
- **Debugging is harder.** No per-kernel `ncu` profile; you profile the one
  kernel and have to reason about which phase the time went to.

### 6.3 When it wins
- Pipeline of small-to-medium kernels (each « 100 µs).
- Same shape every iteration (otherwise per-phase code paths multiply).
- Latency-bound (you'd take a 20% throughput hit for a 5 µs latency win).
- You're already on persistent kernels (§4–5); megakernel is the next step.

For training-shaped throughput on H100/B200, the answer is the opposite: huge
per-kernel tiles, cuBLAS/cuDNN, no megakernel. Know which regime you're in.

### 6.4 Sketch: GEMM → softmax → GEMM as a megakernel

The full implementation is **M12 Capstone Project D** (the megakernel variant
of Project B). A simplified phase dispatcher:

```cuda
__device__ void gemm_phase(GemmArgs* a) {
    // Reuse v6 of Module 6's tile ladder, but parameters come from `a`
    // (work-item args) instead of kernel parameters. Shared-memory tiles
    // are allocated once at kernel entry, not per phase.
}

__device__ void softmax_phase(SoftmaxArgs* a) {
    // Online-softmax row reduction (Module 5 / Module 9).
    // Read input from the SAME shared memory the previous GEMM wrote to,
    // if topology allows. This is the key cross-phase win.
}
```

The win is that `gemm_phase`'s output buffer in shared memory is still hot
when `softmax_phase` runs — no DRAM round-trip. With 100 KB shared on Ada
this is a real win at small batch sizes.

For large batches, the megakernel loses to a well-tuned cuBLAS+cuDNN
sequence (which uses huge per-kernel tile sizes and amortizes any boundary
cost). The megakernel's home is small-shape, latency-bound inference.

## 7. SM partitioning: Green Contexts and friends

Persistent kernels (§4) and megakernels (§6) make per-request launches free,
but they park SMs forever. If one persistent low-latency kernel is running on a
4090 with 128 SMs, you've burned all 128 even when traffic is zero — and you
can't run anything else alongside it without resource contention.

The fix: **statically partition SMs between latency-critical and batch
workloads.** Modern CUDA (12.4+) gives you a real, hardware-enforced way:
**Green Contexts**.

### 7.1 CUDA Green Contexts (CUDA 12.4+, Ada-supported)

A Green Context is a CUDA context bound to a *subset* of SMs. Kernels launched
into it physically only run on those SMs:

```cpp
#include <cuda.h>

CUdevice dev;
cuDeviceGet(&dev, 0);

// Get the SM resource for this device.
CUdevResource sm_resource;
cuDeviceGetDevResource(dev, &sm_resource, CU_DEV_RESOURCE_TYPE_SM);

// Split off 8 SMs for the low-latency path; remainder goes to batch.
unsigned n_groups = 1;
CUdevResource lowlat_group[1];
CUdevResource batch_remainder;
cuDevSmResourceSplitByCount(
    lowlat_group, &n_groups, &sm_resource,
    &batch_remainder,
    CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING,
    /* count = */ 8);

// Build a green context bound to those 8 SMs.
CUgreenCtx green;
cuGreenCtxCreate(&green, lowlat_group[0], dev,
                 CU_GREEN_CTX_DEFAULT_STREAM);

// Get the green context's primary stream.
CUstream green_stream;
cuGreenCtxStreamCreate(&green_stream, green,
                       CU_STREAM_NON_BLOCKING, /* priority */ 0);

// Launch the persistent kernel into it.
cuLaunchKernel(persistent_kernel, /* grid */ 8, 1, 1,
               /* block */ 256, 1, 1, 0, green_stream, args, nullptr);
```

The other 120 SMs remain free for batch ML inference, training, whatever — and
they're not slowed down by the persistent kernel's spin loop. Green Contexts is
the production answer for HFT-shaped designs that need to coexist with
non-latency-critical work on the same GPU.

### 7.2 The `%smid` hack (the educational warm-up)

Before Green Contexts (CUDA < 12.4), the way to do this was to launch enough
blocks to cover all SMs and have each block read its `%smid` and exit early
if it didn't land on a wanted SM:

```cuda
__device__ unsigned smid() {
    unsigned id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

__global__ void filtered_persistent(uint32_t allowed_sms_mask, /* ... */) {
    if (((1u << smid()) & allowed_sms_mask) == 0) return;
    // real work
}
```

Wasteful (you launch many blocks that immediately exit), and you have to
oversaturate the launch grid, but it works on every CUDA version. Useful as
a backward-compat fallback or for understanding what Green Contexts do at
hardware level.

A 4090 has 128 SMs; the `%smid` returns 0..127 for the actual SM serving
the block. (`%nsmid` returns the count.)

### 7.3 Stream priorities

`cudaStreamCreateWithPriority` gives streams a priority level. Higher-priority
work can preempt lower-priority work. This is *prioritization*, not
*partitioning* — both streams share the same SMs, but the high-priority one
gets favored by the scheduler.

```cpp
int low_pri, high_pri;
cudaDeviceGetStreamPriorityRange(&low_pri, &high_pri);
cudaStream_t s_hot, s_batch;
cudaStreamCreateWithPriority(&s_hot,   cudaStreamNonBlocking, high_pri);
cudaStreamCreateWithPriority(&s_batch, cudaStreamNonBlocking, low_pri);
```

Useful when you don't have enough latency-critical traffic to dedicate SMs
permanently — you want batch kernels to run when the hot path is idle, and
yield when the hot path has work.

### 7.4 Composition

The full HFT-shaped design composes everything in this module:

| Layer | Mechanism |
|---|---|
| Reduce per-launch overhead | CUDA Graphs (§2) |
| Eliminate per-request launches | Persistent kernel + ring buffer (§4–5) |
| Eliminate per-pipeline launches | Megakernel (§6) |
| Hardware-isolate the hot path | Green Context on N SMs (§7.1) |
| Prioritize when sharing | Stream priorities (§7.3) |
| Bypass host CPU on input | GPUDirect RDMA (§8) |

### 7.5 Exercise

Take `ring_buffer.cu` (§5). Modify the launch to use a Green Context bound to
16 SMs. Verify with an in-kernel `%smid` print that only those 16 SMs ever
serve work items. (Hint: you'll need the driver API (`cu*`) for
`cuGreenCtxCreate`; the runtime API (`cuda*`) doesn't expose green contexts
as of CUDA 12.6. Use `cuInit(0)` once, then mix `cu*` and `cuda*` freely.)

## 8. Doorbell vs. production HFT: GPUDirect RDMA

The pattern in §4 / §5 has the host CPU on the critical path. Even with PCIe
Gen4 the round-trip is 2–15 µs depending on conditions. For real
low-microsecond market-making, you want the host CPU **out of the loop**.

That's what **GPUDirect RDMA** does. A network interface card (NIC) with
RDMA capability (Mellanox/NVIDIA ConnectX, etc.) and a GPU on the same PCIe
root complex can DMA *directly* between NIC buffers and GPU memory. The
host CPU programs the NIC once at setup; after that, packets arriving at
the NIC land in GPU-resident pinned memory and a persistent kernel sees
them via the same doorbell pattern as §4 — except now the "doorbell"
is the NIC's tail pointer, updated by the NIC itself.

This is **educational territory only** in this course — we can't run real
GPUDirect RDMA without an RDMA-capable NIC on the same root complex as
your GPU. But here's where to look:

| Production users of GPUDirect RDMA | What for |
|---|---|
| **NCCL** | multi-GPU collectives over IB or RoCE; the canonical user |
| **GDRCopy** | Mellanox library exposing GPU memory to CPU code via CPU mmap; useful for debugging RDMA paths |
| **GPUDirect Storage** | NVMe → GPU directly, bypassing CPU bounce buffers |
| **DOCA / DPDK + GPU** | low-latency packet processing pipelines (HFT, telco) |
| **Spectrum-X / BlueField DPU** | latest-gen NIC offload + GPUDirect, sub-microsecond NIC→GPU latency |

Reference reading: NVIDIA's GPUDirect RDMA developer guide,
`docs.nvidia.com/cuda/gpudirect-rdma/`. The CUDA primitives we use here
(host-mapped pinned, `__threadfence_system`, persistent kernel, `volatile`
spin-loops) are exactly the same; only the *source* of the doorbell write
changes from "host CPU store" to "NIC DMA write".

So: doorbell = the *educational* shape of HFT. GPUDirect RDMA = the
*production* shape, identical kernel-side, different host setup.

## 9. Pinned host memory

Pageable host memory can't be DMA'd directly — the OS might page it out
mid-transfer. So `cudaMemcpy` from pageable memory has to first copy into a
pinned staging buffer inside the driver. Two costs:

- The extra copy itself.
- The DMA can't overlap with anything if the source isn't pinned (the staging
  step is synchronous in that sense).

Pinned (page-locked) memory:

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));   // or cudaHostAlloc
// ... use ...
cudaFreeHost(h_pinned);
```

Bandwidth: ~25 GB/s (PCIe Gen4 x16) for pinned, ~6 GB/s for pageable on this
machine. **4× difference for free** — and pinned enables `cudaMemcpyAsync`
which can overlap with kernels on other streams.

The downside is that pinned memory is uncopyable from the OS's perspective and
caps total RAM that can be locked. Don't pin gigabytes; pin the staging
buffers you actually need.

### Unified memory and `cudaMemPrefetchAsync`

If you use `cudaMallocManaged` (unified memory), the runtime migrates pages
on first touch by the GPU — every page-fault is hundreds of microseconds.
For low-latency code paths you want the migration done **before** the
kernel runs:

```cpp
float* m;
cudaMallocManaged(&m, N * sizeof(float));
// ... host populates m ...

cudaMemPrefetchAsync(m, N * sizeof(float), dev,            stream);
my_kernel<<<g, b, 0, stream>>>(m, N);
cudaMemPrefetchAsync(m, N * sizeof(float), cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
// host reads m without paying migration cost on the critical path
```

`cudaMemPrefetchAsync` requires `cudaDevAttrConcurrentManagedAccess == 1`
on the device. This is true on bare-metal Linux + Pascal-or-newer; **it is
NOT true under WSL2** (the events_demo will skip its prefetch test). On
WSL2, fall back to explicit `cudaMalloc` + `cudaMemcpyAsync`.

See `events_demo.cu` for a working snippet (gracefully skips on WSL2).

## 10. Streams

Streams enable overlap between independent kernels and between kernels and
memory copies. The default stream is *synchronous* — every kernel queued there
waits for the previous one. Non-default streams run independently.

```cpp
cudaStream_t s1, s2, s3, s4;
cudaStreamCreate(&s1); /* ... */

kernel<<<g, b, 0, s1>>>(d1);
kernel<<<g, b, 0, s2>>>(d2);
kernel<<<g, b, 0, s3>>>(d3);
kernel<<<g, b, 0, s4>>>(d4);
cudaDeviceSynchronize();
```

The runtime decides when each kernel actually runs based on resource
availability. For *small* kernels that don't fill the GPU on their own, four
streams of small kernels will overlap and finish in roughly the time of one.
For *large* kernels that already saturate the GPU, streams provide no benefit.

Streams are also how you overlap a `cudaMemcpyAsync` with kernel work:
issue the copy on `s1`, the kernel on `s2`, and the two run concurrently.

When stream A's work depends on stream B's result without bouncing through
the host, use a CUDA event with `cudaStreamWaitEvent` (§3).

## 11. Beyond Ada: what evolves on Hopper / Blackwell

The patterns in this module — persistent + ring + megakernel + Green Contexts
— translate forward to Hopper (sm_90) and Blackwell (sm_100), but those
architectures add primitives worth knowing about.

### 11.1 Thread Block Clusters and DSMEM

A **cluster** is a group of blocks (up to 8 on H100, 16 on H200) that schedule
to neighbor SMs and can read/write each other's `__shared__` memory directly
via the SM-to-SM interconnect. That access — **DSMEM** (distributed shared
memory) — has near-shared-memory latency, much faster than going through L2
or DRAM.

Why it's relevant to low-latency: DSMEM lets cooperating blocks exchange data
without a `__threadfence` + global write + global read cycle. For a
megakernel split across multiple blocks (because one block ran out of shared
memory or registers), DSMEM replaces slow inter-block synchronization with a
fast SM-to-SM one.

It's not directly a launch-elimination feature in the M11 sense — it's a
data-locality / parallelism feature. But it *enables* megakernel designs
that on Ada would have to either fit in one block's resources or pay the
global round-trip cost.

### 11.2 TMA: `cp.async.bulk.tensor`

Hopper extends `cp.async` (Module 8) with bulk tensor copies driven by
hardware descriptors. From a low-latency lens: TMA reduces the warp's work
to issue a load — the SM computes addresses for the entire tile in
hardware. Less per-load instruction cost = lower latency for the warp
that issues the load. On Hopper, TMA + cluster + DSMEM is the canonical
"FlashAttention-3-shaped" design.

### 11.3 MIG (A100 / H100 / H200)

Multi-Instance GPU is *hard* hardware partitioning — the GPU is split into
1, 2, 3, 4, or 7 instances each with its own SMs, L2 slice, and memory
bandwidth. Stronger isolation than Green Contexts (which still share L2 and
DRAM bandwidth). Not on consumer / Ada / Blackwell-consumer cards. The
right tool when latency-critical workloads need to be hardware-isolated
from co-tenants on the same physical GPU.

### 11.4 What stays the same

The kernel-side primitives are unchanged across generations: `volatile`,
`__threadfence_system`, `__nanosleep`, persistent loops, doorbell flags.
Hopper/Blackwell give you better *plumbing* for cross-block coordination
and tighter SM partitioning, but the patterns from M11 are the foundation.
Modern frameworks (vLLM, TensorRT-LLM, FlashInfer) all build on this same
shape; the difference is what hardware features they exploit underneath.

---

## Exercises

`bench.cu` measures the launch-overhead and graph patterns; `persistent_demo.cu`,
`events_demo.cu`, and `ring_buffer.cu` are runnable demos. The exercises are
modifications:

1. **Build a CUDA Graph manually.** In `starter.cu`, the harness loads three
   sequential kernels into a stream-captured graph; reproduce the equivalent
   thing without stream capture, using `cudaGraphCreate` /
   `cudaGraphAddKernelNode` directly. (The explicit API is verbose but exposes
   all the knobs — node ordering, dependencies, kernel parameters.)
2. **Modify the persistent kernel** to handle a tiny workload with a
   parameter passed via a second flag. Measure how much slower the round-trip
   gets when the kernel actually does work vs. just signaling.
3. **Cross-stream sync.** Take `events_demo.cu` and replace the
   `cudaStreamWaitEvent` with a `cudaStreamSynchronize(sA)` before launching
   on `sB`. Profile both with `nsys`; see how the host-sync version
   serializes work that should overlap.
4. **Ring-buffer extension.** Modify `ring_buffer.cu` to make the consumer
   multi-warp: warp 0 polls and dispatches, warps 1+ execute the work payload.
   Useful for any per-task workload bigger than one thread can do alone.
5. **(Stretch)** Build a 4-stream pipeline that overlaps `cudaMemcpyAsync(H2D)`,
   a kernel, and `cudaMemcpyAsync(D2H)` for a streaming workload. Use `nsys`
   to confirm the timeline overlap.
6. **(Stretch)** Replace the host CPU producer in `ring_buffer.cu` with a
   second persistent producer kernel on a different stream — GPU-to-GPU
   queue. (Hint: this is the building block for inter-kernel pipelines on
   one device.)

---

## Profiler checklist

```bash
make
./bench                       # launch overhead, graphs, pinned-vs-pageable
./events_demo                 # event-based timing + cross-stream wait + prefetch
./persistent_demo             # measured doorbell RTT
./ring_buffer                 # MPSC ring on host, SPSC consumer kernel
nsys profile --stats=true ./bench   # see overlap on the timeline
```

Look at:

- **Kernel launch latency** in `bench.cu`'s output: empty kernel sequential
  vs. CUDA Graph replay.
- **Doorbell round-trip latency**: 2–5 µs on this machine (RTX 4090, WSL2,
  PCIe Gen4) from `./persistent_demo`, with run-to-run spread driven by
  host idle state and PCIe state. Bare-metal can be tighter; busier hosts
  can climb to 5–15 µs.
- **Pinned vs pageable bandwidth**: 4× difference is typical.
- **Ring buffer per-item latency**: with 4 contending host threads, a
  good run is a few µs/item amortized.
- **`nsys` timeline**: stream overlap is visible as kernels stacked on
  different rows in the timeline.

## Key takeaways

The five-rung ladder, in one place:

- **Reduce** per-launch overhead — **CUDA Graphs** (§2) turn N kernel launches
  into one when the same DAG repeats with the same shape.
- **Eliminate** per-request launches — **Persistent kernel + doorbell** (§4)
  or **ring buffer** (§5). One launch per *lifetime*, not per request. The
  kernel-side primitive is `volatile` + `__threadfence_system()` +
  `__nanosleep`. Cost: a permanently parked SM.
- **Eliminate** per-pipeline launches — **Megakernel** (§6). Switch phases
  internally via a work-item descriptor; no kernel boundary between phases.
  Cross-phase state lives in registers/shared. Cost: worst-case sizing,
  no library composition.
- **Partition** SMs — **Green Contexts** (§7.1) hardware-isolate the hot
  path on N SMs while the other (128 − N) run batch work. The `%smid` hack
  (§7.2) is the educational warm-up.
- **Bypass** the host CPU — **GPUDirect RDMA** (§8). Same kernel-side
  primitives; the doorbell is written by the NIC, not the CPU.

Connective tissue: **CUDA events** + `cudaStreamWaitEvent` (§3) express
dynamic inter-stream dependencies. **Pinned memory + async copies** (§9)
are the difference between PCIe-bound workloads being fast vs slow.
**Streams** (§10) make sense only for kernels that don't saturate the GPU
individually. **Stream priorities** (§7.3) prioritize when partitioning
isn't an option.

Measured on this machine (RTX 4090, WSL2, PCIe Gen4): pinned doorbell +
persistent kernel + pinned response = **2–5 µs end-to-end round trip**.
Going further (sub-µs) requires GPUDirect RDMA (§8), which we describe but
can't run without an RDMA-capable NIC.

Forward to Hopper: §11 covers what changes (DSMEM, TMA, MIG) and what
stays the same (every kernel-side primitive in this module).
