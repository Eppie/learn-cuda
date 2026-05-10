# Module 4 — Profiling: Nsight Compute, Nsight Systems, Compute Sanitizer

**Goal:** by the end of this module you should be able to (a) drive `ncu` and `nsys`
with intent, (b) read a Nsight Compute report well enough to identify whether a kernel
is compute-bound, memory-bound, or latency-bound, (c) catch correctness bugs (out-of-
bounds reads, race conditions, uninitialized memory) with `compute-sanitizer`, and (d)
walk through the workflow "measure → form hypothesis → fix → re-measure" using the
profiler as ground truth.

This is shorter on code and longer on tooling. There are *two* exercises at the end:
the canonical "diagnose me" kernel, and a multi-bottleneck kernel where the wrong
fix doesn't help.

---

## 1. The three tools

| tool | what it does | when |
|------|--------------|------|
| **`ncu`** (Nsight Compute) | Per-kernel deep dive. Hundreds of counters: occupancy, stall reasons, sector efficiency, bank conflicts, source-line attribution. | When you've already identified *which* kernel is slow and want to know *why*. |
| **`nsys`** (Nsight Systems) | Whole-process timeline. Kernel launches, memcopies, CPU↔GPU sync points, stream overlap. | When you want a bird's-eye view: which kernel dominates, are launches stalling, does CPU pre/post-processing fit in the gaps. |
| **`compute-sanitizer`** | Correctness checker (memcheck, racecheck, synccheck, initcheck). | Whenever a kernel produces wrong output, hangs, or "works on Tuesdays". Run it before you tune for speed. |

Rule of thumb: `compute-sanitizer` first to make sure the kernel is *correct*, then
`nsys` to find the hot kernel and check overlap, then `ncu` to optimize that kernel.

## 2. Compute Sanitizer — the second tool every CUDA dev needs

Speed without correctness is a fast lie. The four sub-tools:

| sub-tool      | catches                                                                  |
|---------------|--------------------------------------------------------------------------|
| `memcheck`    | out-of-bounds reads/writes, misaligned accesses, leaks                   |
| `racecheck`   | shared-memory races (writes from one warp racing reads from another)     |
| `synccheck`   | divergent `__syncthreads()` (dead-locks, wrong-branch participation)     |
| `initcheck`   | reads from uninitialized device memory                                   |

Usage is identical: `compute-sanitizer --tool=<name> ./your_binary`. Default is
`memcheck`. A clean memcheck run on Module 3's solution:

```text
$ compute-sanitizer --tool=memcheck ./solution
========= COMPUTE-SANITIZER
transpose_naive                errors=0
transpose_shared               errors=0
transpose_shared_padded        errors=0
transpose_shared_4rows         errors=0
transpose_shared_dynamic       errors=0
transpose_shared_vec4          errors=0
========= ERROR SUMMARY: 0 errors
```

When something *is* wrong, you get a stack-traced report pointing at the offending
line:

```text
========= Invalid __global__ read of size 4 bytes
=========     at vector_add+0x250
=========     by thread (32,0,0) in block (0,0,0)
=========     Address 0x7f12...0040 is out of bounds
```

Run memcheck (and once for each of the other three) on every kernel in this course
the first time it builds. It's not slow enough to skip — typical overhead is 2–5×
runtime. **It catches bugs that show up as "kernel returns the right answer 99 % of
the time"** in production, which are the worst kind.

## 3. Running `ncu` and `nsys`

```bash
# Build with -lineinfo (already in our Makefiles) so source-line attribution works.

# Whole-program timeline + summary stats (text):
nsys profile --stats=true ./bench

# All counter sets for a single kernel run:
ncu --set full ./bench

# Cheaper: only the "speed of light" summary (top-level limiter analysis):
ncu --set basic ./bench

# Filter to one kernel by regex (kernel name from the binary's mangled symbols):
ncu --kernel-name 'transpose_shared' --set full ./bench
```

`ncu` runs each kernel many extra times to gather counters. Expect a kernel that
finishes in 0.2 ms to take a couple of seconds under `--set full`. Don't profile a
training loop with `--set full` — use `--set basic` or `nsys`.

### Other useful tools

- **`nvidia-smi`** — quick "is the GPU alive, what's its temp, who's using it" check.
  `nvidia-smi -l 1` polls every second. Spotcheck before profiling: temperature, power,
  utilization. If your GPU is thermally throttling at 84 °C, your benchmarks are lying.
- **`ncu-ui`** — the GUI version of Nsight Compute. Open a `.ncu-rep` file produced by
  `ncu --export`. **WSL note:** ncu-ui doesn't run inside WSL2 directly; install the
  Windows-host version of Nsight Compute and open the `.ncu-rep` from the Windows side.
  Workflow: `ncu --set full --export report ./bench` in WSL → open
  `\\wsl$\Ubuntu\home\you\…\report.ncu-rep` in the Windows ncu-ui. The graphical
  source view and the per-bank shared-memory chart are worth the friction.
- **`dcgm`** — datacenter-grade telemetry / diagnostics. You almost certainly don't
  need this; it's mentioned because professional ML clusters use `dcgm-exporter` to
  feed Prometheus.

## 4. Reading an `ncu` report — the four screens that matter

When you run `ncu --set full`, you get a long report. The 80/20 of it:

### a) GPU Speed Of Light

Top of the report. One paragraph and a chart. It tells you the **dominant limiter**:
"Compute Throughput", "Memory Throughput", or one of the latency reasons. This is
where you should always start.

### b) Compute Workload Analysis & Memory Workload Analysis

Two sister sections. Each one gives:

- **% of peak achieved** for that resource
- A breakdown by sub-unit (which pipeline / which memory level)
- Specific issue callouts ("Excessive Sectors", "Uncoalesced Global Memory Access")

These are where Nsight Compute literally tells you what's wrong, in English.

### c) Launch Statistics

Block size, grid size, registers per thread, shared memory per block, **achieved
occupancy**. If achieved occupancy is much lower than theoretical, dig further — maybe
the block is too big, or registers are spilling, or shared memory caps the resident
blocks.

### d) Source view

Line-by-line counters mapped back to your `.cu` (this is what `-lineinfo` enables in
the Makefiles). The most useful columns:

- **Sampling Data (All)** — where time is spent
- **Memory operations** — sectors / requests, divergence
- **Stall reasons** — what the warp was waiting on at that line

You can find a single bad memory load this way.

## 5. A real `ncu` walkthrough — Module 2's three copy kernels

Below is **actual** `ncu --set basic` output captured on this RTX 4090 from the
`02-memory-coalescing/solution` binary, with annotations explaining what each line
tells you.

### 5a) `copy_scalar` — the well-behaved baseline

```text
copy_scalar(const float *, float *, int) (131072, 1, 1)x(256, 1, 1), CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.24
    SM Frequency                    Ghz         2.23
    Memory Throughput                 %        93.58   # ← deeply memory-bound; this is good for a copy
    DRAM Throughput                   %        93.58   # ← we are at 94% of DRAM peak
    Duration                         us       242.21
    L1/TEX Cache Throughput           %        13.55
    L2 Cache Throughput               %        34.18
    SM Active Cycles              cycle    527851.06
    Compute (SM) Throughput           %         8.96   # ← compute is idle, as expected
    ----------------------- ----------- ------------

    INF   The kernel is utilizing greater than 80.0% of the available compute or
          memory performance of the device.            # ← ncu is confirming "saturated"
```

Interpretation: this kernel is doing exactly what it should. Memory throughput at 94 %
of peak, compute throughput at 9 %. **You can't speed this up by optimizing arithmetic**
— there is no arithmetic. To go faster, move fewer bytes (impossible for a copy) or
exploit L2 (smaller working set).

`Achieved Occupancy = 82.5 %` — the slight gap to 100 % is the standard launch ramp-up
/ tail effects across 131072 blocks. Not a problem.

### 5b) `copy_strided` — the "what does uncoalesced look like" kernel

```text
copy_strided(const float *, float *, int) (131072, 1, 1)x(256, 1, 1), CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Memory Throughput                 %        84.61   # ← high again, but…
    DRAM Throughput                   %        84.61
    Duration                         ms         1.48   # ← 6× slower than copy_scalar
    L2 Cache Throughput               %        37.89
    Compute (SM) Throughput           %         2.76
    ----------------------- ----------- ------------
```

This is a **gotcha**: DRAM throughput is *also* high (~85 %)! The DRAM bus is busy
moving bytes — those bytes just aren't useful. The kernel duration is 1.48 ms vs 0.24
ms for `copy_scalar`. Same data moved, ~6× slower.

The real story is in the **per-request sector count** (visible under
`--set memory_workload` or via the source view). Where `copy_scalar`'s loads averaged
~4 sectors per request (a single line per warp), `copy_strided` averages closer to 32
— each warp is fetching 32 distinct lines and using only one float from each.

**Lesson:** "DRAM is 84 % busy" is *not* a sufficient stopping condition. Always cross-
check sector efficiency. A coalesced kernel and a fully-strided kernel can both look
"memory-bound" by Speed-of-Light alone.

### 5c) `copy_vec4` — vectorized loads, same coalescing

```text
copy_vec4(const float4 *, float4 *, int) (32768, 1, 1)x(256, 1, 1), CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Memory Throughput                 %        93.76   # ← essentially identical to scalar
    DRAM Throughput                   %        93.76
    Duration                         us       239.20   # ← 1% faster than copy_scalar
    Compute (SM) Throughput           %         2.22
    ----------------------- ----------- ------------
```

`Grid Size = 32768` (= 131072 / 4) because each thread now handles 4 floats. Memory
throughput identical, duration nearly identical. On a 4090 with 128 SMs and a wide
memory subsystem, the scalar kernel was already saturating DRAM; vectorization just
takes the same throughput with 4× fewer instructions. The win shows up in
`Compute (SM) Throughput` (2.2 % vs 9 %) — there's more compute headroom for fusing
work into the kernel later.

### Counter cheat sheet (with the values you just saw)

| counter | copy_scalar | copy_strided | copy_vec4 | what it means |
|---|---|---|---|---|
| `dram__throughput.%peak` | 93.6 % | 84.6 % | 93.8 % | DRAM bus utilization |
| Duration | 242 µs | **1480 µs** | 239 µs | wall-clock |
| `Compute Throughput` | 9.0 % | 2.8 % | 2.2 % | SM math pipeline |
| `Achieved Occupancy` | 82.5 % | 78.3 % | (similar) | warps active vs max |

If you only looked at DRAM throughput, you'd think `copy_strided` was healthy.
*Always* look at duration too.

## 6. A real `nsys` walkthrough — Module 3's transpose

```bash
nsys profile --stats=true ./solution
```

Captured output (CUDA-API host-side trace, WSL2):

```text
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Name
 --------  ---------------  ---------  ----------  ----------  ----------------------
     52.4        164327254          2  82163627.0  82163627.0  cudaMalloc
     46.1        144747795          4  36186948.8  36113657.0  cudaMemcpy
      0.9          2892556          2   1446278.0   1446278.0  cudaFree
      0.3           913819          3    304606.3    110689.0  cudaMemset
      0.3           902465          3    300821.7     80798.0  cudaLaunchKernel
```

What this tells you:

- **52 % of wall time is `cudaMalloc`** — first-call allocations are slow because
  the driver has to actually back the virtual range. In a long-running program, this
  is one-time cost. Don't optimize the kernels until you've subtracted it.
- **46 % is `cudaMemcpy`** — host↔device copies of the 256 MB matrix dominate
  everything else. If your real workload involves repeated H2D copies, that's where
  to look first (pinned host memory, async copies, overlapping with compute — see
  Module 8).
- **`cudaLaunchKernel` per launch ≈ 80 µs (median).** That's our launch overhead
  measurement from Module 1, on this exact machine. It's well above the per-kernel
  GPU time of these tiny kernels — for short kernels in tight loops, **the host
  overhead can dominate the GPU work**.

The full `nsys` UI (open the `.nsys-rep` file in nsys-ui on a native-Linux or Windows
machine) shows the same data on a timeline so you can see the gaps between kernels.

> **WSL note:** on WSL2, `nsys` does *not* capture GPU-side kernel/memcpy traces in
> the SQLite report — only the CPU-side CUDA API calls above. (The `cuda_gpu_kern_sum`
> and `cuda_gpu_mem_time_sum` reports show "SKIPPED: does not contain CUDA kernel
> data".) On native Linux you also get a per-kernel GPU duration table. The
> CPU-side trace is still the right starting point: it's where you find launch
> overhead, sync stalls, and CPU↔GPU bottlenecks.

## 7. Counters worth memorizing

These come up over and over and are the ones the rest of the course assumes you can
read. Subsequent modules' "Profiler checklist" sections cite this glossary.

## Profiler counters introduced in this module

This anchor is referenced by Modules 2, 3, and onward. If you see "see Module 4 §X
counter Y" in another module's profiler checklist, this is the table.

| counter | meaning | what's the red flag |
|---------|---------|---------------------|
| `gpu__time_duration.sum` | Total kernel wall-clock time. | (this is the answer) |
| `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | How busy the SMs were. | Low → latency-bound or under-utilized; the SMs spent most of the kernel idle. |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bus utilization vs peak (1008 GB/s on RTX 4090). | High (>80 %) → memory-bound. Low + slow runtime → access-pattern problem (see sector counter below). |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request` | Average sectors fetched per global load instruction. | Should be near 4 for coalesced (one cache line). Balloons to ~32 for fully strided / scattered access. |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum.per_request` | Same, but for stores. | Same red flag as the load version. |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | Total shared-memory bank conflicts in the kernel. | Should be 0. Anything else means lanes are serializing on shared loads/stores. |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy. | < 25 % usually leaves performance on the table for memory-bound kernels. |
| `smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued.sum` | Time warps spend stalled waiting on global memory loads ("Long Scoreboard"). | High and you're memory-bound. Normal for streaming kernels. |
| `smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued.sum` | Stalls waiting on shared memory ("Short Scoreboard"). | High → bank conflicts or shared-memory bandwidth limit. |
| `smsp__pcsamp_warps_issue_stalled_barrier_not_issued.sum` | Stalls waiting on `__syncthreads()`. | High → load imbalance among threads in a block. |
| `smsp__pcsamp_warps_issue_stalled_no_instruction_not_issued.sum` | Warps with nothing to do (compiler/register-pressure problem). | High → kernel has too few independent instructions to keep the pipeline fed. |
| `smsp__pcsamp_sample_count` | Where time is spent (source view). | Concentrate fixes on the line with the most samples. |
| `launch__registers_per_thread` | Compile-time register count. | High (>= 64) starts capping occupancy on `sm_89`. |
| `smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` | FP32 FMA utilization. | High → compute-bound on FMA pipe; low + memory-bound → bandwidth limited. |
| `smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` | Tensor-core (HMMA) utilization. Module 7 onward. | Low when you expect tensor-core throughput → check WMMA/mma.sync coverage. |

You'll also see sections labeled **"Stall Reasons"** in the report. Common ones:

- *Long Scoreboard* — waiting on global memory load. Normal for memory-bound kernels;
  a problem if your kernel should be compute-bound.
- *Short Scoreboard* — waiting on shared memory. Usually fine; if very high, look for
  bank conflicts.
- *Wait* — waiting on `__syncthreads()`. High → load imbalance among threads.
- *No Instructions* — out of things to do. Compiler / register-pressure problem.
- *Selected* — actually running. You want lots of this.

## 8. Roofline thinking via `ncu`

For a given kernel, `ncu` plots arithmetic intensity (FLOPs / bytes) against achieved
performance and shows where you are relative to the roofline:

```
   FLOP/s
     ^
peak |          /---------------  compute roof
     |         /
     |        /
     |       / *       <-- your kernel sits here
     |      /
     |     / <-- memory roof slope = bandwidth
     |    /
     +---/--------------> arithmetic intensity (FLOP/byte)
```

If your dot is on the slanted memory roof, you're memory-bound; the only way up is to
move fewer bytes (cache more, fuse, lower precision). If it's on the flat compute roof,
you're compute-bound; the only way up is to do less work per output (algorithm,
Tensor Cores).

The interactive viz at [`viz/roofline.html`](../viz/roofline.html) lets you place
several kernels onto the 4090's roofline and click each to see suggested next steps.

## 9. The workflow

1. **Run `compute-sanitizer --tool=memcheck`.** If it complains, fix that first;
   profiling broken code is wasted effort.
2. **Run `nsys`.** Identify the kernel(s) that dominate. Check that streams overlap as
   you expect; check that there aren't surprise `cudaDeviceSynchronize` calls eating
   time.
3. **Run `ncu --set basic` on the hot kernel.** Look at the Speed Of Light section.
   Identify the limiter.
4. **Form a specific hypothesis.** "Sector efficiency is 12 % → reads are uncoalesced."
   "Bank conflicts are non-zero → shared array layout is wrong." "Occupancy is 12 % →
   register pressure."
5. **Fix and re-measure.** If the limiter changed, repeat. If it didn't, your
   hypothesis was wrong.

The mistake newcomers make is fixing things that *look* slow without checking whether
they're actually the limiter. Resist that. The profiler is the truth.

---

## Exercises

### Exercise A — single-bottleneck diagnosis (the canonical puzzle)

`starter.cu` contains a small kernel — a "saxpy with a twist" — that runs much
slower than it should and *also computes the wrong answer*. Your job:

1. Build it and run `./starter`. The host-side verify will tell you it's wrong.
2. Profile: `ncu --set full ./bench` (or `--set basic` for a quicker pass). Find the
   limiter and the specific counter that explains the badness.
3. Edit the kernel in `starter.cu` to fix it. Re-run `./starter` to confirm the fix
   (PASS) and the speedup.

Hint: there's exactly one issue. It will be obvious once you look at the right counter
in `ncu`. The reference fix is in `solution.cu` — don't peek until you've found it
yourself.

### Exercise B — multi-bottleneck: where the wrong fix doesn't help

`starter_multibottleneck.cu` (build with `make starter_mb`) contains a 2D matrix-row
operation that is *simultaneously* (i) coalescing-broken on its strided global access,
(ii) using more registers than necessary (artificial spill), and (iii) launching with
a block size that caps occupancy. Three plausible fixes; only one is the actual
limiter.

The lesson: this is what real kernels look like. Don't guess. Run `ncu`, find which of
the three counters is *the* limiter, fix that one, re-profile, and the next limiter
will reveal itself. Order matters.

### Lab using earlier modules

If you want more practice without the puzzle, try:

```bash
ncu --set basic ../02-memory-coalescing/bench
ncu --set basic ../03-shared-memory/bench
```

Compare the numbers you get to the table in §5 of this README. Verify that the metrics
match the explanations in those module READMEs. The point is to build the muscle memory
of "I expect counter X to be Y for kernel Z" — once you have that, you can debug
performance problems without having seen the kernel before.

---

## Key takeaways

- Three tools, ordered: `compute-sanitizer` for correctness, `nsys` for "where is
  time spent", `ncu` for "why is this kernel slow."
- Always identify the limiter (compute / memory / latency) before changing code.
- Build with `-lineinfo` so the source view in `ncu` is useful.
- Specific counters answer specific questions. Memorize the table in §7; later
  modules cite it.
- "DRAM throughput is 90 %" doesn't always mean "fast"; the kernel could be moving
  90 % nonsense.
- Trust the profiler over your intuition. Fix what's actually slow.
