# Module 2 — Memory hierarchy & coalescing

**Goal:** by the end of this module you should be able to (a) draw the GPU memory
hierarchy from registers down to DRAM, (b) recognize a coalesced vs. uncoalesced access
pattern at a glance, and (c) hit > 90 % of DRAM peak on a streaming copy.

This is *the* most important module for the rest of the course. Almost every kernel you
write — until you start fusing things — is bandwidth-bound, which means optimizing it
means reducing wasted bytes off DRAM.

---

## 1. The memory hierarchy

```
                  size           latency        bandwidth (4090)
Registers          64 KB / SM partition   1 cycle        many TB/s aggregate
                   (256 KB / SM total)
Shared / L1       128 KB / SM     ~30 cycles     ~20 TB/s aggregate
L2                 72 MB          ~200 cycles    ~5 TB/s
DRAM (GDDR6X)      24 GB          ~500 cycles    ~1008 GB/s
```

Two things to internalize:

1. **DRAM is ~1000× slower than registers.** Hiding that latency requires either
   massive parallelism (Module 1) or moving data into faster levels and reusing it
   (Modules 3+).
2. **DRAM bandwidth is finite and shared.** All 128 SMs draw from the same memory bus.
   If a kernel asks for more bytes than it strictly needs, every wasted byte directly
   reduces throughput.

## 2. Sectors and cache lines

DRAM is not byte-addressable from the SM's point of view. The memory subsystem moves
data in **128-byte cache lines**, which themselves are split into four **32-byte
sectors**. When a warp issues a load, the L1/L2 controllers count *which sectors are
needed* and fetch each one once.

For a warp of 32 threads each loading a 4-byte float, the warp asks for 128 useful
bytes (= one cache line = 4 sectors). The hardware always counts in *sectors*. The
question is: **how many sectors does the address pattern actually span?**

| pattern                               | sectors fetched, used | bytes useful / fetched | efficiency |
|---------------------------------------|-----------------------|------------------------|------------|
| consecutive (`a[gid]`)                | 4 sectors fetched, 4 used | 128 / 128            | 100 %      |
| stride 2 (`a[gid*2]`)                 | 8 sectors fetched, 4 used | 128 / 256            | 50 %       |
| stride 4 (`a[gid*4]`)                 | 16 sectors fetched, 4 used | 128 / 512           | 25 %       |
| stride 32 (`a[gid*32]`)               | 32 sectors fetched, 4 used | 128 / 1024          | ~12 %      |
| fully scattered (each lane → different 128 B line) | 32 lines × 4 sectors = 128 sectors fetched, 32 used | 128 / 4096 | ~3 % |

The last row is the worst case: each lane lives in a *different* 128-byte cache line,
so we fetch 32 lines (= 128 sectors) just to use one float (one sector partially) from
each. We pay 32× the bandwidth for the same amount of useful data. **This is what
"uncoalesced" means.** The starter's `copy_strided` kernel reproduces exactly this
pattern (each warp's 32 lanes touch 32 distinct lines), and the bench will measure the
3 %-ish efficiency end-to-end.

> Note: `ncu`'s `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request` reports
> sectors *per request*. For a fully coalesced 32-lane warp, this should be near 4. For
> the fully-scattered worst case, expect ~32.

Newer GPUs are forgiving — the L2 will absorb some of the damage if your access
pattern is *somewhat* close to coalesced — but you should never rely on it.

## 3. The coalescing rule, simplified

A warp's accesses to global memory are coalesced when threads in the warp touch
addresses that fall into a small number of consecutive 32-byte sectors. The simplest
recipe:

> Thread `i` of the warp accesses element `base + i` of an array of natively-aligned
> elements (4 / 8 / 16 bytes).

You'll see this idiom everywhere:

```cpp
int gid = blockIdx.x * blockDim.x + threadIdx.x;
out[gid] = f(in[gid]);
```

Lane 0..31 of each warp hits 32 consecutive elements → one cache line, perfect
coalescing. Whenever you index global memory, ask: *as `threadIdx.x` increments by 1,
how does my address change?* If by 1 element, you're coalesced. If by anything else,
you're paying.

## 4. Vectorized loads: `float4`

A single thread can load 16 bytes at once with `float4`:

```cpp
const float4* a4 = reinterpret_cast<const float4*>(a);
float4 v = a4[gid];          // one 128-bit load (LDG.E.128)
```

Now lane 0..31 of a warp pulls 32 × 16 = 512 bytes — four cache lines per instruction.
The benefits are subtle but real:

- **Fewer instructions** to keep the same bytes flowing → less instruction-issue pressure.
- **Better at saturating** the memory pipeline because outstanding-request slots aren't
  wasted on small loads.
- Helps a lot when you're *almost* bandwidth-bound and want to push the last 5–10 %.

There's no magic: vectorized loads still have to be coalesced. They just get there with
fewer transactions.

## 5. AoS vs SoA — the #1 real-world coalescing trap

The strided-load case from §2 is a useful teaching example, but the version you'll
actually trip over in production is **array-of-structs (AoS)** vs **struct-of-arrays
(SoA)**. Suppose you're processing 3D points:

```cpp
struct Vec3 { float x, y, z; };

__global__ void scale_aos(Vec3* p, float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) p[gid].x *= s;
}
```

This *looks* coalesced — `p[gid]` is the same idiom as before. But each `Vec3` is 12
bytes, so consecutive lanes touch addresses 0, 12, 24, 36, … . When the kernel reads
`p[gid].x`, the warp ends up touching bytes at offsets `0, 12, 24, … , 372` — 12 ×
32 = 384 bytes spanning ~4 sectors of which only the `.x` halves are useful. Worse, the
compiler often issues a *full 12-byte* (or 16-byte) load per thread because the struct
isn't naturally aligned. You end up doing 3× the byte traffic to update `.x`.

The SoA fix:

```cpp
__global__ void scale_soa(float* px, float* /*py*/, float* /*pz*/, float s, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) px[gid] *= s;
}
```

Now the warp touches `px[0..31]` — 128 contiguous bytes, one cache line, 100 %
efficient. `aos_soa.cu` in this directory is a working bench that measures the gap on
the actual GPU; expect SoA to be roughly 2–3× faster on a 4090 for sparse-component
access. Take the lesson: **lay out your data so that the dimension consecutive lanes
stride is contiguous in memory.**

## 6. The read-only cache and `__ldg`

CUDA exposes a separate **read-only data cache** (called the "constant" or "uniform"
or "read-only" path depending on the era; on Ampere/Ada it's the texture cache
repurposed). Loads through this path:

- come from a per-SM cache that's separate from the L1/shared-memory pool, so they
  don't compete for the same lines;
- are issued with `LDG.E.NC` (NC = "non-coherent" / "read-only") in SASS instead of
  the default `LDG.E`;
- can sometimes give a measurable speedup for kernels that are L1-thrashed.

You can request this path explicitly:

```cpp
float v = __ldg(&x[gid]);            // read-only cache load
```

Or — and this is what you'll see in real code — you let the compiler do it for you by
marking the pointer `const __restrict__`:

```cpp
__global__ void k(const float* __restrict__ x, ...) {
    float v = x[gid];                // compiler knows x is read-only & non-aliasing
                                     // → emits LDG.E.NC automatically
}
```

That's why every kernel in this course (and in CUTLASS, FlashAttention, etc.) uses
`const __restrict__` on every input pointer it can. The `__restrict__` tells the
compiler the pointer doesn't alias other writable pointers, which (a) opens the
read-only cache path and (b) lets the compiler hoist loads aggressively.

You don't need to write `__ldg` by hand unless you're working without `__restrict__`
or want to be explicit. Just keep the abstraction in mind — when you see `LDG.E.NC` in
a SASS dump (Module 13 will show you how to read these), that's `__ldg`.

## 7. Roofline thinking

For any kernel, ask:

```
arithmetic intensity = useful FLOPs per byte read from DRAM
```

If this is low (say, < 5 FLOP/byte for FP32), you're **bandwidth-bound** — your
ceiling is `peak_DRAM_bandwidth × intensity`. No amount of math optimization will
help. You have to either move fewer bytes (better locality) or cache more aggressively
(shared memory, registers).

Vector add: 2 FLOP per 12 bytes → ~0.17 FLOP/byte → deeply bandwidth-bound.
GEMM with cache-friendly tiling: hundreds of FLOPs per byte → compute-bound; the
optimization story shifts from memory to math throughput.

## 8. L2 effects

The RTX 4090 has a 72 MB L2. If your working set fits in L2, repeated reads come from
there at ~5 TB/s instead of 1 TB/s, and you'll see "DRAM bandwidth" reported as
*higher than DRAM peak* by naive metrics (because you're really hitting L2). Cross
the L2 boundary and bandwidth drops back to DRAM peak.

`bench.cu` sweeps `N ∈ {1M, 4M, 16M, 64M}` floats (4 MB, 16 MB, 64 MB, 256 MB) so you
can see the transition. Below ~18 MB, the data fits in L2; above 72 MB, you're firmly
in DRAM-bound territory. The interesting region is the middle: partial-fit, where the
first launch warms L2 and subsequent ones hit it.

The takeaway: when interpreting "GB/s", ask yourself whether you're measuring
**sustained DRAM** or **L2-cached** bandwidth. The number you publish should be on
data sets large enough to defeat L2 (≥ 128 MB on a 4090).

---

## Visualization

Open [`viz/thread-memory-map.html`](../viz/thread-memory-map.html) (added by the viz
track). The sliders let you change block size, stride, and AoS-vs-SoA layout; the page
colors each cache line by lanes-touching-it so you can see coalescing fail in real time.

## Exercises

> Open `starter.cu` and complete the TODOs.

**1. Coalesced copy.** Implement `copy_scalar`: `b[i] = a[i]`. Measure bandwidth
against the 1008 GB/s peak.

**2. Strided copy.** Implement `copy_strided`. Each thread reads from a permuted
index so that the 32 threads of a warp end up touching 32 *different* cache lines.
Reads strided, writes coalesced — only the read pattern changes. Compare its
bandwidth to the coalesced version.

**3. Vectorized copy.** Implement `copy_vec4` using `float4` loads/stores. The bench
will tell you whether you've gained anything; if not, look at the *issue rate* of
your loads in Nsight Compute.

### Stretch

**4.** What happens if you use `float4` with the *strided* pattern? Does vectorization
help, hurt, or do nothing? Why?

**5.** *L2 effects (now in main bench.)* Ran already by `./bench`'s size sweep — verify
that `copy_scalar` reports markedly higher bandwidth at `N = 1M` (4 MB working set,
fully L2-resident) than at `N = 64M` (256 MB, DRAM-bound). The crossover is around
18 MB (= 72 MB L2 / 4 because each iteration does read + write and the previous
launch's writes plus this launch's reads need to fit).

---

## Profiler checklist

```bash
make
ncu --set full ./solution
```

Look at:

- **`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request`**: average sectors
  per global load request. Should be near 4 for coalesced (4 sectors = 128 bytes = one
  warp's worth of floats), and balloon to ~32 for fully strided.
- **DRAM Throughput** (% of peak): coalesced should be > 90 %; strided will be much
  lower because most of the time is spent waiting on serialized line fetches, not
  actually getting useful bytes.
- **Memory Workload Analysis → "Excessive sectors"**: this is Nsight Compute literally
  shouting at you that your access pattern is wrong.

For the full counter glossary used across this course, see [Module 4 — Profiler
counters introduced in this module](../04-profiling/README.md#profiler-counters-introduced-in-this-module).

## Key takeaways

- DRAM moves data in 32-byte sectors / 128-byte lines. Anything you don't use is wasted.
- "Coalesced" = consecutive lanes of a warp hit consecutive elements.
- Vectorized (`float4`) loads reduce instruction count; they can help close the last
  few % to peak.
- AoS-vs-SoA is the same coalescing problem, dressed up. Lay out so the lane-strided
  axis is contiguous.
- `const __restrict__` opens the read-only cache path — use it on every input pointer.
- "GB/s" only means "DRAM bandwidth" if the working set exceeds L2.
- Until you reach high arithmetic intensity, bandwidth is your ceiling. Manage it.
