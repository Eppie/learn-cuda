# Module 13 — PTX appendix (reference)

> **How to use this appendix.** This isn't an end-of-course addendum — it's a
> reference. Other modules forward-link to specific sections here:
>
> - **M07** points to [§ mma.sync vs WMMA](#mmasync-vs-wmma) and [§ ldmatrix](#ldmatrix-loading-fragments-from-shared-memory).
> - **M08** points to [§ cp.async PTX form](#cpasync-ptx-form).
> - **M11** points to [§ __threadfence and PTX memory ordering](#__threadfence-and-ptx-memory-ordering).
>
> Read sections in isolation as you encounter them; you don't need to read
> top-to-bottom unless you want a guided tour.

**Goal:** by the end of this appendix you should be able to (a) read the PTX a
CUDA kernel compiles into, (b) write inline PTX from CUDA C++ when the
compiler won't emit what you need, and (c) inspect the SASS that finally runs
to confirm what the GPU actually does. This is the layer below everything in
the previous twelve modules.

This is an *appendix*: it adds nothing to the perf you can write in pure CUDA
C++, but it lets you understand and tweak that perf at a level the high-level
language sometimes hides.

---

## The compilation pipeline

```
.cu  ──────►  .ll      ──────►  .ptx          ──────►  .cubin       ──►  GPU
       (clang/                  (NVPTX           (ptxas: register     (SASS,
        nvcc)                   backend)         alloc + sched)       per-arch
                                                                       machine
                                                                       code)
```

Two intermediate forms matter to us:

- **PTX** is NVIDIA's virtual ISA. Register count is unlimited
  (`.reg .b32 %r<32>` declares 32 abstract 32-bit regs), memory operations are
  abstract, no specific instruction scheduling. It's stable across hardware
  generations — the same PTX file runs on Volta and Hopper, you just get
  different SASS.
- **SASS** is the actual machine code. Per-arch (`sm_89` SASS ≠ `sm_80` SASS).
  When you read PTX you're reading what the compiler *intended*; when you read
  SASS you're reading what `ptxas` and the assembler *did*. They aren't 1:1.

Inspect them with:

```bash
nvcc -arch=sm_89 -O3 --ptx vector_add.cu                        # vector_add.ptx
nvcc -arch=sm_89 -O3 --cubin -o k.cubin vector_add.cu
cuobjdump --dump-sass k.cubin                                   # human SASS
nvdisasm --print-instructions-only k.cubin                      # finer SASS
```

The `Makefile` here exposes `make ptx` and `make sass` shortcuts for every
example file in this directory.

## Reading PTX

Take the world's simplest kernel (`vector_add.cu`):

```cpp
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) c[gid] = a[gid] + b[gid];
}
```

The committed `vector_add.ptx` (regenerated from `make ptx`) starts with
directives. The exact lines below are quoted from the file with line numbers:

```ptx
// vector_add.ptx:9-11
.version 8.5
.target sm_89
.address_size 64

// vector_add.ptx:15-20
.visible .entry _Z10vector_addPKfS0_Pfi(
    .param .u64 _Z10vector_addPKfS0_Pfi_param_0,
    .param .u64 _Z10vector_addPKfS0_Pfi_param_1,
    .param .u64 _Z10vector_addPKfS0_Pfi_param_2,
    .param .u32 _Z10vector_addPKfS0_Pfi_param_3
)
```

Then a register declaration block:

```ptx
// vector_add.ptx:22-25
.reg .pred 	%p<2>;
.reg .f32 	%f<4>;
.reg .b32 	%r<6>;
.reg .b64 	%rd<11>;
```

A *predicate* register (`%p<2>`) for the bounds check, six 32-bit ints, four
fp32, and eleven 64-bit pointers / large ints. PTX gives the compiler infinite
abstract registers; ptxas decides which physical registers to allocate.

The body loads parameters, computes `gid`, checks bounds, then performs the
load + add + store. Real lines from the file:

```ptx
// vector_add.ptx:29-58
ld.param.u64 	%rd1, [_Z10vector_addPKfS0_Pfi_param_0];   // load A pointer
ld.param.u64 	%rd2, [_Z10vector_addPKfS0_Pfi_param_1];   // B pointer
ld.param.u64 	%rd3, [_Z10vector_addPKfS0_Pfi_param_2];   // C pointer
ld.param.u32 	%r2, [_Z10vector_addPKfS0_Pfi_param_3];    // n
mov.u32 	%r3, %ctaid.x;                              // blockIdx.x
mov.u32 	%r4, %ntid.x;                               // blockDim.x
mov.u32 	%r5, %tid.x;                                // threadIdx.x
mad.lo.s32 	%r1, %r3, %r4, %r5;                         // gid
setp.ge.s32 	%p1, %r1, %r2;                          // gid >= n ?
@%p1 bra 	$L__BB0_2;                                  // if so, skip to ret

cvta.to.global.u64 	%rd4, %rd1;                         // A: generic→global
mul.wide.s32 	%rd5, %r1, 4;                           // gid * 4
add.s64 	%rd6, %rd4, %rd5;                           // &a[gid]
cvta.to.global.u64 	%rd7, %rd2;                         // B: generic→global
add.s64 	%rd8, %rd7, %rd5;                           // &b[gid]
ld.global.nc.f32 	%f1, [%rd8];                        // b[gid]
ld.global.nc.f32 	%f2, [%rd6];                        // a[gid]
add.f32 	%f3, %f2, %f1;                              // a + b
cvta.to.global.u64 	%rd9, %rd3;                         // C: generic→global
add.s64 	%rd10, %rd9, %rd5;                          // &c[gid]
st.global.f32 	[%rd10], %f3;                           // c[gid] = ...
```

Useful patterns to recognize:

- `ld.param.*` — kernel arguments live in a special parameter space.
- `ld.global.*` — DRAM. Optional cache modifiers: `.ca` (default; cache
  everywhere), `.cg` (cache global only — skip L1), `.cs` (streaming),
  `.lu` (last-use), `.cv` (volatile), `.nc` (read-only cache path; the
  compiler picks this automatically when it can prove the pointer is
  `__restrict__` and read-only — that's why `vector_add.ptx` shows `.nc`
  without any inline asm).
- `cvta.to.global.u64` — convert generic address space to global. PTX's
  generic-pointer model needs an explicit cast before a `.global` load.
- `mad.lo.s32` — 3-input multiply-add, low 32 bits of product. `mad` and
  `fma` are how the compiler writes the 1-cycle multiply-add the SM has.
- `setp.<cmp>.<type>` — set predicate from comparison.
- `@%p1 bra LABEL` — predicated branch.

### What about the corresponding SASS?

The same kernel in `vector_add.sass` (Ada, sm_89). Real lines:

```sass
// vector_add.sass:5-31
/*0000*/  MOV R1, c[0x0][0x28] ;
/*0010*/  S2R R6, SR_CTAID.X ;                              // blockIdx.x
/*0020*/  S2R R3, SR_TID.X ;                                // threadIdx.x
/*0030*/  IMAD R6, R6, c[0x0][0x0], R3 ;                    // gid
/*0040*/  ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT ;      // gid >= n ?
/*0050*/  @P0 EXIT ;
/*0060*/  MOV R7, 0x4 ;                                     // sizeof(float)
/*0070*/  ULDC.64 UR4, c[0x0][0x118] ;
/*0080*/  IMAD.WIDE R4, R6, R7, c[0x0][0x168] ;             // &b[gid]
/*0090*/  IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160] ; // &a[gid] (.reuse!)
/*00a0*/  LDG.E.CONSTANT R4, [R4.64] ;                      // b[gid] (read-only)
/*00b0*/  LDG.E.CONSTANT R3, [R2.64] ;                      // a[gid] (read-only)
/*00c0*/  IMAD.WIDE R6, R6, R7, c[0x0][0x170] ;             // &c[gid]
/*00d0*/  FADD R9, R4, R3 ;                                 // a + b
/*00e0*/  STG.E [R6.64], R9 ;                               // c[gid] = ...
/*00f0*/  EXIT ;
```

Things SASS shows that PTX hides:

- **Register reuse caches.** Note `R6.reuse` and `R7.reuse` on line `/*0090*/`.
  Ada SMs have a small reuse cache that lets repeated reads of the same
  register share an operand-fetch slot. SASS shows it; PTX doesn't.
- **Constant-bank addressing.** `c[0x0][0x178]`, `c[0x0][0x160]`, etc. are
  reads from the kernel-parameter constant bank — much cheaper than reading
  from global memory. PTX's `ld.param` becomes these.
- **`LDG.E.CONSTANT`.** This is the SASS encoding for the read-only cache
  path that PTX called `ld.global.nc.f32`. The compiler picked it automatically
  because both `a` and `b` were `const __restrict__`. (Compare against the
  three-way contrast in `cache_hints.sass`, below.)
- **`FADD`, not `FFMA`.** The kernel does `a + b`, not `a*x + b`, so the
  compiler emits a plain add, not a fused multiply-add.
- **Issue scheduling.** The two `LDG.E.CONSTANT` loads issue back-to-back at
  `/*00a0*/` and `/*00b0*/` — ptxas hoisted both loads ahead of the FADD so
  their latencies overlap.

For most CUDA work you can stay in PTX; but if you're pushing the last
microsecond out of a kernel, SASS is the truth.

## Inline PTX from CUDA C++

The general form:

```cpp
asm volatile (
    "instruction operands"
    : "=output_constraint"(output_var)
    : "input_constraint"(input_var)
    : /* clobbers */
);
```

Constraint letters:

| code | meaning                                             |
|------|-----------------------------------------------------|
| `r`  | 32-bit integer register                             |
| `l`  | 64-bit integer register (also pointers)             |
| `f`  | 32-bit floating point register                      |
| `d`  | 64-bit floating point register                      |
| `h`  | 16-bit register                                     |
| `n`  | immediate (compile-time constant)                   |
| `+f` | read+write fp32 (for accumulators)                  |
| `=r` | write-only 32-bit integer                           |

### Example A — Cache hint on a global load

The compiler usually picks a smart cache modifier for you, but sometimes you
need to override:

```cpp
__device__ __forceinline__ float ld_cg(const float* __restrict__ p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
```

`cache_hints.cu` benches three variants — see [§ Cache modifiers and the
__restrict__ trap](#cache-modifiers-and-the-__restrict__-trap), below.

### Example B — `cp.async` without the wrapper

`__pipeline_memcpy_async(dst, src, 16)` from `cuda_pipeline.h` emits a
`cp.async` PTX instruction. The committed `cpasync_inline.ptx` proves this
side-by-side; see [§ cp.async PTX form](#cpasync-ptx-form), below.

## Cache modifiers and the __restrict__ trap

This section earns its own number because it's the highest-density gotcha in
the entire appendix. We'll work through `cache_hints.cu` line by line.

The premise: three loads at the same global address, with three different
cache modifiers:

```cpp
// cache_hints.cu — paraphrased
__device__ __forceinline__ float ld_default(const float* p) {        // NO __restrict__
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float ld_cg(const float* __restrict__ p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ float ld_nc(const float* __restrict__ p) {
    float v;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}
```

The pedagogical trap: **if `ld_default`'s parameter were `const float* __restrict__`,
the compiler would silently lift the dereference to `ld.global.nc.f32`** —
identical to `ld_nc`. The whole comparison would collapse to a two-way one,
even though the source code looks three-way. Removing `__restrict__` from
`ld_default` (the surrounding kernel still has it on its argument list, so
you don't lose the alias guarantee at the kernel boundary) prevents the
promotion.

Inspect `cache_hints.ptx` and grep for `ld.global` to confirm three distinct
instructions:

```
$ grep "ld.global" cache_hints.ptx
48:	ld.global.f32 %f1, [%rd3];        // copy_default kernel
93:	ld.global.cg.f32 %f1, [%rd3];     // copy_cg kernel
138:	ld.global.nc.f32 %f1, [%rd3];    // copy_nc kernel
```

Three lines, three distinct instructions. Good.

### What ptxas does with these — the SASS view

`cuobjdump --dump-sass` (run via `make sass`) on the three kernels produces
three different SASS instructions. From `cache_hints.sass`, with line numbers:

```
// cache_hints.sass — copy_default kernel (no cache hint)
/*0090*/  LDG.E R3, [R2.64] ;                    // line 129

// cache_hints.sass — copy_cg kernel (.cg → skip L1)
/*0090*/  LDG.E.STRONG.GPU R3, [R2.64] ;         // line 76

// cache_hints.sass — copy_nc kernel (.nc → read-only cache)
/*0090*/  LDG.E.CONSTANT R3, [R2.64] ;           // line 23
```

The three SASS encodings:

- **`LDG.E`** — vanilla global load through the L1 + L2 hierarchy.
- **`LDG.E.STRONG.GPU`** — strongly-ordered global load that skips L1 (the
  `.cg` modifier maps here on Ada). On consumer GPUs L1 capacity is precious;
  for streaming reads where you'll never re-read the line, skipping L1 is
  free perf.
- **`LDG.E.CONSTANT`** — read-only cache path. "Constant" in the SASS name
  is a misnomer — the data isn't constant, but the cache assumes the data
  *won't change during the kernel*, which lets it dedupe loads across warps
  more aggressively. Also routes through the read-only cache hardware (used
  to be a separate unit; on Ada it's blended with the texture cache).

You'll see `LDG.E.CONSTANT` *all over the place* in compiled CUDA code,
because most input tensors are declared `const __restrict__`. That's the
compiler taking the read-only fast path automatically. When you read
"`LDG.E.CONSTANT`" in SASS, think "the compiler proved this load is
read-only and routed it to the read-only cache" — not "this came from
constant memory" (`LDC` is constant memory).

### Bench numbers (RTX 4090)

```
ld.global (default)    ~0.57 ms   ~940 GB/s
ld.global.cg           ~0.58 ms   ~920 GB/s
ld.global.nc           ~0.58 ms   ~920 GB/s
```

On streaming workloads (every line read once) the deltas are small — the L1
isn't doing much for you anyway because the data won't be re-read before
eviction. The reason to know about cache modifiers is the *non-streaming*
case — kernels with mixed read patterns where you want to keep L1 for the
hot data and bypass it for the cold streaming data.

## cp.async PTX form

`cpasync_inline.cu` puts the wrapper and inline forms side by side in the
same compilation unit. The committed `cpasync_inline.ptx` (run `make ptx`
to regenerate) shows both translate to a `cp.async.{cg,ca}.shared.global`
instruction:

```ptx
// cpasync_inline.ptx:58 — the wrapper version
cp.async.cg.shared.global [%r8], [%rd3], 16, 16;

// cpasync_inline.ptx:145 — the inline-asm version
cp.async.ca.shared.global [%r8], [%rd3], 16;
```

Two things to notice:

1. The wrapper `__pipeline_memcpy_async` defaults to **`.cg`** (skip L1,
   cache at L2 only). The inline version we wrote uses **`.ca`**. If you
   want to match the wrapper exactly, write `.cg` in the inline asm.
2. The wrapper also emits `16, 16` (the trailing optional fill-size argument).
   `cp.async` lets you copy fewer bytes than the slot and zero-fill the
   rest; when both are equal it's a plain transfer.

Around the `cp.async`, both versions emit `cp.async.commit_group` and
`cp.async.wait_group 0` — the in-flight tracking that lets you overlap
the copy with compute.

For more on `cp.async` see Module 8.

## mma.sync vs WMMA

`mma_sync_example.cu` runs exactly one
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and verifies against
a host triple-loop matmul. Compile and run:

```bash
make mma_sync_example && ./mma_sync_example
# Device: NVIDIA GeForce RTX 4090
# max_abs=7.153e-07   max_rel=3.127e-06   PASS
```

The committed `mma_sync_example.ptx` shows the inline asm passed through
unchanged:

```ptx
// mma_sync_example.ptx:98
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
{...}, {...}, {...}, {...};
```

In SASS this becomes a single `HMMA.16816.F32` instruction — the actual
hardware tensor-core instruction:

```sass
// mma_sync_example.sass:75
/*0230*/  HMMA.16816.F32 R4, R8, R2, R4 ;
```

### Why use raw `mma.sync` instead of WMMA?

The C++ WMMA wrapper (`<mma.h>`, `nvcuda::wmma::*`) issues the same hardware
instructions but hides the lane-element mapping behind opaque `fragment`
types. That makes WMMA easy to write but hard to *fuse* with epilogues:

- You can't inspect the fragment layout, so you can't directly read fragment
  elements from the host or interleave them with custom math.
- WMMA's `m16n16k16` tile shape doesn't match the `m16n8k16` shape that the
  hardware actually issues — under the hood WMMA emits two `m16n8k16`
  instructions, hidden from you.
- All production ML kernels (CUTLASS, FlashAttention, FasterTransformer)
  drop down to raw `mma.sync` for these reasons.

The lane-element mapping for `m16n8k16` with FP16 inputs and FP32 accumulator
(reproduced from PTX ISA documentation):

| Fragment | Per-lane content | Total |
|----------|-----------------|-------|
| A (16×16, row-major, fp16) | 4 b32 = 8 fp16 | 32 lanes × 8 = 256 fp16 |
| B (16×8, col-major, fp16)  | 2 b32 = 4 fp16 | 32 lanes × 4 = 128 fp16 |
| C (16×8, row-major, fp32)  | 4 fp32         | 32 lanes × 4 = 128 fp32 |

For lane `L` (with `g = L/4`, `t = L%4`):
- A holds `A[g, 2t..2t+1]`, `A[g+8, 2t..2t+1]`, `A[g, 2t+8..2t+9]`, `A[g+8, 2t+8..2t+9]`.
- B holds `B[2t..2t+1, g]`, `B[2t+8..2t+9, g]`.
- C holds `C[g, 2t..2t+1]`, `C[g+8, 2t..2t+1]`.

The full per-lane packing code is in `mma_sync_example.cu`; mirror it for
your own kernels.

## ldmatrix: loading fragments from shared memory

The natural input to `mma.sync` is fragments that came from shared memory
(via `cp.async` from global). But the lane-element layout `mma.sync` expects
isn't the same as a contiguous shared-memory tile — naively writing a load
loop produces an enormous shuffle pattern.

`ldmatrix.sync.aligned.m8n8.x4.shared.b16` solves this in one instruction:
it cooperatively loads four 8×8 fp16 matrices from shared into the warp's
registers, with the lane layout `mma.sync.m16n8k16` expects on the A side.

The mechanics, from `ldmatrix_example.cu`:

```cpp
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    :  "r"(smem_int));
```

Each lane provides one source address (`smem_int`). The hardware uses lanes
0-7 to address rows 0-7 of matrix 0, lanes 8-15 for matrix 1, lanes 16-23
for matrix 2, lanes 24-31 for matrix 3. After the load, every lane holds
4 b32 (one per matrix) packed as 2 fp16 each, in the layout
`mma.sync.m16n8k16` reads.

PTX from `ldmatrix_example.ptx:104`:

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r27, %r28, %r29, %r30}, [%r31];
```

SASS from `ldmatrix_example.sass:253`:

```sass
/*07c0*/  LDSM.16.M88.4 R4, [R4] ;
```

`LDSM` = "Load Shared Matrix"; `.16` = 16-bit elements; `.M88` = 8×8 matrix;
`.4` = `.x4` (four matrices).

For an executable demo, run `./ldmatrix_example` — it loads 4 known 8×8
matrices and dumps lane 0 and lane 7's holdings, plus a range check.

## SASS-level cycle counting with clock64()

`clock_microbench.cu` brackets a tight FMA loop with `clock64()` reads to
measure cycles per instruction. It demonstrates the difference between
*throughput* (independent FMAs, no chain) and *latency* (one chain — each
FMA waits on the previous):

```
Throughput (4 independent chains):  ~7 cycles / FMA
Latency    (single chain):          ~27 cycles / FMA
```

These numbers won't be identical run-to-run (warp scheduling, partition
stalls), but the order-of-magnitude is right: independent FMAs run at the
issue rate; chained FMAs run at the latency.

Use this when you want to compare two implementations of the same hot loop
in cycles, not microseconds. The ratio is what matters.

## __threadfence and PTX memory ordering

> **Forward reference target for Module 11.**

CUDA C++ exposes three memory-fence intrinsics:

| Intrinsic | PTX form | Meaning |
|-----------|----------|---------|
| `__threadfence_block()`   | `membar.cta;`  | order memory ops within the CTA (block) |
| `__threadfence()`         | `membar.gl;`   | order memory ops within the GPU |
| `__threadfence_system()`  | `membar.sys;`  | order memory ops across CPU + GPU + peers |

For Module 11's persistent kernel, the host writes a doorbell and the device
reads it. The relevant question is: does the host's write become visible to
the device read? The answer requires `__threadfence_system()` (or the
equivalent `mbarrier` with system-scope) on the device side, and the right
mapping (`cudaHostAllocMapped` or unified memory) on the host side. See
Module 11 for the full pattern.

Atomic ops have their own ordering modifiers in PTX:

```ptx
atom.global.relaxed.gpu.add.u32  %r0, [%rd1], 1;   // no ordering
atom.global.acquire.gpu.add.u32  %r0, [%rd1], 1;   // acquire
atom.global.release.gpu.add.u32  %r0, [%rd1], 1;   // release
atom.global.acq_rel.gpu.add.u32  %r0, [%rd1], 1;   // both
```

These map to `cuda::std::atomic_ref` with the corresponding `memory_order`
in C++. For most code, `__threadfence()` + plain atomics is enough.

## bar.sync variants and named barriers

> **Forward reference target for Module 11.**

`__syncthreads()` is `bar.sync 0;` — barrier ID 0, all threads in the block.
PTX exposes more variants:

```ptx
bar.sync     0;             // = __syncthreads()
bar.sync     N;             // barrier ID N (0..15), all threads
bar.sync     N, COUNT;      // barrier ID N, only COUNT threads required
bar.red.{and,or,popc} N, COUNT, p;   // barrier-and-reduce
```

Why care:
- "Named barriers" (different IDs) let producer/consumer subgroups sync
  independently — handy when only half the block needs to wait on the
  shared-mem write.
- `bar.red.or` does a barrier + OR-reduce in one instruction. Useful for
  "did any lane fail?" early-exit logic without a separate scan.

`bar_sync_example.cu` demonstrates both. PTX form for `bar.red.or`:

```ptx
bar.red.or.pred  %p1, 4, %p0;    // barrier 4, count = block size, OR of %p0
```

## When to actually write PTX

Almost never. Don't write PTX as your default. Reach for it when:

1. **The compiler won't emit the instruction you need.** Cache modifiers,
   `cp.async.bulk` (Hopper), `wgmma`, `tensormap.replace.tile`, certain
   atomic variants — all PTX-only.
2. **You're matching a CUTLASS-style hand-tuned kernel.** Production GEMMs
   sometimes need exact register layouts that the compiler's register
   allocator won't produce on its own.
3. **You're chasing tail latency.** For HFT-style kernels where every
   instruction matters, looking at SASS to confirm there are no surprise
   spills or extra moves is worth doing. PTX rarely fixes these directly,
   but it's the level where you can see what's happening.

For everything else, write CUDA C++. The compiler is good. Reading PTX is
useful for understanding; writing it is rarely the right answer.

---

## What's in this directory

```
13-ptx-appendix/
  README.md              # this file
  Makefile               # builds binaries; `make ptx` and `make sass` dump
                         # the corresponding files for inspection

  vector_add.cu          # the simple kernel from § Reading PTX
  cache_hints.cu         # ld.global default vs .cg vs .nc, three-way
  cpasync_inline.cu      # __pipeline_memcpy_async wrapper vs raw cp.async PTX
  mma_sync_example.cu    # one-warp m16n8k16 mma.sync, verified vs host
  ldmatrix_example.cu    # ldmatrix.x4 from shared memory, verified
  clock_microbench.cu    # clock64() FMA throughput / latency
  bar_sync_example.cu    # named bar.sync and bar.red.or
```

### Suggested workflow

1. `make` to build the binaries.
2. `make ptx` — generates `*.ptx` files. Open them in your editor and walk
   through the PTX with the explanations in the Reading PTX section.
3. `make sass` — generates `*.sass` files. Compare against the PTX. Notice
   what changed (`ld.global.nc.f32` → `LDG.E.CONSTANT`, `mma.sync` →
   `HMMA.16816.F32`, `ldmatrix.x4` → `LDSM.16.M88.4`, etc.).
4. Run the binaries (`./mma_sync_example`, `./ldmatrix_example`,
   `./bar_sync_example`, `./clock_microbench`) to see the kernels execute
   and self-verify.
5. Edit `cache_hints.cu` to add another inline-PTX variant (e.g. `.cs`
   streaming) and observe the SASS.

## Cheat sheet: most-used inline PTX

```cpp
// Cached vs non-cached global loads
asm("ld.global.f32  %0, [%1];"     : "=f"(v) : "l"(p));   // default
asm("ld.global.cg.f32 %0, [%1];"   : "=f"(v) : "l"(p));   // skip L1
asm("ld.global.nc.f32 %0, [%1];"   : "=f"(v) : "l"(p));   // read-only cache path

// Vectorized load (16 bytes = float4)
asm("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(p));

// cp.async (Ampere+)
asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem));
asm("cp.async.commit_group;\n");
asm("cp.async.wait_group %0;\n"   :: "n"(N));

// mma.sync (HMMA on FP16, m16n8k16, FP32 accumulator)
asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
    :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
       "r"(b0),  "r"(b1));

// ldmatrix (Ampere+, used to feed mma.sync)
asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_int));

// Read SM clock for cycle-accurate microbench
unsigned long long t;
asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));

// Memory fences
asm("membar.cta;");      // = __threadfence_block()
asm("membar.gl;");       // = __threadfence()
asm("membar.sys;");      // = __threadfence_system()

// Named barrier
asm("bar.sync 1, %0;" :: "r"(count));
```

That's most of what you'll ever inline.
