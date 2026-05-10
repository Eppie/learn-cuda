# Module 3 — Shared memory & tiling

**Goal:** by the end of this module you should be able to (a) load a tile of data into
shared memory cooperatively, (b) explain when shared memory will create *bank
conflicts* and how to fix them, and (c) optimize matrix transpose from naive to within
spitting distance of `cudaMemcpy`.

Shared memory is your first real escape from the DRAM bandwidth ceiling — and it's the
foundation for every fast GEMM, attention, or stencil kernel you'll ever write.

---

## 1. What is shared memory?

Shared memory is a programmer-managed scratchpad on each SM. It's:

- **Fast.** Latency ~30 cycles vs. ~500 for DRAM.
- **Small.** The unified L1/shared pool is 128 KB / SM on `sm_89`. Up to **100 KB**
  of it can be reassigned to shared memory **per block** — but only via opt-in. The
  default per-block static cap is 48 KB; to push past that, the kernel must call
  `cudaFuncSetAttribute` at host setup:
  ```cpp
  cudaFuncSetAttribute(my_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      96 * 1024);   // request up to 96 KB
  ```
  And the per-launch shared-memory argument has to fit. If you skip this and try to
  declare an 80 KB static `__shared__` array, the kernel won't link.
- **Block-scoped.** Only threads in the same block can share a shared-memory buffer.
  Different blocks see independent allocations.
- **Banked.** It's split across 32 parallel banks, which is where bank conflicts come from.

Declare it inside a kernel with `__shared__`:

```cpp
__global__ void k() {
    __shared__ float tile[32][33];      // static-size shared array (≤ 48 KB by default)
    // ...
}
```

Or sized at launch (dynamic shared memory):

```cpp
extern __shared__ float buf[];          // size determined at launch
k<<<grid, block, /*bytes=*/ 32*33*sizeof(float)>>>(...);
```

The third `<<<...>>>` argument is the dynamic shared-memory size in bytes. You only
pay this once per block; threads in the block share the allocation. Dynamic shared
memory is what you need when the tile size depends on a runtime parameter.

## 2. The tiling pattern

The classic shared-memory pattern is:

```
1. Each thread cooperatively loads one element from global → shared
2. __syncthreads()                      // wait for everyone to finish loading
3. Each thread does work on the shared tile, possibly multiple loads
4. __syncthreads()                      // before reusing or rewriting the tile
5. Write results back to global memory
```

This converts a memory-heavy access pattern into one DRAM read + many shared accesses.
That's why shared memory is so important for kernels that *reuse* data (e.g., GEMM,
convolutions, stencils).

`__syncthreads()` is a barrier across the whole block. Forgetting it is the source of
~50 % of new CUDA bugs.

## 3. Banks and bank conflicts

Shared memory is divided into **32 banks**, each 4 bytes wide. Successive 4-byte words
go to successive banks:

```
addr (bytes):  0    4    8   12  ...  124  128  132 ...
bank:          0    1    2    3  ...   31    0    1 ...
```

When a warp issues a shared-memory load, the hardware checks which bank each lane
needs. If all 32 lanes touch **distinct banks** (or the same word, broadcast), the
access happens in one cycle. If two or more lanes hit the **same bank with different
words**, the accesses *serialize*: an N-way conflict means N cycles.

### Why `transpose_shared` conflicts: the lane → column mapping

The kernel's read-back is `tile[threadIdx.x][threadIdx.y]`. Inside a single warp:

- `threadIdx.x` runs 0..31 across the warp's 32 lanes (it's the fast-varying axis of
  a 2D `(32, 32)` block).
- `threadIdx.y` is **constant** within a warp — the warp is one row of the 32×32
  block, and that row has a single fixed `threadIdx.y`.

So lane `i` of the warp accesses `tile[i][const_y]`. With the storage layout
`tile[32][32]`, that address is `i*32 + const_y` — and `(i*32) mod 32 == 0`. Every
lane lands in **the same bank** (bank `const_y`). 32 lanes, 32 different words, same
bank → 32-way conflict, serialized over 32 cycles.

This is the precise mechanism. The fix doesn't change which words you load; it just
moves them to different banks.

### The +1 padding trick

Add one extra column to break the alignment:

```cpp
__shared__ float tile[32][33];          // <-- 33, not 32
```

Now `tile[i][j]` lives at offset `i*33 + j`. For a column access (varying `i`,
constant `j`), the offsets differ by 33 — coprime with 32. Each lane lands in a
different bank, conflict gone. The wasted byte per row is the cheapest performance
optimization in CUDA.

### When +1 isn't enough: 16-byte loads → swizzling

The +1 padding works because we're loading **4-byte** words, and the stride between
column elements (33 floats = 132 bytes, mod 32 banks of 4 B = 33 mod 32 = 1) puts
each lane in a unique bank. For **16-byte** vector loads (`float4`, or what tensor-core
fragments use), each load consumes 4 banks at once. The arithmetic that makes +1
work for 4-byte loads doesn't generalize — you have to permute *which* element each
lane reads. That's the **swizzling** trick used in production GEMM and FlashAttention
kernels: instead of changing the array's strides, you change the indexing function so
that warps hitting a column read from XOR-permuted addresses that land in distinct
4-bank groups. We'll cover this in **Module 7 §3 "Swizzled shared memory"**, where
it's required to feed `mma.sync` cleanly. For this module's `float` workload, +1 is
sufficient.

## 4. Why matrix transpose?

Transpose is the perfect teaching example because it's *pure data movement*:
arithmetic intensity is exactly zero. Optimizing it is entirely about access patterns.
Three steps:

| version          | reads        | writes       | bandwidth (rough) |
|------------------|--------------|--------------|-------------------|
| naive            | coalesced    | uncoalesced  | ~5–10 % of peak   |
| shared, no pad   | coalesced    | coalesced    | better, but bank-conflict-bound on the read-back |
| shared, padded   | coalesced    | coalesced    | ~70–85 % of peak — close to plain copy |

The naive version reads `a[y*W + x]` (coalesced over `x`) and writes
`b[x*W + y]`. The write is strided by `W` — every lane in a warp writes to a
different cache line. Same disease as Module 2.

The fix: every block reads a 32×32 tile of `a` coalesced, stores it transposed in a
shared tile, then writes the shared tile coalesced into `b`. Now both phases are
coalesced; the per-tile transposition happens cheaply in shared memory.

The padded version adds one column of padding to defang bank conflicts on the diagonal
write-back from shared memory.

## 5. Visualization

Open [`viz/bank-conflicts.html`](../viz/bank-conflicts.html) (added by the viz track).
The page shows a 32×32 vs 32×33 vs swizzled tile with each cell colored by which of the
32 banks it lives in; a warp-access animation shows the conflict (or absence thereof).

---

## Exercises

> Open `starter.cu` and complete the TODOs.

**1. Naive transpose.** Write the obvious row-major-to-column-major copy. Confirm
correctness for a small matrix. Bench it; note the gap to `copy_scalar` from Module 2.

**2. Shared-memory transpose (no padding).** Use a 32×32 `__shared__` tile. Load
coalesced, transpose in shared, write coalesced. Don't forget `__syncthreads()`.

**3. Padded shared-memory transpose.** Pad the tile to `[32][33]`. The kernel logic
doesn't change — only the array dimension does. Bench again.

**4. 32×8 block, 4 rows per thread.** This is the canonical, production-grade shape
for transpose, not a stretch problem. The block is `(32, 8)` = 256 threads, and each
thread handles 4 input rows: `for (int j = 0; j < 32; j += 8) tile[ty + j][tx] = ...`.
Why is this faster than the 1024-thread version? Two reasons. (a) **Occupancy**: the
shared-memory budget per block is the same, but each block now occupies 4× fewer
threads, so the SM can resident 4× more blocks → better latency hiding. (b)
**Per-thread instruction-level parallelism**: the 4 loads/stores per thread have no
dependency between iterations, so the compiler can overlap them. Implement it; the
bench reports the improvement.

**5. Dynamic shared-memory variant.** Re-implement (3) using `extern __shared__ float
tile[]` instead of static. The padded layout becomes `tile[ty * (TILE+1) + tx]`. Pass
`(TILE * (TILE+1) * sizeof(float))` as the third `<<<...>>>` launch argument. Same
correctness, same speed; the point is to know the API.

**6. `float4` shared-memory load.** Add a kernel that loads each 32×32 tile from
global memory with **vectorized** loads (one `float4` per thread, so 32×8 threads
cover the tile). Store into shared memory normally (`float`-strided), then transpose
+ store as before. Verify that the kernel still passes; what does `ncu` report for
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request`? Note: this exercise
does *not* use `float4` to *read from shared memory*; that needs swizzling (Module 7
§3) to avoid 4-bank conflicts.

### Stretch

**7.** Use `ncu` to count shared-memory bank conflicts on the unpadded version. The
metric is `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`.

> *Hint:* On `transpose_shared`, expect this counter to be substantial (the read-back
> column access is 32-way conflicting). On `transpose_shared_padded`, it should be ~0.
> The throughput gap between the two is *exactly* the cost of those serializations.

---

## Profiler checklist

```bash
make
ncu --set full ./solution
```

Look at:

- **`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`** — should drop to ~0 after
  padding.
- **DRAM throughput** — naive will be far below peak; padded should approach the
  Module 2 `copy_scalar` ceiling.
- **Shared Memory chart** — visualizes per-bank traffic; conflicts show as red.

For the central counter glossary, see [Module 4 — Profiler counters introduced in
this module](../04-profiling/README.md#profiler-counters-introduced-in-this-module).

## Key takeaways

- Shared memory is a programmer-managed cache. Use it whenever you want to read data
  more than once.
- The tiling pattern (cooperative load → `__syncthreads` → compute → write) underpins
  every fast CUDA kernel from this point on.
- Bank conflicts are real and easy to introduce; a `+1` of padding is often the fix.
- For 16-byte loads, +1 doesn't help — see **Module 7 §3 "Swizzled shared memory"**.
- 100 KB shared-per-SM is opt-in via `cudaFuncSetAttribute`. Default static cap is 48 KB.
- Transpose is pure access-pattern work; the bench numbers tell the whole story.
