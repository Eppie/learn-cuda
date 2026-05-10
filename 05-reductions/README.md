# Module 5 — Reductions, scans, and warp shuffles

**Goal:** by the end of this module you should be able to (a) write a fast sum
reduction in three escalating forms — shared memory, warp shuffles, grid-stride loop —
(b) write a parallel scan (prefix sum) in both Hillis-Steele and Blelloch forms,
and (c) compose those primitives into the building blocks downstream modules need:
segmented (per-row) reductions and the running-max/running-sum recurrence behind
**online softmax**. Reductions and scans show up *everywhere* — GEMM dot products,
softmax denominators, layernorm statistics, FlashAttention's K-tile rescaling — so
the patterns here are reused for the rest of the course.

---

## 1. The shape of a reduction

A reduction collapses N elements into 1 using an associative operator (`+`, `max`,
`min`, `xor`, ...). The work is "embarrassingly serial" on a CPU, but on a GPU we
*want* parallelism, so we shape it as a tree:

```
  a0  a1  a2  a3  a4  a5  a6  a7
   \  /    \  /    \  /    \  /
   a01     a23     a45     a67
     \      /        \      /
      a0..3            a4..7
        \                /
            a0..7
```

`log2(N)` levels deep. The interesting question is *which thread does which add at
which level*, which determines memory traffic and synchronization.

We'll build three versions and watch each one approach DRAM peak.

## 2. The hierarchy: warp → block → grid

Modern fast reductions use *all three* levels of the GPU's hierarchy:

1. **Warp-level**: 32 threads inside a warp can exchange data via `__shfl_*` without
   touching shared memory at all. One register-to-register hop.
2. **Block-level**: warps within a block coordinate via shared memory. Usually one
   `__syncthreads()` is enough.
3. **Grid-level**: blocks running on different SMs can either (a) write per-block
   partial sums and have the host (or a second kernel) finish, or (b) atomically add
   into a single output, or (c) use cooperative groups for a true grid-wide barrier.

The template "warp reduce → block reduce → grid reduce via partial sums" handles 95 %
of cases.

## 3. Warp shuffles: `__shfl_*_sync`

Inside a warp, threads share a register file partition, and the hardware exposes
fast lane-to-lane communication primitives:

- `__shfl_sync(mask, v, src_lane)` — read `v` from lane `src_lane`
- `__shfl_xor_sync(mask, v, laneMask)` — butterfly exchange (lane `i` swaps with `i ^ laneMask`)
- `__shfl_down_sync(mask, v, delta)` — lane `i` reads `v` from lane `i + delta`
- `__shfl_up_sync(mask, v, delta)` — lane `i` reads `v` from lane `i - delta`

The `mask` argument is a 32-bit bitmap of which lanes participate. For full-warp
operations use `0xffffffff`. (Don't forget the `_sync` versions — the older mask-less
shuffles are deprecated.)

A warp-wide sum reduction is **five** shuffle-add steps (offsets 16, 8, 4, 2, 1) —
five `__shfl_down_sync` instructions plus five adds, ten ops total in the SASS,
but the conceptual depth is 5:

```cpp
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;   // lane 0 holds the final sum
}
```

No shared memory, no `__syncthreads()`. This is a huge step up from the textbook
shared-memory tree.

### `__reduce_*_sync` (sm_80+)

Ampere and later expose hardware-accelerated warp reductions as a single intrinsic:

```cpp
unsigned s = __reduce_add_sync(0xffffffff, x);   // u32 only
unsigned m = __reduce_max_sync(0xffffffff, x);   // u32 only
```

These compile to a single `REDUX.SUM` (or `REDUX.MAX`, etc.) SASS instruction —
faster than the five-shuffle dance, but the operand types are limited (no FP
overload). Useful when you're reducing integer histogram bins or argmax indices.

## 4. The three versions (sum reduction)

### v0 — Classic shared-memory tree

Each block loads `BLK` elements into shared memory, then reduces them with sequential
addressing (`smem[tid] += smem[tid + s]`, halving `s`). One `__syncthreads()` per
level. Conceptually clean, but uses shared memory and barriers all the way down.

### v1 — Warp shuffles, one element per thread

Skip shared memory for the in-warp part of the reduction. Each warp reduces its 32
elements via shuffles into a single value (in lane 0). A small shared array holds
*one float per warp*, so the first warp can do one more shuffle pass to combine them.
Total: one `__syncthreads()`.

### v2 — Grid-stride loop + warp shuffles

The previous versions launch one thread per input element. For huge inputs this means
millions of blocks, each doing very little work. Better: launch *fewer* blocks (just
enough to fill the GPU) and have each thread sum many input elements with a
**grid-stride loop**:

```cpp
float v = 0.0f;
for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
    v += in[i];
}
// then block-reduce v
```

This shifts work from "spawn millions of tiny blocks" to "fewer blocks, each doing
real work" — better launch overhead, better instruction-level parallelism, less
contention on the partial-sum output array. v2 is what production reductions actually
look like.

## 5. Scans (prefix sums)

A reduction collapses N → 1; a **scan** computes N → N. The output's `i`-th element
is the running fold of the first `i+1` inputs (inclusive) or the first `i` inputs
(exclusive). Scans are the basic primitive behind sorting, stream compaction, sparse
indexing, and (in this course) the cumulative softmax denominator inside an online
softmax pass.

We'll write two warp-level scan kernels and then sketch the block- and grid-level
extensions.

### 5.1 Hillis-Steele (work-inefficient, depth = log N)

Inside a single warp, an inclusive scan with shuffles is just:

```cpp
__device__ __forceinline__ float warp_scan_inclusive(float v) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float t = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) v += t;
    }
    return v;
}
```

This is the Hillis-Steele scan: each step doubles the offset and adds the upstream
neighbor. 5 steps for a 32-lane warp. Total work is `N log N` — every level redoes
work — so it's *work-inefficient* but has minimum depth.

### 5.2 Blelloch (work-efficient, two-phase)

For very large arrays, work-inefficiency hurts. Blelloch's algorithm does two passes
over a balanced binary tree:

1. **Up-sweep (reduce):** build partial sums going up the tree.
   `a[i] += a[i - stride]` at each level for the right children.
2. **Down-sweep:** start with the root set to identity; at each level, the right child
   gets `(left + parent_old)`, the left child gets `parent_old`.

Total work is `2N` — work-optimal — and depth is `2 log N`. The constant factor is
worse than Hillis-Steele for short inputs, so production scans (CUB's `BlockScan`)
typically use Hillis-Steele inside a warp and a Blelloch-style hierarchy across warps
or blocks.

> See `scan_starter.cu` for the warp-level Hillis-Steele scan, the block-level Blelloch
> scan over `BLK = 256` elements, and a bench that compares them.

### 5.3 Parametric scans: scan isn't just `+`

Both algorithms above generalize: anywhere the combine `+` appears, you can
substitute *any associative operator*. With max + sum-of-exponentials you
get online softmax (§7). With matrix multiply you get a parallel reduction
of a sequence of matrix products. With a 2-tuple `(a, b)` and combine
`(a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)` you get a parallel solver for
linear recurrences `x[t] = a[t]·x[t-1] + b[t]` — the operation at the heart
of **state-space models like Mamba**.

This is the modern face of "scans are useful." Mamba's selective scan
(diagonal-A simplification: every `a` and `b` becomes a vector instead of a
scalar; combine is element-wise) is the most prominent recent example, and
it's the inner kernel of several open SSM model implementations. See
M12 Capstone Project E for a project pulling this in.

> **Stretch (deferred):** swap your Blelloch scan's combine from `+` to the
> parametric Mamba combine. Discretize Δ via `Ā = exp(Δ·A)`, `B̄ ≈ Δ·B`.
> Implement at single-warp first (32 elements via shuffle), then block-level
> using the Blelloch scaffold above. Estimated 4–6 hours after Blelloch is
> solid. The reference: `state-spaces/mamba` repo, `selective_scan_cuda`.

## 6. Segmented / strided reductions

GEMM-shaped problems often want **per-row** reductions: for an `M × N` row-major
matrix `X`, compute `out[i] = reduce(X[i, :])` for all `i`. This is the building
block under softmax (per-row), layernorm (per-row mean and variance), and any
attention denominator.

The natural mapping is **one block per row, BLK threads sweep the columns**:

```cpp
__global__ void row_sum(const float* X, float* out, int rows, int cols) {
    int row = blockIdx.x;
    float v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) v += X[row * cols + c];
    v = block_reduce_sum(v);
    if (threadIdx.x == 0) out[row] = v;
}
```

Each block handles one row; threads inside the block do a grid-stride sweep over the
columns. This kernel is reused (almost verbatim) in M09's layernorm and M10's softmax.

## 7. Online softmax: the running (max, sum) recurrence

Naïve softmax requires two passes over the input (one for the max, one for the
denominator). FlashAttention can't afford two passes over its K-tiles, so it uses
the **online softmax** trick: maintain a running pair `(m, s)` and update it as each
new element arrives.

For one element `x`:

```
m_new = max(m, x)
s_new = s * exp(m - m_new) + exp(x - m_new)
```

When `x` is a new maximum, the existing partial sum `s` gets *rescaled* by
`exp(m - m_new)` (a number ≤ 1) so the running denominator stays consistent with
the new reference max.

Combining two streams `(m1, s1)` and `(m2, s2)` is the same recurrence:

```
m = max(m1, m2)
s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

This is the **building block** M10 leans on. We define it here as a reusable
device function:

```cpp
struct ms_pair { float m; float s; };
__device__ __forceinline__ ms_pair online_softmax_combine(ms_pair a, ms_pair b);
__device__ __forceinline__ ms_pair online_softmax_warp(float x);   // warp-level reduction
__device__ __forceinline__ ms_pair online_softmax_block(float x, ms_pair* smem);
```

Implementations live in `solution.cu`. M09 (layernorm/softmax) and M10 (FlashAttention)
will `#include` this header pattern (or copy the inlined version).

## 8. Grid-level reductions

For a single output (true `N → 1`), choices are:

- **Two-kernel pattern** (most common): kernel 1 writes per-block partials; kernel 2
  reduces those.
- **Atomic finish**: each block does an `atomicAdd` to a single output. Simpler code,
  but contention scales with `gridDim.x`. Fine for `gridDim.x ≤ ~256`.
- **Cooperative groups grid sync**: a single kernel can do the whole reduction,
  using `cooperative_groups::this_grid().sync()` between the per-block partial-sum
  phase and the final-reduction phase. Requires `cudaLaunchCooperativeKernel` and a
  grid that fits on the device simultaneously (RTX 4090: 128 SMs × 16 blocks max ≈
  2048 cooperative blocks).

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void reduce_grid(const float* in, float* out, int n) {
    cg::grid_group grid = cg::this_grid();
    // Phase 1: each block reduces its slice into partial[blockIdx.x].
    float v = block_grid_stride_reduce(in, n);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;

    grid.sync();   // <-- the magic: wait for *all blocks* across the grid.

    // Phase 2: block 0 reduces the partials.
    if (blockIdx.x == 0) {
        float v2 = (threadIdx.x < gridDim.x) ? partial[threadIdx.x] : 0.0f;
        v2 = block_reduce_sum(v2);
        if (threadIdx.x == 0) *out = v2;
    }
}
```

Launch with `cudaLaunchCooperativeKernel`, not the triple-chevron. See the stretch
exercise.

## 9. Correctness gotchas

- **Initialize out-of-range values to the operator's identity.** For sum it's 0; for
  max it's `-INFINITY`; for product it's 1. Otherwise threads near the end of the
  array contribute garbage.
- **`__syncthreads` requires *all* threads in the block to reach it.** Putting it
  inside a divergent `if (tid < s)` is undefined behavior. Hoist it.
- **Float reductions are non-deterministic in order** — different launch / block
  configurations sum in different orders, giving slightly different last bits. Don't
  bit-compare two reductions; use a tolerance.
- **Online softmax stability.** Keep the running max as a *reference*; never compute
  `exp(x_i)` directly. Always `exp(x_i - m)` so the inputs to `exp` are ≤ 0 and never
  overflow.

---

## Exercises

> Open `starter.cu` and `scan_starter.cu` and complete the TODOs.

**1. v0: shared memory tree.** Implement the classic per-block reduction with
sequential addressing.

**2. Warp helpers.** Implement `warp_reduce_sum` (five shuffles) and
`block_reduce_sum` (warp reduce → write per-warp result to shared mem → first warp
reduces those).

**3. v1.** Use your `block_reduce_sum`. Each thread loads one element; block reduces;
lane 0 of warp 0 writes `out[blockIdx.x]`.

**4. v2.** Same as v1 but with the grid-stride loop on the front. Run with
`gridDim.x = 256` (a few blocks per SM); each thread will sum `n / (256 * BLK)`
elements before the block reduction.

**5. `__reduce_add_sync` rewrite.** Implement `reduce_v0_intrinsic` using
`__reduce_add_sync` for the warp reduction (cast floats through `__float_as_uint`
won't work — switch the test to `int` input). Compare SASS between this and v1.

**6. Per-row sum.** Implement `row_sum` over an `M × N` row-major matrix, one block
per row. Verify against a naive CPU loop.

**7. Hillis-Steele warp scan.** Open `scan_starter.cu` and complete `warp_scan_inclusive`.

**8. Blelloch block scan.** Implement `block_scan_blelloch` over 256 elements (up-sweep
+ down-sweep). Use shared memory; one `__syncthreads` per level.

**9. Online softmax warp helper.** Implement `online_softmax_warp(float x)` returning
the (max, sum) pair after a single shuffle-based warp reduction using the recurrence.

### Stretch

**10.** Fuse the per-block partial sums into a single output via `atomicAdd`. How does
contention show up in the profiler?

**11.** Implement `max` reduction. Same skeleton, different operator. Note the change
in the identity element.

**12.** Grid-level reduction using `cooperative_groups::this_grid().sync()`.
Launch with `cudaLaunchCooperativeKernel`. Compare to the two-kernel pattern.

**13.** Block-level Hillis-Steele scan over `BLK = 256` elements (combine warp scans
across warps). Compare against your Blelloch from exercise 8.

---

## Profiler checklist

```bash
make
ncu --set full ./bench
ncu --set full ./scan_bench
```

Look at:

- **DRAM throughput** — should climb with each version. v2 should hit ~85–95 % of
  peak (it's a bandwidth-bound problem).
- **Achieved occupancy** — v0 will be limited by shared-memory usage; v2 by registers.
- **Stall reasons** — v0 will show heavy "Wait" (barriers); v1 cuts that; v2 cuts it
  further.
- **Issued vs executed instructions** — fewer is better; each shuffle replaces a
  shared-mem load + barrier sequence.
- **`smsp__inst_executed_pipe_redux.sum`** — non-zero confirms `__reduce_*_sync` hit
  the `REDUX` pipe (counter introduced in M04).

## Key takeaways

- Reductions are everywhere — and most are bandwidth-bound, not compute-bound.
- Warp shuffles (`__shfl_down_sync`) replace shared memory + `__syncthreads()` for
  the in-warp part of a reduction. Always use them.
- The hierarchy is warp → block → grid. Master the block-level helper and you can
  drop it into any future kernel.
- A grid-stride loop turns "millions of small blocks" into "thousands of useful
  blocks". This is the single biggest win going from a textbook reduction to a
  production one.
- Scans are reductions that *also* keep the per-position partial result. Hillis-Steele
  is short and simple inside a warp; Blelloch is work-optimal across many elements.
- Online softmax's `(m, s)` recurrence is the cornerstone of FlashAttention. We
  define it here so M09 and M10 can use it without re-deriving.
