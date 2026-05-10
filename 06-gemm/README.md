# Module 6 — The GEMM journey

**Goal:** take a 4096³ FP32 matrix multiply from a one-line naive kernel to within
~80 % of cuBLAS, in eight steps. Every previous module pays off here.

This module is the centerpiece of the course. It's also the longest. Don't try to do
all eight versions in one session — work through them one at a time, profile each,
and read the bench output before moving on. The compounding insight is the point.

The structure follows Simon Boehm's
[CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM), adapted for `sm_89`
and FP32-only (Tensor Cores arrive in Module 7, where the v6 launcher is *re-used*
with the inner FMA loop replaced by WMMA).

---

## 1. Why GEMM?

GEMM (`C = A·B`) is the canonical compute-heavy kernel:

- **Arithmetic intensity scales with tile size.** A 4096³ GEMM does 137 GFLOPs of
  math against ~192 MB of unique data — over 700 FLOPs per byte if you reuse data
  perfectly. That puts it deep in the compute-bound regime, unlike everything else
  we've written so far.
- **Every fast kernel uses the same building blocks.** Convolutions, attention,
  layernorm, softmax — they all reduce to GEMMs or GEMM-shaped tile patterns.
- **It's measurable against a strong baseline.** cuBLAS is *very* good and gives us a
  ceiling to chase.

If you only deeply understand one CUDA kernel, make it this one.

### Block-index convention

Every kernel in this module uses the same mapping:

```
blockIdx.x  ->  column tile of C   (cCol)
blockIdx.y  ->  row    tile of C   (cRow)
```

That matches CUDA's natural intuition for `gridDim.x` (the "horizontal" dimension)
and is the convention every kernel from 6.0 through 6.6 follows. (Earlier versions
of this module mixed conventions — `v0/v1/v2` had `blockIdx.x` as the row tile while
`v3..v6` had it as the column tile. Now uniform.)

## 2. Setup and the reference

| symbol | role | shape       | layout      |
|--------|------|-------------|-------------|
| `A`    | input| `M × K`     | row-major   |
| `B`    | input| `K × N`     | row-major   |
| `C`    | output| `M × N`    | row-major   |

We use `M = N = K = 4096` throughout. That's 64 MB per matrix, 192 MB total — out of
L2 (72 MB), so reuse can't happen for free.

Total FLOPs: `2 · M · N · K = 137.4 G`.
Total unique bytes (no reuse): `(M·K + K·N + M·N) · 4 = 192 MB`.

There are two arithmetic-intensity numbers worth distinguishing:

- **Peak possible AI** (perfect, full reuse of every byte): `137 G / 192 M ≈ 716 FLOP/byte`.
  This is what you'd see if every input byte were read exactly once.
- **Per-tile achieved AI** is much lower because tiles are loaded and reloaded as we
  scan along K. For block tile `BM × BN` with K-step `BK`:
  `AI_tile = (2·BM·BN·BK) / ((BM·BK + BK·BN) · 4) = (BM·BN) / (2·(BM+BN))`.
  For `BM=BN=128`: `AI_tile = 128*128 / (2*256) = 32 FLOP/byte` per K-iteration —
  high enough to be compute-bound, but two orders of magnitude below the peak above.

For reference, the **RTX 4090 ceilings** are:

- FP32 compute: ~83 TFLOPs peak (achievable: ~50 TFLOPs in practice with cuBLAS)
- DRAM bandwidth: ~1008 GB/s

A back-of-envelope: at 50 TFLOPs, our GEMM takes `137 G / 50 T ≈ 2.7 ms`. cuBLAS
typically lands in the 3–5 ms range on this size. Anything substantially slower is
leaving compute on the table.

## 3. The eight steps

Each step keeps the previous fixes and adds one new optimization. Keep that mental
model — the journey is monotone.

### 6.0 Naive (Simon's bad mapping)

```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * 32 + threadIdx.x;   // varies within warp -> bad
    int col = blockIdx.x * 32 + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row*K + k] * B[k*N + col];
        C[row*N + col] = acc;
    }
}
```

Each thread computes one output element. `threadIdx.x` varies fastest within a warp,
so `row` varies 0..31. That means within a warp, threads read 32 different rows of
`A` and write 32 different rows of `C` — both *uncoalesced*. This is exactly the
Module 2 mistake at industrial scale.

Expected: a few hundred GFLOPs, well under 1 % of cuBLAS.

### 6.1 Global memory coalescing

The fix is one line: lay out threads as a 1D array of size 1024 (= 32×32) and
*derive* row/col such that consecutive lanes hit consecutive columns, not rows.

```cpp
int row = blockIdx.y * 32 + threadIdx.x / 32;   // slow within warp
int col = blockIdx.x * 32 + threadIdx.x % 32;   // fast within warp
```

Now within a warp, `row` is constant and `col` runs 0..31. Reads of `B[k*N + col]`
and writes of `C[row*N + col]` are coalesced. Reads of `A[row*K + k]` are now
broadcast (all 32 lanes read the same element each iteration) — also efficient.

Expected: ~10× speedup. We're still memory-bound, but now we're using the bandwidth
we ask for.

### 6.2 Shared memory cache-blocking

Each output `C[i,j]` reads a full row of `A` and a full column of `B`. With `K = 4096`,
that's 32 KB per output element from global memory. Wasteful: many output elements
share the same input rows / columns.

Tile the computation. Each block of 32×32 threads computes a 32×32 tile of `C` by
walking along `K` in chunks of `BK = 32`:

```
For each kBlock in [0, K, BK):
    Cooperatively load A[bRow:bRow+32, kBlock:kBlock+BK]  → As (32 × 32 shared)
    Cooperatively load B[kBlock:kBlock+BK, bCol:bCol+32]  → Bs (32 × 32 shared)
    __syncthreads()
    Each thread t accumulates 32 partial-products from As, Bs into a register.
    __syncthreads()
```

Loads are amortized 32× — each global load feeds 32 threads. Expected: ~2× from 6.1,
~17 % of cuBLAS.

### 6.3 1D blocktiling

Each thread still computes only *one* output element — wasting registers and ILP.
Switch each thread to compute `TM = 8` output elements *stacked vertically* in a
column. That means one block of `(BM/TM) × BN = 8 × 64 = 512` threads computes a
`64 × 64` tile of C, with each thread holding 8 accumulators.

```
threadResults[TM] in registers
for kBlock:
    Load A tile (BM × BK = 64 × 8) and B tile (BK × BN = 8 × 64) into shared
    __syncthreads()
    for k in [0, BK):
        Btmp = Bs[k * BN + threadCol]                  // 1 shared read
        for t in [0, TM):
            threadResults[t] += As[(threadRow*TM + t) * BK + k] * Btmp
    __syncthreads()
```

Each thread now does 8 multiply-adds per shared-`Bs` load, dramatically reducing
shared-memory traffic. Expected: ~2× from 6.2, ~40 % of cuBLAS.

### 6.4 2D blocktiling

Same trick, both dimensions. Each thread computes a `TM × TN = 8 × 8` square of C
(64 elements per thread, in registers). Block size: `(BM/TM) × (BN/TN) = 16 × 16 =
256` threads, computing a `128 × 128` tile.

The inner product becomes a 2-level register reuse:

```
regM[TM], regN[TN]                    // small register caches
threadResults[TM][TN]                 // 64 accumulators per thread
for k in [0, BK):
    for i in [0, TM): regM[i] = As[(threadRow*TM + i)*BK + k]
    for i in [0, TN): regN[i] = Bs[k*BN + threadCol*TN + i]
    for i in [0, TM):
        for j in [0, TN):
            threadResults[i][j] += regM[i] * regN[j]
```

Each shared-memory load now feeds `TM × TN = 64` multiply-adds. We're approaching
the regime where shared-memory bandwidth, not DRAM bandwidth, is the limit.
Expected: ~70 % of cuBLAS.

### 6.5a Vectorized loads

Two ideas live in v5; we split them so each can be measured independently. Step a:
**`float4` for global loads.** Each thread issues 128-bit loads (`LDG.E.128` instead
of `LDG.E`) when copying tiles to shared memory. Fewer instructions, fewer
outstanding-request slots wasted.

`As` is still stored in row-major (`As[BM][BK]`) so the inner-loop access
`As[(threadRow*TM + i)*BK + dotIdx]` is *strided across `i` by `BK`*. The compiler
can't vectorize that load. We're missing half the win.

### 6.5b Vectorized loads + transposed As

Step b: store `As` transposed (`As[BK][BM]`). Now the inner-loop access becomes

```
regM[i] = As[dotIdx * BM + threadRow * TM + i]
```

— contiguous in `i`. The compiler emits `LDS.128` for this load. The transposition
happens at the moment of write into shared memory: each `float4` global load is
"scattered" into four columns of the transposed `As`. That looks ugly but is on the
fast path — there's plenty of L1 / shared bandwidth to spare.

Expected (combined a + b): ~75–80 % of cuBLAS.

### 6.6 Warp tiling — full sub-tiling

So far the hierarchy has been block tile → thread tile. cuBLAS adds a *warp tile* in
between, and one more level: each warp's 32 threads claim a contiguous **WM × WN**
warp tile, then iterate over multiple **WSUBM × WSUBN** *sub-tiles* inside their
warp tile, with one thread owning one **TM × TN** thread tile per sub-tile.

```
block tile (BM x BN)
  warp tile (WM x WN)
    sub-tile (WSUBM x WSUBN)        // iterated WMITER * WNITER times per warp
      thread tile (TM x TN)         // one per lane per sub-tile
```

The configuration this module ships with (and what's in `kernels.cuh`):

| param   | value | meaning |
|---------|-------|---------|
| `BM`    | 128   | block tile rows |
| `BN`    | 128   | block tile cols |
| `BK`    | 16    | K-step |
| `WM`    | 64    | warp tile rows  → `BM/WM = 2` warp rows per block |
| `WN`    | 64    | warp tile cols  → `BN/WN = 2` warp cols per block (4 warps total) |
| `WMITER`| 2     | sub-tiles per warp in M direction |
| `WNITER`| 2     | sub-tiles per warp in N direction (4 sub-tiles total per warp) |
| `WSUBM` | 32    | `= WM / WMITER` |
| `WSUBN` | 32    | `= WN / WNITER` |
| `TM`    | 4     | thread tile rows |
| `TN`    | 8     | thread tile cols |

The lane layout inside one sub-tile is `(WSUBM/TM) × (WSUBN/TN) = 8 × 4 = 32` lanes
— exactly one warp. Each thread accumulates `TM*TN * WMITER*WNITER = 4*8 * 2*2 =
128` floats in registers across all 4 sub-tiles. Block has 128 threads (4 warps).

The structural difference from v5:

- v5: each thread owns *one* TM × TN thread tile and reads `As/Bs` slices once per
  K-iter.
- v6: each warp loads one slice per K-iter and *iterates over WMITER × WNITER
  sub-tiles*, reusing `regM/regN` caches across sub-tiles. The compiler now has
  4 independent sub-tiles' worth of FMAs to schedule per K-iter — substantially
  more ILP, and more chances to hide each shared-memory load.

This is the structural step that makes Modules 7 (Tensor Cores) and 8 (async copy)
clean — both replace the inner FMA loop with a different primitive while keeping
the same warp-tile geometry.

Expected: 80–85 % of cuBLAS on `sm_89` without Tensor Cores.

### Beyond — what cuBLAS still does that we don't

- **Tensor Cores** (Module 7) — for FP16/BF16 GEMM, an order-of-magnitude jump.
  Module 7 *re-uses this v6 launcher* and replaces the inner FMA loop with
  `wmma::mma_sync` fragments.
- **Async copy with double buffering** (Module 8) — overlap the next tile's load
  with the current tile's compute via `cp.async`.
- **Autotuning** — picking block / warp / thread tile sizes per (M, N, K) shape.
- **PTX/SASS-level scheduling** — register allocation tuned for specific
  microarchitecture quirks.

Mention this to set expectations: "near cuBLAS without Tensor Cores or async copy"
is the realistic ceiling for this module.

---

## 4. Code layout

This module uses a slightly different file layout than the others because there are
eight kernels:

```
06-gemm/
  README.md         # this file
  Makefile          # builds solution / starter / bench (links cuBLAS)
  gemm.h            # M/N/K constants, host helpers, verification
  kernels.cuh       # all eight kernel implementations + thin launchers
  solution.cu       # uses kernels.cuh; runs all eight, verifies vs cuBLAS
  starter.cu        # has 8 kernel TODO stubs; same verification main
  bench.cu          # uses kernels.cuh + cuBLAS; verify pass + perf table
```

`starter.cu` has eight `// TODO` blocks in the order above. You can implement them
left-to-right; each one only relies on what came before plus one new idea.

`bench.cu` runs a verify pass on every kernel before reporting any TFLOPs numbers —
silently-wrong kernels can post 50 TFLOPs, so we catch them first.

`solution.cu` and `starter.cu` accept `--small` / `-s` to run at `M=N=K=512` (fast
sanity check).

---

## Exercises

> Open `starter.cu` and complete the TODOs.

1. **6.0 naive.** Get the indexing and the inner loop right.
2. **6.1 coalesced.** Re-map threads so warps walk columns of C, not rows.
3. **6.2 shared.** Tile size `BM = BN = BK = 32`, one element per thread.
4. **6.3 1D tiling.** `BM = 64, BN = 64, BK = 8, TM = 8`.
5. **6.4 2D tiling.** `BM = BN = 128, BK = 8, TM = TN = 8`.
6. **6.5a vectorized.** Same as 6.4 + `float4` global loads. As still row-major.
7. **6.5b transposed As.** Same as 6.5a + transpose As during the shared-mem store.
8. **6.6 warp tiling.** Full sub-tiling: WMITER × WNITER sub-tiles per warp, TM × TN
   per thread per sub-tile. Use the parameter table above.

Each version should produce identical results (within FP tolerance) on the
verification step.

### Stretch

- **Autotune** `BM/BN/BK/WM/WN/WMITER/WNITER/TM/TN` for the 4096³ shape. There is no
  single best answer; shapes matter.
- **Try non-square shapes** (e.g., `M=4096, N=4096, K=1024`). Which version wins
  changes — wider tiles help when `K` is short.
- **Mixed-precision accumulator** forward-ref: M07 uses FP16 inputs with FP32
  accumulators. The choice has both perf and numerical-accuracy consequences.
- **Split-K** forward-ref: when M and N are small but K is huge, splitting along K
  across multiple blocks (and reducing in a second kernel) keeps the SMs busy.
- **Warp-spec / specialized warps** forward-ref: in M08 / Hopper, some warps are
  dedicated to loading via `cp.async` while others compute — one of the standard
  "next steps" beyond what we cover here.

---

## Profiler checklist

```bash
make
ncu --set full ./bench
```

Track these as you go:

- **DRAM throughput** → falls steadily as tiling improves (you're moving fewer
  bytes); from ~95 % at 6.0 down to ~20 % at 6.6.
- **Compute throughput (SM)** → rises steadily; should hit ~50 % of peak by 6.5b.
- **Achieved occupancy** → drops as register usage grows (each thread holds a
  bigger thread-tile of accumulators). That's fine *if* compute throughput rises
  to compensate. v6 with 128 threads/block × 128 accumulators is intentionally
  low-occupancy / high-ILP.
- **Shared bank conflicts** → 0 except possibly in 6.5b/6.6 if your transpose
  layout causes them.
- **Stall reasons** → "Long Scoreboard" (DRAM) dominates early versions; "MIO
  Throttle" (shared-memory pressure) appears as you tile harder; "Selected"
  (actually running) should grow.

The roofline plot in `ncu` is especially satisfying for this module: you'll watch
the dot move from "stuck on the memory roof" up to "approaching the compute roof".

## Key takeaways

- A naive matmul is two orders of magnitude away from cuBLAS. Most of the gap is
  closed by re-using data through the memory hierarchy.
- Tiling is hierarchical: block tile → warp tile → sub-tile → thread tile. Each
  level amortizes loads from the level above.
- Once you're compute-bound, optimization shifts from *bytes* to *instructions* —
  vectorized loads, register reuse, sub-tile iteration for ILP, occupancy/register
  tradeoffs.
- v6 (full Boehm sub-tiling) is the structural anchor for the next two modules:
  M07 swaps the inner FMA loop for Tensor-Core fragments, M08 swaps the loads for
  `cp.async` + double buffering.
