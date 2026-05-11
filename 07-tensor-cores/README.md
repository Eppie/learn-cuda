# Module 7 — Tensor Cores

**Goal:** by the end of this module you should be able to (a) explain what a Tensor
Core does in one sentence, (b) write a working FP16 → FP32 GEMM using the WMMA C++
API by evolving Module 6's v6 (warp-tiled) kernel, (c) write the same GEMM using
raw `mma.sync` PTX with explicit fragment layouts, and (d) reason about when
Tensor Cores actually help vs. when they don't.

This module evolves directly from Module 6 v6. The block-level structure (BM=BN=128,
BK=16, 4 warps/block, each warp owning a 64×64 tile) is identical; only v6's
inner FMA loop changes — replaced with WMMA fragments in §4 and with raw
`mma.sync` instructions in §5.

---

## 1. What is a Tensor Core?

Each SM contains a small number of Tensor Cores: fixed-function units that compute a
**small dense matrix multiply-accumulate** as a single instruction.

```
    D = A · B + C
```

where the matrices are tiny (e.g. 16 × 8 × 16 for FP16 inputs in the m16n8k16 shape
the hardware actually issues), all live in registers, and the whole operation
completes in a few cycles. One MMA instruction issued by *one warp* of 32 threads
computes 16·8·16 = 2048 multiply-adds — the hardware extracts that work from the
Tensor Core array, returns results into the warp's registers, and moves on.

The big-picture trade is the same as anywhere else in HPC: **sacrifice precision for
throughput**. Tensor Cores are most useful in mixed-precision regimes that ML loves:
FP16 / BF16 / TF32 / FP8 inputs, with FP32 (or sometimes FP16) accumulators.

## 2. Generations and what they support

| arch                 | gen | new types                     |
|----------------------|-----|-------------------------------|
| Volta (V100)         | 1   | FP16 (with FP16 / FP32 acc)   |
| Turing (T4 / 20xx)   | 2   | + INT8, INT4                  |
| Ampere (A100, 30xx)  | 3   | + BF16, TF32, FP64; cp.async  |
| Ada (RTX 4090)       | 4   | + FP8                          |
| Hopper (H100)        | 4   | + FP8, TMA, async tensor memory, DSMEM |

You're on `sm_89` (Ada / 4th gen). All the precisions through FP8 are available.
`async tensor memory` (the Tensor Memory Accelerator, TMA) is **Hopper-only** —
Ada has `cp.async` (Module 8) but no TMA. See §7 for the Hopper successor story.

### RTX 4090 throughput (NVIDIA's published peaks, dense)

| op                                  | TFLOPs |
|-------------------------------------|--------|
| FP32 (CUDA cores, no Tensor Cores)  |  82.6  |
| FP16 / BF16 / TF32 → FP32 acc (TC)  | 165.2  |
| FP16 / BF16        → FP16 acc (TC)  | 330.3  |
| FP8 → FP32 acc (TC)                 | 660.6  |

In practice cuBLAS hits ~140-160 TFLOPs of FP16/FP32-acc on a 4096³ problem, so the
Tensor Core path is genuinely ~2× faster than CUDA cores for this shape. Our naive
WMMA implementation in this module will be much slower than that — closing the gap
requires *async copy with double buffering* (Module 8) and careful avoidance of
shared-memory bank conflicts (§3). The point of this module is to introduce the
APIs, not to match cuBLAS.

## 3. Swizzled shared memory

> **Anchor for forward-refs from Module 3.** This is the section M03's "swizzling"
> pointer leads to.

The shared-memory tiling we inherit from Module 6 stores `As[BM][BK]` row-major,
with `BK = 16` halves = 32 bytes per row. Module 3 taught a `+1`-padding trick
that fixes bank conflicts for 4-byte loads. **It does not work for the 16-byte
loads tensor cores need.** Here's why.

### The bank-conflict trap on `load_matrix_sync`

Shared memory has 32 banks of 4 bytes each. One row of `As` is 32 bytes = 8 banks
wide (banks 0..7). Adjacent rows offset by stride `BK = 16` halves = 32 bytes,
so row `r` *starts* at bank `(r · 8) mod 32 = (r mod 4) · 8`. Rows 0,4,8,12 all
start at bank 0; rows 1,5,9,13 at bank 8; etc.

`load_matrix_sync` for a `matrix_a` row-major fragment of 16×16 fp16 distributes
each column across pairs of lanes — every 4 lanes hit the *same column* of the
fragment, which means the *same starting bank*. Result: a 4-way bank conflict on
*every* fragment load.

```
row r in As           |   bank assignment of half c (c in [0,16))
                      |   bank = (r·8 + c/2) mod 32
─────────────────────┼─────────────────────────────────────────
r = 0                 |   banks  0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
r = 1                 |   banks  8, 8, 9, 9,10,10,11,11,12,12,13,13,14,14,15,15
r = 4                 |   banks  0, 0, 1, 1, ...     (collides with r=0)
r = 8                 |   banks  0, 0, 1, 1, ...     (collides with r=0,4)
```

The naive +1 padding (per Module 3) doesn't fix this for 16-byte vector loads
because the stride is so coarse that any padding under one full bank-row (32×4
= 128 bytes) just shuffles the conflict pattern.

### Two fixes

1. **Padded layout (used in `gemm_v1_wmma_swizzled`).** Store `As` with leading
   dimension `LDAs = BK + 8` halves (= 24 halves = 48 bytes). Now adjacent rows
   shift by 12 banks (rather than 8), so the cycle that previously visited only
   banks {0, 8, 16, 24} now visits {0, 12, 24, 4, 16, 28, 8, 20, ...} — a full
   permutation of all 32 banks across 8 consecutive rows. This is what CUTLASS
   calls a *padded layout*. Cost: `+8 × BM` halves of shared memory.

2. **XOR-swizzle (production form).** Permute the column index of `As[r][c]` by
   XOR-ing `c` (or `c/8` for 16-byte chunks) with a row-derived value:

   ```
   write   As[r][c ^ ((r >> 1) & 7)] = A[r][c]
   read    As[r][c ^ ((r >> 1) & 7)] when fetching A[r][c]
   ```

   The XOR with `(r/2) & 7` distributes the 8 chunks across all 8 bank groups
   without needing extra storage. This is the layout `ldmatrix` and the CUTLASS
   `Layout::TensorOpMultiplicand` use under the hood; for an explicit treatment
   see Module 13's `ldmatrix` example.

`gemm_v1_wmma_swizzled` in `kernels.cuh` uses the padded form because it's
clearer pedagogically — the more elaborate XOR pattern shows up in the Module 8
async-copy follow-up and (in production form) in CUTLASS.

## 4. The WMMA C++ API

There are three ways to drive Tensor Cores from CUDA:

1. **WMMA C++ (`#include <mma.h>`, `nvcuda::wmma`)** — what we use in §4.
   Warp-wide, compiler-level abstractions over PTX `mma` instructions.
2. **PTX `mma.sync`** — inline assembly, finer-grained control over fragment
   layout. Used by CUTLASS, FlashAttention, FasterTransformer internally. We
   use this in §5.
3. **cuBLAS / CUTLASS** — call somebody else's optimized kernel. Often the right
   answer in production.

The WMMA model is built around three types:

```cpp
using namespace nvcuda::wmma;

// One 16×16 chunk of A, FP16 elements, row-major in shared/global memory
fragment<matrix_a,    16, 16, 16, half, row_major> a_frag;

// One 16×16 chunk of B, FP16, row-major
fragment<matrix_b,    16, 16, 16, half, row_major> b_frag;

// 16×16 accumulator, FP32 (mixed precision)
fragment<accumulator, 16, 16, 16, float>           c_frag;
```

A fragment holds the warp's *collective* data for one 16 × 16 chunk — but each lane
holds only its 8 elements (256 elements / 32 lanes = 8). Three primitives drive them:

```cpp
load_matrix_sync (a_frag, ptr, stride);                // load from shared/global
mma_sync         (c_frag, a_frag, b_frag, c_frag);     // c = a · b + c   (one MMA)
store_matrix_sync(ptr, c_frag, stride, mem_row_major); // store accumulator
```

All three are **warp-collective**: every lane in the warp must call them with the
same arguments. The lane-to-element mapping inside a fragment is opaque — you don't
get to dictate it; you just trust the API. (See § "Beyond WMMA" below, and Module 13
[§ mma.sync vs WMMA](../13-ptx-appendix/README.md#mmasync-vs-wmma) for what's
actually under the hood.)

### Putting it together — evolving v6

The block-level kernel structure mirrors Module 6's v6 verbatim:

```
Block tile       BM × BN  =  128 × 128
Warp tile        WM × WN  =   64 ×  64    (4 warps/block, 128 threads)
WMMA fragment    16 × 16 × 16              (the hardware unit)
```

Per warp, you allocate a 2D array of accumulator fragments of shape
`(WM/16) × (WN/16) = 4 × 4 = 16` fragments (1024 fp32 elements per warp).
You walk the K dimension in chunks of `BK = 16` (= `WMMA_K`); for each chunk, you
load 4 `a_frag`s and 4 `b_frag`s, then `mma_sync` into the 16 accumulators. After
the K loop, you `store_matrix_sync` the accumulators to `C`.

The shared-memory tiling and cooperative load are **unchanged** from Module 6.
The diff vs v6 is exactly:

```cpp
// v6 inner FMA loop (06-gemm/kernels.cuh):
for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // ... regM / regN loads + outer-product FMAs ...
}

// v0_wmma inner loop:
for (int kk = 0; kk < BK; kk += WMMA_K) {
    for (int wm = 0; wm < WMITER; ++wm) {
        load_matrix_sync(a_frag, &As[(warpRow*WM + wm*WMMA_M)*BK + kk], BK);
        for (int wn = 0; wn < WNITER; ++wn) {
            load_matrix_sync(b_frag, &Bs[kk*BN + warpCol*WN + wn*WMMA_N], BN);
            mma_sync(c_frag[wm][wn], a_frag, b_frag, c_frag[wm][wn]);
        }
    }
}
```

Same warp tile, same tile loads, different inner loop. That's the whole module
on one slide.

### m16n16k16 fragment-layout caveat

The visualization at `viz/wmma-fragment-layout.html` renders the m16n16k16
fragment as a 2× tile of the spec'd m16n8k16 layout. This is illustrative,
not authoritative: `wmma::fragment<...>` is officially **opaque** — NVIDIA
declines to specify its lane-element mapping precisely, and is free to change
it across PTX versions. Under the hood the WMMA wrapper actually issues two
m16n8k16 `mma.sync` instructions to cover one m16n16k16 tile, but you can't
observe (or rely on) that in source code. Treat the viz as a useful mental
model that happens to match current `ptxas`, not as an API contract.

## 5. Beyond WMMA: mma.sync and ldmatrix

> **Forward refs:** Module 13 [§ mma.sync vs WMMA](../13-ptx-appendix/README.md#mmasync-vs-wmma)
> shows the same instruction with PTX/SASS side-by-side; Module 13
> [§ ldmatrix — loading fragments from shared memory](../13-ptx-appendix/README.md#ldmatrix-loading-fragments-from-shared-memory)
> covers the cooperative-load instruction `ldmatrix` that pairs with `mma.sync`.

WMMA is convenient but limits two things production code cares about:

1. **Epilogue fusion.** You can't poke at fragment elements directly — the
   layout is opaque, so you can't (e.g.) apply a per-element bias in registers
   between `mma_sync` and `store_matrix_sync` without going through shared
   memory and back. Module 9's fused-epilogue exercises hit this wall.
2. **Tile-shape flexibility.** WMMA exposes a few tile shapes (m16n16k16,
   m32n8k16, m8n32k16). The hardware actually has more (m16n8k16, m16n8k8 for
   FP16; m16n8k32 for INT8; m64n8k16 for FP8 on Hopper; etc.). To pick a shape
   that isn't WMMA's, you have to drop down.

The PTX form is `mma.sync.aligned.<shape>.<layoutA>.<layoutB>.<typeD>.<typeA>.<typeB>.<typeC>`.
The Ada workhorse for FP16 inputs / FP32 accumulator is

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Per-warp inputs:
- A: 16×16 row-major fp16, distributed as 8 fp16 per lane in 4 b32 regs.
- B: 16×8  col-major fp16, distributed as 4 fp16 per lane in 2 b32 regs.
- C/D: 16×8 row-major fp32, distributed as 4 fp32 per lane.

For lane `L` (let `g = L/4`, `t = L%4`):

| Fragment | Per-lane content                                                           |
|----------|----------------------------------------------------------------------------|
| A        | `A[g, 2t..2t+1]`, `A[g+8, 2t..2t+1]`, `A[g, 2t+8..2t+9]`, `A[g+8, 2t+8..2t+9]` |
| B        | `B[2t..2t+1, g]`, `B[2t+8..2t+9, g]`                                       |
| C        | `C[g, 2t..2t+1]`, `C[g+8, 2t..2t+1]`                                       |

`gemm_v2_mma_sync` in `kernels.cuh` uses this layout, with the same block tile
as v0/v1: 4 warps/block, each warp owns a 64×64 tile, each tile is 4×8 = 32
m16n8k16 sub-tiles per warp per K-step. The packing code is the bit that makes
WMMA "feel" easy by comparison.

### A note on `ldmatrix`

The natural input to `mma.sync` is fragments that came from shared memory.
Writing the per-lane shared-memory loads by hand (as `gemm_v2_mma_sync` does
for clarity) is correct but unidiomatic — production code uses
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` to cooperatively load four 8×8 fp16
matrices in one warp-collective instruction, with the lane layout `mma.sync`
expects already baked in. See Module 13 [§ ldmatrix](../13-ptx-appendix/README.md#ldmatrix-loading-fragments-from-shared-memory)
for the PTX form and a worked example.

## 6. Accumulator dtype tradeoff (FP16 vs FP32, and TF32)

WMMA / `mma.sync` let you choose the accumulator type independently of the input
type:

| inputs | acc | RTX 4090 peak | when to use |
|--------|-----|---------------|-------------|
| FP16 / BF16 | FP32 | 165 TFLOPs   | most ML training; numerical safety floor |
| FP16        | FP16 | 330 TFLOPs   | ML inference where overflow is bounded |
| TF32        | FP32 | 165 TFLOPs   | "FP32-ish" with TC speed; A100/Ada only |
| FP8         | FP32 | 660 TFLOPs   | inference, with per-tensor scaling |

The FP16-acc path is *2× faster* but accumulator overflow / catastrophic
cancellation can bite — for a row of 4096 fp16 multiplies summed in fp16,
you'll lose mantissa bits halfway through. The standard mitigations are
*split-K accumulation* (accumulate in fp16 within a tile, fp32 across tiles)
or *scaled accumulation* (rescale every K′ steps).

**TF32 is a callout, not an exercise.** It's an FP32 input format that drops
13 mantissa bits before going through the FP32-acc path, giving you "FP32
correctness" with FP16-input throughput. The cuBLAS compute type
`CUBLAS_COMPUTE_32F_FAST_16F` toggles it; nothing about your CUDA C++ source
changes.

## 7. Beyond Ada (Hopper successor)

`sm_89` is Ada (4th-gen Tensor Cores). The next architecture, Hopper (`sm_90`,
H100), adds three things that change how you write a tensor-core kernel:

- **Thread-block clusters + DSMEM.** Up to 16 blocks form a *cluster* whose
  blocks can read/write each other's `__shared__` over a new SM-to-SM
  fabric. Distributed shared memory (DSMEM) lets a multi-block GEMM tile
  cooperate without round-tripping through L2 — the Ada / Ampere design
  treats each block as an island; Hopper makes them neighbors.
- **TMA — `cp.async.bulk.tensor`.** A descriptor-based tensor-load instruction
  that issues one PTX op per multi-D tile copy. The descriptor encodes shape,
  strides, layout, and (optionally) swizzling; the hardware does the
  per-element address math. On Ada you write the swizzle by hand (§3); on
  Hopper TMA does it for you. Module 8's `cp.async` is the conceptual
  predecessor.
- **FP8 in production.** Already on Ada hardware-wise, but Hopper adds
  `wgmma.mma_async` (warp-group async MMA) that fuses TMA-style loads with
  the FP8 mma into one async pipeline. This is what production LLM inference
  in 2025+ runs on.

### FlashAttention-3 (2024) — what the new primitives unlock

The headline application of the three Hopper primitives above is
**FlashAttention-3** (Dao et al., 2024). FA-3's perf on H100:

- FP16: **~740 TF/s** = **75%** of H100's 990 TF/s tensor peak (vs FA-2 on
  H100 at ~550 TF/s = 55% of peak)
- FP8: ~1.2 PF/s

The +20-percentage-point efficiency over FA-2 *on the same hardware* comes
from three things, all enabled by the primitives above:

1. **WGMMA** (warp-group MMA): one warp-group (4 warps) issues one async
   m64n64k16-sized MMA. Bigger fragments, fewer issue slots, overlap with
   other warp-groups' work in the same SM.
2. **TMA**: zero per-element address math. The hardware streams tiles
   asynchronously from global to shared while warps work on other things.
3. **Ping-pong warp-group specialization**: half the warp-groups run
   GEMM 1 (Q·K^T) of block A while the other half run GEMM 2 (P·V) of
   block A−1. Hides softmax behind matmuls — the bottleneck FA-2 still
   has.

**None of this hardware is on Ada (sm_89).** Module 10's optimization
ladder ends at FA-2-shape perf because that's what the hardware allows;
the path to FA-3 numbers is hardware, not algorithm. M10.7's writeup
notes which FA-3 ideas (warp specialization) are *partially* expressible
on Ada and which are Hopper-locked.

When you read CUTLASS 3.x or FlashAttention 3 source, the three primitives
are most of what's new versus CUTLASS 2 / FlashAttention 2.

## 8. Where Tensor Cores *don't* help

- **Tiny matrices** (M, N, or K < 16). The MMA shape is fixed; if your dims
  aren't multiples of 16, you either pad (wasting compute) or fall back to
  non-TC code.
- **Memory-bound regimes.** If your kernel is bandwidth-bound, swapping the
  inner multiplier for a faster one doesn't help — you're stuck on the DRAM
  roof.
- **High-precision needs.** FP16/BF16 inputs are not always good enough; if
  you need FP32 throughout, Tensor Cores can only buy you the TF32 mode
  (which still loses 13 bits of mantissa).
- **Non-GEMM-shaped work.** Tensor Cores compute small dense matmuls. Scans,
  reductions, stencils, hash lookups — they don't fit.

This is why "we made it Tensor Cores" is rarely a one-line speedup story. The
restructuring usually requires you to redesign your kernel around 16-element
tiles and accept the precision trade.

---

## Exercises

> Open `starter.cu` and complete the TODOs.

1. **Allocate accumulator fragments** for the warp's `(WM/16) × (WN/16) = 4×4`
   chunks of C and zero them with `fill_fragment`.
2. **Inside the K loop**, after `__syncthreads()`, walk `kk` over `BK` in steps
   of `WMMA_K = 16`. For each `kk`, load each `a_frag` (one per `wm` chunk) and
   each `b_frag` (one per `wn` chunk), and call `mma_sync`. The right inner
   loop order matters for register reuse — load a single `a_frag`, then loop
   over `wn`s with `b_frag`s and accumulate.
3. **After the K loop**, `store_matrix_sync` each accumulator into the right
   slot of the `C` block.

The verification harness compares your output against cuBLAS FP16 GEMM. Tolerance
is generous because FP16 accumulation has more rounding error than FP32.

### Stretch

- **Swizzled shared memory.** Compare your v0 to `gemm_v1_wmma_swizzled` in
  `kernels.cuh`. Read §3, then change v0's `As` to use the padded leading
  dimension and observe the bank-conflict count change in `ncu --metrics
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bench`.
- **Raw `mma.sync`.** Read §5 and the m16n8k16 lane-element table. Then read
  `gemm_v2_mma_sync` in `kernels.cuh`. Modify its accumulator dtype to fp16
  (compose your own `.f16.f16.f16.f16` instruction) — note the per-lane c
  registers shrink from 4 fp32 to 2 b32.
- **FP16 accumulator (WMMA).** `gemm_v0_wmma_fp16acc` is the FP16-acc form of
  v0. ~2× speedup over FP32 acc on Ada at the cost of accumulator precision.
- **`half2` writes.** Switch the FP16-acc path to write FP16 output with
  `half2` for two-element vectorized stores.
- **cuBLAS comparison.** Set the cuBLAS compute type to
  `CUBLAS_COMPUTE_32F_FAST_16F` (TF32-style). Same FP32 inputs as Module 6,
  but Tensor Core accelerated. Where does it land?

---

## Profiler checklist

```bash
make
ncu --set full ./bench
```

Look at:

- **`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`** —
  Tensor Core utilization. Should be substantially > 0; cuBLAS-quality kernels
  reach 50–70 %.
- **`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`** — should drop
  from v0 to v1 (the swizzled variant) by roughly 4× on the `As` loads.
- **Compute (SM) Throughput** — should be much higher than the FP32 number
  from Module 6.
- **DRAM Throughput** — *also* should be lower as a fraction of peak (FP16
  halves the input bytes per FLOP), confirming we've moved further into the
  compute-bound regime.

## Key takeaways

- Tensor Cores compute a small dense matmul as one warp-wide instruction.
- WMMA C++ wraps it in three primitives: `load_matrix_sync`, `mma_sync`,
  `store_matrix_sync`. Fragments are warp-collective; the lane mapping is
  opaque. Use this for prototyping and for code that doesn't need fragment-
  level fusion.
- `mma.sync` is the underlying PTX instruction. Production ML kernels
  (CUTLASS, FlashAttention, FasterTransformer) use it directly because
  fragment layout is documented and stable, enabling epilogue fusion and
  custom tile shapes.
- The 16-byte shared-memory loads tensor cores need expose bank-conflict
  patterns the +1-padding fix from Module 3 doesn't solve. Padded layouts
  or XOR-swizzles are the real fix; see §3.
- Measured on RTX 4090 at 4096³ FP16-in / FP32-acc: WMMA v0 lands at
  **~40% of cuBLAS hgemm** (62.9 / 159.0 TFLOPs). Adding shared-memory
  swizzling (v1) doesn't move it — the bottleneck is elsewhere. Raw
  `mma.sync` (v2), which exposes the documented fragment layout and lets
  the compiler keep more of the working set in registers, hits **~76%
  of cuBLAS** (120.7 TFLOPs). The remaining gap is hand-tuned scheduling
  (CUTLASS / cuBLAS internals) and `cp.async` double-buffering (M08).
- Restructuring around 16-element tiles is the price of admission.
- Hopper adds DSMEM (cross-block shared memory in clusters), TMA
  (`cp.async.bulk.tensor`), and `wgmma` (warp-group async MMA). Ada has
  the FP8 hardware but not the cluster / TMA / wgmma plumbing.
