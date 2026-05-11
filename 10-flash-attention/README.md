# Module 10 — FlashAttention

**Goal:** by the end of this module you should be able to (a) explain the
*online softmax* recurrence and how it tiles, (b) implement a working
FlashAttention forward pass in CUDA at five points along the optimization
ladder — from "one thread per Q row" through warp-cooperative FP32, WMMA
tensor cores, cp.async double-buffering, all the way to a raw `mma.sync`
register-resident-softmax kernel at FlashAttention-2 shape perf — and (c)
describe the four common shape variants (MHA, causal, KV-cache, GQA) and
the path from there to FlashAttention-2/3.

This module is the most algorithmically dense in the course. The *math* is
all in §1-§4. The *engineering ladder* — actually optimizing the kernel for
the GPU — is in §5 onward.

---

## 1. The online softmax recurrence (recap)

You already built online softmax in **Module 9** — the running-max + rescaled
running-sum recurrence is in `09-fused-epilogues/kernels.cuh::softmax_online`.
FlashAttention is what you get when you apply that same recurrence to the
softmax that lives *inside* an attention computation, with one twist: the
elements arrive in *tiles* (a chunk of the score row at a time, not one
element at a time), and the recurrence has to maintain the partial output
`O` alongside the `(m, l)` state.

Quick refresher of the rule. Given the running pair `(m, l)` and a new score
`x`:

```
m_new = max(m, x)
l_new = l · exp(m - m_new) + exp(x - m_new)
m, l ← m_new, l_new
```

For tiles of multiple scores `(x₁, …, x_BC)` with local max `m_tile` and local
sum `l_tile = Σ exp(xᵢ - m_tile)`, the combine becomes

```
m_new = max(m_state, m_tile)
α     = exp(m_state - m_new)
β     = exp(m_tile  - m_new)
l_new = α · l_state + β · l_tile
```

And critically — this is what FlashAttention adds — the running output `O` is
also rescaled by `α`:

```
O_new = α · O_state + β · (P_tile · V_tile)
```

where `P_tile = exp(S_tile - m_new)`. (Folding the `β` into the row of `P` is
the same as multiplying by it after the matmul; the version in the kernel just
uses `m_new` directly when computing `P_tile`, so the explicit `β` disappears.)

## 2. The problem

Standard scaled-dot-product attention is:

```
S = Q · K^T / √D            // [N, N]
P = softmax(S, axis=-1)     // [N, N]
O = P · V                   // [N, D]
```

with `Q, K, V ∈ ℝ^{N×D}`. The middle two matrices, `S` and `P`, are **N×N** —
quadratic in sequence length. For `N = 8192`, that's 256 MB in FP32 *per
attention layer*, before even thinking about gradients.

Materializing those matrices is the bottleneck at long sequences. The
KV-cache calculation that often gets quoted —

> "32K context, 32-head model, 16 GB per layer just for the attention matrix"

is `H × N × N × bytes = 32 · 32768² · sizeof(fp16) = 64 GB` — but only one
*row of S* per query is needed at a time, and that's 32K · 2 B = 64 KB per
row. The "16 GB" figure quoted in some FA writeups conflates the materialized
`P` matrix with the KV cache itself; the **KV cache** is `2 · L · H · N · D ·
bytes` (factor 2 for K and V; `L` layers): for 32 K, 32 layers, 32 heads,
`D=128`, fp16, that's `2 · 32 · 32 · 32768 · 128 · 2 ≈ 16 GB` — *that's* where
the 16 GB number comes from, and it's about the cache, not the attention
matrix. FlashAttention solves the `P` problem; the KV cache is a separate
concern that we'll address with a dedicated kernel below.

## 3. The FlashAttention-1 algorithm

Pick block sizes `BR` (rows of `Q` per block) and `BC` (rows of `K`/`V` per
inner tile). For each *row block of Q*:

```
load Q_i (BR × D) into shared / registers
m_i = -∞ (vector of length BR)
l_i =  0 (vector of length BR)
O_i =  0 (BR × D)

for each column block j ∈ [0, N/BC):
    load K_j (BC × D), V_j (BC × D) into shared
    S_ij = (Q_i · K_j^T) / √D                 # BR × BC, in registers
    m_ij = rowmax(S_ij)                       # BR
    P_ij = exp(S_ij - m_ij[:, None])          # BR × BC
    l_ij = rowsum(P_ij)                       # BR

    m_new = max(m_i, m_ij)                    # BR
    α = exp(m_i  - m_new)                     # BR
    β = exp(m_ij - m_new)                     # BR
    O_i = α[:, None] * O_i + β[:, None] * (P_ij · V_j)
    l_i = α * l_i + β * l_ij
    m_i = m_new

write O_i / l_i[:, None] to O
```

The `O_i / l_i` at the end completes the deferred normalization — we never had
to divide by `Σ exp(...)` until the whole row was visited.

## 4. Why this is a big deal

- **Memory.** Peak DRAM use for attention drops from `O(N²)` to `O(N·D)` for
  the materialized intermediates (`S`, `P`). That's the difference between a
  32-K context fitting on a single GPU and not.
- **Bandwidth.** Inputs are read more times (each K/V tile is read by every
  row block), so total bytes moved are **not** asymptotically smaller — flash
  is still O(N²·D) work, same as naive. What flash wins is the *constant*: it
  never writes `S` or `P` to DRAM. For typical N at typical D, that's ~2-3×
  fewer total bytes moved than the three-kernel naive version.
- **Wall-clock.** Even at "small" N where memory wasn't the binding
  constraint, fusion saves kernel launches and DRAM round trips.

So no, flash does **not** "scale linearly with N" — it's still quadratic. It
just has a much smaller bandwidth constant per quadratic FLOP, which is what
makes long-context inference and training feasible.

### Aside: a different shape — state-space models

Attention scales O(N²) compute with O(N) state per layer. **State-space
models** like Mamba (and the broader linear-attention family) scale O(N)
compute with O(1) state per token — the inner operation is a parametric
parallel scan (see M05 §5.3) rather than a Q·Kᵀ matmul. For inference
specifically, that O(1) state per token means **no KV cache**, which is a
qualitatively different deployment story.

Same design ethic — keep state in SRAM, never materialize between time
steps — applied to a different inner operation. M12 Capstone Project E
covers Mamba inference as a megakernel; if you finish this module and
want a "what's the alternative to attention" project, that's it.

---

## 5. The optimization ladder

The §3 algorithm is *the same* at every rung of this ladder. Only the
thread-→-data mapping and the inner matmul implementation change. We deliver
three working kernels and a stretch spec:

| Rung   | Kernel                          | Inputs   | Inner matmul        | Headline N=8192 |
|--------|---------------------------------|----------|---------------------|-----------------|
| 10.0   | `flash_attention`               | FP32     | per-thread scalar   | 6.9 TF/s        |
| 10.1   | `flash_attention_warp`          | FP32     | per-thread scalar   | 9.3 TF/s        |
| 10.2   | `flash_attention_wmma`          | FP16     | WMMA tensor-cores   | 22 TF/s         |
| 10.3   | `flash_attention_async_wmma`    | FP16     | WMMA + cp.async ×2  | 28 TF/s         |
| 10.4   | `flash_attention_mma`           | FP16     | raw `mma.sync` + cp.async | 62 TF/s   |
| 10.5   | `flash_attention_mma_ldmatrix`  | FP16     | `mma.sync` + `ldmatrix` fragment loads | 61 TF/s |
| 10.6   | `flash_attention_mma_swizzled`  | FP16     | `mma.sync` + `ldmatrix` + XOR-swizzled smem | 90 TF/s |
| 10.7   | `flash_attention_mma_fa2`       | FP16     | 10.6 + STAGES=3 + K-via-ldmatrix + Q-overlay | 100 TF/s |

(Numbers are min-of-N timing on RTX 4090, single head, D=64; see
`bench.cu`. The cuBLAS hgemm peak on this card is ~159 TF/s, so 10.7 is at
~60% of peak — and lands solidly inside the production FA-2 / CUTLASS-shape
band on this card. The 10.5 vs 10.4 result — essentially tied — is the
diagnostic that motivates 10.6; see §§11–12. The +10% step from 10.6 to 10.7
stacks three FA-2-shape levers with a shared-memory overlay; see §13.)

**Roofline at this exact shape:** the compute ceiling for FP16-in / FP32-acc
on RTX 4090 is ~165 TF/s; cuBLAS run as two separate hgemms (QK^T then PV)
only hits ~39 TF/s combined at our skinny K=64, so **our fused 10.4 beats the
cuBLAS-split baseline by 1.5×**. Production FA-2 / CUTLASS-shape kernels on
this card land at ~80-130 TF/s. The roofline tool (`make roofline-run`)
reproduces the cuBLAS reference numbers; full breakdown in
[`BENCH-RESULTS.md`](../BENCH-RESULTS.md) under "Roofline reference".

### 5.1 The four shape variants

Orthogonal to the optimization ladder, real production attention shows up in
several "shapes": multi-head, causal, KV-cache, GQA. They're all the same
algorithm with different head-index and masking patterns. We deliver all four
on top of the simplest base (10.0):

| Kernel                        | What it does                                              |
|-------------------------------|-----------------------------------------------------------|
| `flash_attention_mha`         | Multi-head: indexed by `(batch, head, q_row)`             |
| `flash_attention_causal`      | Causal mask + tile-skip (lower-triangular S)              |
| `flash_attention_kvcache`     | One-query inference attending to `K[0..T-1]` from a cache |
| `flash_attention_gqa`         | Grouped-query: `H_q` query heads, `H_kv < H_q` shared KV  |

Each is a thin wrapper over the §3 loop. **They're written on top of 10.0**
because the shape concern is independent of the speed concern. Porting them to
the warp-cooperative or WMMA base is a mechanical exercise (see "Stretch"
below).

## 6. Rung 10.0 — "one thread per Q row" (FP32, the pedagogical base)

```c
// kernels.cuh, flash_attention()
// BR = BC = 32, D = 64. Block: BR threads (one warp). Grid: N/BR blocks.
//
// Each thread owns one Q row. K/V tile is loaded cooperatively into shared
// memory, then every thread does its own D-element dot product against
// every K column — no cross-lane reuse.
```

This is the simplest possible CUDA implementation of §3. Every thread runs
the entire FA recurrence for one Q row, in scalar code. The K/V tile loads
are cooperative across the warp, but the dot-product work is not.

**Why this is ~10× off peak:** at FP32, the FMA pipe is the bottleneck. With
32 threads/block doing one D-element dot product per K column each, the SM is
running at roughly the same FMA rate as M06 v0 — the *fully un-tiled* GEMM.

## 7. Rung 10.1 — Warp-cooperative (still FP32)

```c
// kernels.cuh, flash_attention_warp()
// BR1 = 128 (rows per block), BC1 = 32, D1 = 64. Block: 128 threads.
```

The single change from 10.0 is **block size 32 → 128**, so each block now
services BR1 = 128 Q rows from one cooperative K/V tile load. Inner math is
unchanged. The K/V load is now amortized over 4× more rows, and the SM has 4×
more in-flight work to choose from.

For pure FP32 attention this turns out to be the lever that matters: at this
shape the FMA pipe is well-fed and no amount of cleverness with shuffles or
warp-cooperative dot products beats "more rows per K-tile load" in the simple
register-file regime. Roughly +40% over 10.0.

A *true* warp-cooperative inner matmul (lane c computes column c of S, etc.)
is what M07's GEMM v6 does, and it's what 10.2 does on tensor cores. At FP32
without tensor cores, the pay-off is small — the FMA throughput is already
saturated on the simple form.

## 8. Rung 10.2 — WMMA (FP16 inputs, FP32 accumulator)

```c
// kernels.cuh, flash_attention_wmma()
// BR2 = 16, BC2 = 32, D2 = 64. WARPS_M10_2 = 4.
// Block: 128 threads, BR2_BLOCK = 64 Q rows per block.
```

Now the inner Q·Kᵀ and P·V matmuls run on tensor cores via the `wmma::`
fragment API. FP16 inputs, FP32 accumulator. Block layout:

- 4 warps per block, each warp owns a separate 16×64 Q-tile (registers, kept
  across all inner iterations).
- One block-shared K/V tile (BC2 × D2 = 32 × 64 FP16). 4 warps share the load.
- Per warp: an opaque accumulator fragment for the running output O (16 × D2,
  i.e. 4 fragments of 16×16).

The interesting bit is the **online softmax on fragment-layout S**. WMMA
fragment elements are distributed across lanes in a layout the spec doesn't
fully expose, so we can't run softmax on `s_frag.x[i]` directly. The dance:

1. `wmma::store_matrix_sync(Ssm, s_frag, ...)` — get S out into shared memory
   in row-major.
2. Per-row max + exp on shared `Ssm`. With 16 rows and 32 lanes, lane r
   handles its row; lanes 16..31 idle for this short reduction.
3. Write `P = exp(S - m_new)` to shared as FP16 (`Psm`).
4. **Scale O by α[row]:** `o_frag` is a register-resident accumulator with
   opaque layout, so we materialize all 4 of its 16×16 chunks to shared
   memory (`Osm`, 16×64 floats), scale row-by-row in shared, then load each
   chunk back into a fresh `o_frag`. This is the single most expensive
   non-tensor-core step per iteration.
5. `wmma::load_matrix_sync(p_frag, Psm)` and run the second matmul:
   `o_frag[d] += P · V[:, d*16:(d+1)*16]` for each d-chunk.

What you give up by using `wmma::` instead of raw `mma.sync` is exactly
step 4: with raw `mma.sync` (M07's v2_mma_sync, M13's mma_sync_example) the
register layout is *documented*, so you can scale `c_frag[wm][wn][q] *= alpha`
directly per register without going through shared memory. CUTLASS and
production FlashAttention go that route. We're keeping WMMA for readability;
the shared-memory round-trip is the price of that readability.

**At runtime** this kernel runs at ~22 TF/s at N=8192 — about 14% of the
~160 TF/s cuBLAS hgemm peak. The remaining gap to 50% of peak is mostly
(a) the O-scaling shared-memory round trip in step 4, (b) no `cp.async`
overlap between K/V load and compute, (c) the WMMA fragment layout overhead
that raw `mma.sync` would eliminate. (b) is 10.3; (a) and (c) are 10.4.

## 9. Rung 10.3 — `cp.async` + WMMA

```c
// kernels.cuh, flash_attention_async_wmma()
// BR2 = 16, BC2 = 32, D2 = 64. WARPS_M10_2 = 4. STAGES = 2.
// Block: 128 threads, BR2_BLOCK = 64 Q rows per block.
// Shared mem ≈ 45.6 KB / block (fits the default 48 KB — no opt-in needed).
```

The 10.2 kernel has a hard barrier between "load K/V tile from DRAM" and
"compute on K/V tile" — both happen in the same iteration of the outer loop,
so the SM stalls for memory during the load and stalls for compute during
the WMMA. M08's `cp.async` (`gemm_v1_async`) overlaps the two; 10.3 does
exactly that on top of 10.2.

**Double-buffered shared memory.** Two copies of `BC2 × D2` halves each
for K and V, indexed `Ks[STAGES][BC2*D2]` and `Vs[STAGES][BC2*D2]`. With
`STAGES = 2` we pay an extra 8 KB on top of 10.2's ~38 KB — total ~45.6 KB,
still under the default 48 KB per block, so no `cudaFuncSetAttribute` opt-in.

**Pipeline shape.** Same as M08's `gemm_v1_async`:

```
issue_loads(K[0], V[0] -> Ks[0], Vs[0])
__pipeline_commit()

for j in [0, Tc):
    if j + 1 < Tc:
        issue_loads(K[j+1], V[j+1] -> Ks[(j+1) % 2], Vs[(j+1) % 2])
    __pipeline_commit()
    __pipeline_wait_prior(STAGES - 1)   // = wait_prior(1): oldest commit done
    __syncthreads()

    // M10.2 inner body on Ks[j % 2], Vs[j % 2]:
    //   Q · K^T via WMMA → S_frag
    //   store → row-softmax → write FP16 P → shared
    //   rescale O by alpha (shared-memory round trip)
    //   P · V via WMMA → o_frag accumulator

    __syncthreads()   // WAR fence — see M08 §3a comment
```

**Out-of-bounds handling.** Each cp.async transfer is 16 bytes (8 halves).
For the final tile, rows past `N` would issue from out-of-bounds global
addresses; instead we clamp the source row to `min(kcol, N-1)`, getting
finite garbage. The post-softmax `Ssm[col >= N] = -INFINITY` mask zeroes the
corresponding P columns, so the garbage Vs never reach the output. (The
garbage must be finite, not NaN/inf, because `0 · NaN = NaN`.)

**One micro-optimization** worth calling out: on iteration 0, the running
`o_frag` is identically zero (`fill_fragment`) and `alpha == 0` (because
`m_state == -INFINITY` is the identity), so the shared-memory round trip
that rescales O by alpha is a no-op. Skipping it on iter 0 saves one
store/scale/load cycle per block — a small but free win.

**At runtime** this kernel hits ~28 TF/s at N=8192 (1.27× over 10.2's 22
TF/s, on the low end of the 1.3–1.5× spec-projection). The gap to the
projected 30 TF/s is the per-iter softmax-on-fragments dance and the
O-rescale shared-memory round trip; both go away with raw `mma.sync`,
which is what §10 (Rung 10.4) does.

We use the legacy `__pipeline_memcpy_async` API from `<cuda_pipeline.h>`
(emits `cp.async.cg.shared.global ..., 16, 16`), matching M08's
`gemm_v1_async`. The modern `cuda::pipeline` wrapper is functionally
equivalent — see M08 §8.2 for the comparison.

## 10. Rung 10.4 — Raw `mma.sync` (the FA-2 shape)

```c
// kernels.cuh, flash_attention_mma()
// BR4 = 16, BC4 = 32, D4 = 64. WARPS_M10_4 = 4. STAGES = 2.
// Block: 128 threads, BR4_BLOCK = 64 Q rows per block.
```

10.3 left two big costs on the table: the **softmax-on-fragments** round
trip (store `S` to shared, read row-wise, write FP16 `P` back) and the
**O-rescale** round trip (store `O` to shared, multiply by α row-wise,
load back). Both exist because `wmma::fragment` has an *opaque* layout —
the spec doesn't say which lane holds which element of the 16×16
accumulator, so we can't operate on `o_frag.x[i]` row-aware in registers.

Raw `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` fixes that. The
PTX ISA documents the per-lane register layout exactly. For the D / C
operand (FP32, 16×8):

```
Lane (q, t)   where  q = laneIdx / 4 ∈ [0,8),  t = laneIdx % 4 ∈ [0,4)
  d[0] = (row q,   col 2t)        d[1] = (row q,   col 2t+1)
  d[2] = (row q+8, col 2t)        d[3] = (row q+8, col 2t+1)
```

The instruction is m16-by-n8 — 16 rows of output, 8 cols. Each row is
owned by 4 lanes (one `t`-group), each holding 2 of its 8 col entries.
Row max and row sum become a tiny in-register reduction:

```c
v = fmaxf(d[0], d[1]);                                  // 2 cols local
v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));        // 4 cols
v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));        // 8 cols = full row
```

No shared memory. Same for the row-sum. And the per-lane O accumulator
is just a `float o_frag[8][4]` — rescaling by α[row] is a plain scalar
multiply on registers.

The other thing we gain is the **register-resident P→A repack** for the
P·V mma:

```
S frag for n=0 (cols 0..7) → A reg a0, a1     # row q   / row q+8
S frag for n=1 (cols 8..15) → A reg a2, a3    # cols 0..7 / 8..15 of A frag
```

The D-fragment layout of a 16×8 mma output (over cols 0..7) plus the
D-fragment layout of the adjacent mma (cols 8..15) is exactly the A-
fragment layout of a 16×16 A input. So we convert the S-fragments to FP16
*in-place per register*, no shuffle, and feed straight into the next mma.

**Tile decomposition.** `S = Q · K^T` is 16×32 per warp:
  * 4 mma `k`-tiles (D=64 / 16) × 4 mma `n`-tiles (BC=32 / 8) = 16 `mma.sync`s.
`O += P · V` is 16×64 per warp:
  * 2 mma `k`-tiles (BC=32 / 16) × 8 mma `n`-tiles (D=64 / 8) = 16 `mma.sync`s.

We keep the M10.3 cp.async double-buffer pipeline — it's orthogonal to
the softmax fix and free perf.

**Manual fragment loads, not `ldmatrix`.** We pack A (Q) and B (K) operands
by hand from shared memory with `__half2` loads. For Q and K the lane
layout aligns with contiguous 32-bit groups in row-major storage, so the
manual code is two `__half2` reads per lane per mma — clean and fast. V is
the awkward case (B for P·V wants col-major while V is stored row-major),
which costs us 4 scalar `__half` reads per lane per mma. `ldmatrix.x2.trans`
would do this in one PTX instruction; that's the next rung — see §11
(M10.5) for the swap-in and a discussion of why it doesn't actually move
TF/s on this card without a swizzled smem layout.

**At runtime** this kernel runs at ~62 TF/s at N=8192, ~2.2× over 10.3
and ~39% of the cuBLAS hgemm peak. The remaining gap to the FA-2
production number (which lives around 60–80 TF/s on this card) is mostly
the manual transposed V load and the lack of `ldmatrix` shared-bank
swizzling — both pure constant-factor wins on top of the algorithmic shape
this kernel already has.

The shared-memory footprint per block is `STAGES * 2 * BC4 * D4 * 2 = 16 KB`
for the K/V tiles plus `WARPS * BR4 * D4 * 2 = 8 KB` for the Q tiles — 24 KB
total, well under the 48 KB default.

## 11. Rung 10.5 — `ldmatrix` fragment loads

```c
// kernels.cuh, flash_attention_mma_ldmatrix()
// Identical tile shape and algorithm to 10.4. Only the Q and V
// shared→register loads change.
```

10.4 packs the A and B operand registers by hand. For Q (A of `S = Q · K^T`)
that's 4 `__half2` reads per lane per k-chunk — already efficient. For K
(B of `S = Q · K^T` via K^T) that's 2 `__half2` reads per lane — also fine.
The slow one is V (B of `O += P · V`): the B fragment for `m16n8k16` is
col-major 16×8 but V is stored row-major, so each lane does 4 scalar
`__half` reads per (p_chunk, d_chunk), and four lanes in the same t-group
read the same byte offset across different rows — a high-conflict pattern.

`ldmatrix.sync.aligned.m8n8.shared.b16` is the "right" way to feed
`mma.sync` from shared memory: one warp-cooperative instruction loads
8×8 fp16 tiles directly into the lane layout `mma.sync` expects. M13's
`ldmatrix_example.cu` is the canonical reference for the lane→pointer
mapping. The two flavors we use here:

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16  {r0,r1,r2,r3}, [smem];   // A side
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16  {r0,r1}, [smem];    // B side
```

**Q load (`.x4`).** A 16×16 A fragment = four 8×8 tiles. Lane L provides
the row pointer for tile `L/8`, row `L%8`. The four returned `b32`
registers per lane map directly to the `mma.sync` A operand `{a0,a1,a2,a3}`
with no further packing — one PTX instruction per k-chunk replaces eight
`__half2` reads.

**V load (`.x2.trans`).** A 16×8 B fragment = two 8×8 tiles. Read row-major
from shared, but the B operand wants col-major, so we use the `.trans`
variant which transposes each 8×8 during the load. Lane L (0..15) provides
`&Vs[(pc*16 + L) * D4 + dc*8]`; lanes 16..31's addresses are ignored. After
the load, lane `(q,t)` holds `b0 = (rows 2t..2t+1, col q)` and
`b1 = (rows 2t+8..2t+9, col q)` — the exact B layout `mma.sync` wants. One
hardware instruction replaces four scalar `__half` reads per `(pc, dc)`.

**K stays on the manual `__half2` path.** K is the B of `S = Q · K^T`, and
because we store K row-major in shared and the `mma.sync` B layout wants
col-major-of-the-16×8, the manual code happens to align with contiguous
32-bit groups (two `__half2` per lane per mma) — no win to chase here.

**Verification.** Identical `max_abs` to 10.4 at every N: 3.80e-5 / 1.71e-5
/ 1.26e-5 at N ∈ {128, 1024, 2048}. The compiler emits 20 `LDSM` SASS
instructions (4 Q + 16 V loads per outer iteration) replacing 64 vanilla
`LDS` halves in 10.4.

**Bench result — and why it's not the +10-15% you expect.** On RTX 4090
sm_89, 10.5 measures essentially equal to 10.4: 61.0 vs 61.7 TF/s at
N=8192 (within noise). Same registers (95), same shared (24 KB) — the
kernels are byte-for-byte cousins on every metric except how the inner
loop loads from shared.

Why no speedup? `ldmatrix` is a necessary-but-not-sufficient step.
The headline FA-2 / CUTLASS win from "use `ldmatrix`" really comes from
**`ldmatrix` + a swizzled shared-memory layout**. With contiguous
row-major V in shared (our case), the V-side `ldmatrix.x2.trans` hits
the *same* bank-conflict pattern as the manual scalar loads: 16 lanes
all reading the same column offset across different rows = a high-way
conflict on the shared banks. The hardware's transposing wires save
instruction count (20 LDSMs vs 64 LDS) but not the underlying memory-
system throughput.

To unlock the headline gain, the next step is to **swizzle** the K/V
tiles when writing them to shared: each row's column indices XOR'd with
a function of the row index, so a column-major read across rows hits
different banks per lane. That's the change CUTLASS makes; it's a
non-trivial rewrite of the `cp.async` `issue_tile` lambda and of the
ldmatrix pointer math. We do exactly this in 10.6 (§12).

**Pedagogical takeaway.** Use `ldmatrix` to feed `mma.sync` — that's the
production idiom. But don't expect it to be a free win on its own;
without swizzled smem, you're trading the instruction-count cost for
the same conflict-cost as the manual path. The win compounds with
swizzle, and only with swizzle.

## 12. Rung 10.6 — XOR-swizzled shared memory

```c
// kernels.cuh, flash_attention_mma_swizzled()
// Same shape, same mma.sync, same ldmatrix instructions as 10.5.
// Only the shared-memory layout for Q, K, and V changes.
```

10.5 measured a real disappointment: `ldmatrix` cut the instruction count from
64 LDS to 20 LDSM, but the bench was a wash (~61 TF/s vs 10.4's ~62). The
profiler confirmed why: on the *unswizzled* row-major layout, the V-side
`ldmatrix.x2.trans` instruction asks 16 lanes for pointers to 16 different
rows at the **same** column offset, and 128-byte rows in shared map straight
onto 32 banks — a 16-way conflict. The hardware's transposing wires save
instruction count but not memory-system throughput.

The fix is to permute the column indices when *writing* to shared (in
`cp.async`) and *reading* from shared (in `ldmatrix` / packed `__half2`
loads), so the same column-offset across different rows lands on different
banks.

**The XOR swizzle.** For an element logically at `(row, col)` in a tile, we
store it at column `swizzle(row, col)` in the same row:

```
chunk        = col >> 3                       // 16-byte chunk index (0..7)
chunk_swizz  = chunk ^ (row & 7)
col_swizz    = (col & 7) | (chunk_swizz << 3)
```

The swizzle is **a bijection on 8-half chunks** that depends on `row & 7`.
Two key properties:

1. *8-half alignment is preserved.* If `col` is 8-aligned, so is
   `swizzle(row, col)`. So a `cp.async` of 16 bytes (= 8 halves = 1 chunk)
   from `gmem` row-major lands at exactly one chunk-slot in `smem`, and an
   `ldmatrix` pointer that reads 8 contiguous halves picks up the original
   logical 8-element block.
2. *Within any window of 8 consecutive rows, the chunks form a permutation
   of `0..7`.* So an `ldmatrix.x2.trans` asking 16 lanes for rows
   `r, r+1, ..., r+15` all at the same column origin hits each bank-group
   exactly twice — 2-way conflict instead of 16-way.

**Application.** We swizzle on the way *in* (the `cp.async` `issue_tile`
lambda computes `col_sw = swizzle(row_in_tile, col_in_row)` for the
destination) and on the way *out* (every `ldmatrix` and `__half2` read
recomputes the swizzled column for its row pointer).

**Q is also swizzled** for layout uniformity. The kernel-entry Q write goes
out swizzled; the four `ldmatrix.x4` reads compute their swizzled column
once and feed it to the instruction. Per-iter cost is zero.

**Profiler numbers (RTX 4090, single head N=8192, D=64):**

| Counter | 10.5 (unswizzled ldmatrix) | 10.6 (swizzled) |
|---|---|---|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 3,684,352 | **0** |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`    | 4,210,688 | **526,336** |

Bank conflicts go from ~87% of wavefronts to **zero**, and the wavefront
count itself drops 8× because conflict-replays count as separate wavefronts.

**Verification.** Identical `max_abs` to 10.4/10.5 at every N: 3.80e-5 /
1.71e-5 / 1.26e-5 at N ∈ {128, 1024, 2048}. Bit-equivalent output;
`compute-sanitizer memcheck/synccheck/racecheck` clean.

**Bench result.** This is the win the optimization ladder was building toward:

| Rung | N=2048 | N=4096 | N=8192 |
|---|---:|---:|---:|
| 10.4 raw mma.sync         | 13.8 TF/s | 28.7 TF/s | 61.0 TF/s |
| 10.5 +ldmatrix            | 13.8 TF/s | 28.2 TF/s | 59.5 TF/s |
| **10.6 +swizzle**         | **20.6 TF/s** | **42.4 TF/s** | **89.7 TF/s** |

At N=8192 that's **+47% over 10.4** and **+51% over 10.5**, putting us at
~56% of the cuBLAS hgemm peak (~159 TF/s) and *inside* the band where real
production FA-2 / CUTLASS-shape kernels land (80–130 TF/s on this card).
The two stages of the M10.5/10.6 split — instruction count first, then bank
conflicts — were each necessary; together they compound.

**Pedagogical takeaway.** Shared-memory bank conflicts on a tensor-core path
are a *layout* problem, not an *instruction* problem. `ldmatrix` alone gives
you the right SASS; the swizzle gives you the right traffic. CUTLASS's XOR
swizzle is the same shape used everywhere from `Sm80_K64` GEMM kernels to
production FA-2: 3 ALU ops per address computation, zero extra storage, and
it removes an entire class of conflicts deterministically.

## 13. Rung 10.7 — FlashAttention-2-shape kernel

```c
// kernels.cuh, flash_attention_mma_fa2()
// Same tile shape, mma.sync, and swizzled smem as M10.6. Three FA-2 levers
// stacked: STAGES=3 cp.async, K via ldmatrix.x4 (no .trans), and a
// Q-shared-memory overlay that lets us keep STAGES=3 *without* losing the
// 4-blocks-per-SM occupancy M10.6 enjoyed.
```

10.6 landed at **89.7 TF/s** — comfortably inside the production FA-2 band
(80–130 TF/s on RTX 4090) but only 54% of the 165 TF/s compute roofline. The
remaining levers from production-grade FA-2 are: (1) Q register-resident
across K-tile iters (already done in 10.6), (2) deeper cp.async pipeline,
(3) `ldmatrix` for K loads. We stack levers (2) and (3) on a fourth
optimization — a shared-memory overlay — that preserves occupancy.

### 13.1 Lever 1 (already harvested at 10.6): Q register-resident

10.6 hoists the `ldmatrix.x4` Q loads *out of* the main K-tile loop:
`q_frag[K_CHUNKS_S][4]` is computed once at kernel entry and reused across
every K-tile iter. Per-iter Q-load traffic is zero. No code change at 10.7.

### 13.2 Lever 2: `STAGES=3` cp.async

10.6 double-buffers K and V (`STAGES=2`). 10.7 goes to three stages, so the
compute can run two K-tiles behind the loads in flight. The orchestration is
the same shape as `gemm_v1_async` with `STAGES=3` in `08-async-copy`:

- Prologue issues `STAGES-1 = 2` `__pipeline_commit()`s ahead of the loop.
- Main loop's `__pipeline_wait_prior(STAGES-1)` becomes `wait_prior(2)`.
- The new tile issued in iter `j` goes into stage `(j + STAGES - 1) % STAGES`.

### 13.3 Lever 3: `ldmatrix` for K

10.6 loads K's B-fragment with two manual `__half2` reads per lane per
`(kk, n)` mma — 16 mmas × 2 loads = 32 LDS.32 per K tile.

Here's the trick: **K-as-B is naturally col-major**. K's shared layout has
the d-axis fast, and the d-axis is precisely the K-dim of mma's B operand;
mma's `.col` convention means K-dim fast. So K does *not* need the `.trans`
variant of `ldmatrix` (unlike V, where the K-dim of mma — `v_row` — is
slow in storage and `.trans` is required).

That lets us use **`ldmatrix.x4` (no .trans)** to load a 16×16 K sub-tile,
and the resulting A-fragment layout maps directly onto the B-fragments of
*two consecutive* `(kk, n)` and `(kk, n+1)` mmas. With `M = K_tile[n2*16 ..
n2*16+15, kk*16 .. kk*16+15]`:

```
a0 = M[g,   2t..2t+1]   = K_tile[n2*16+g,   kk*16+2t..]   = b0 for (kk, 2n2)
a1 = M[g+8, 2t..2t+1]   = K_tile[n2*16+g+8, kk*16+2t..]   = b0 for (kk, 2n2+1)
a2 = M[g,   2t+8..2t+9] = K_tile[n2*16+g,   kk*16+2t+8..] = b1 for (kk, 2n2)
a3 = M[g+8, 2t+8..2t+9] = K_tile[n2*16+g+8, kk*16+2t+8..] = b1 for (kk, 2n2+1)
```

So 8 `ldmatrix.x4` per K-tile (K_CHUNKS_S=4 × 2 n-pairs each) replaces 32
manual `LDS.32` per lane. Same swizzle pattern as Q/V on the row pointer.

### 13.4 The Q/K/V shared-memory overlay (the unblocker)

Naively, going from `STAGES=2` (24 KB per block: Q=8, K×2=8, V×2=8) to
`STAGES=3` (32 KB: Q=8, K×3=12, V×3=12) costs an occupancy slot — sm_89 has
100 KB shared per SM, so 24 KB→4 blocks but 32 KB→3 blocks. M10.6 ran at 4
blocks/SM; M10.7 at the naive layout would run at 3.

Observation: **Q is dead after the kernel-entry ldmatrix loads.** It's
written into shared, copied into `q_frag` registers, and then never read
again. So we can union Q's 8 KB buffer with stage 0 of K and V (4 KB each =
8 KB total), saving exactly the 8 KB that going from STAGES=2 to STAGES=3
adds for K/V together. Result: 24 KB shared per block, **4 blocks/SM**.

```c
// Q (8 KB) lives at offset 0 of an 8-KB overlay buffer.
// After Q→regs + __syncthreads, the same 8 KB hosts K[0] (offset 0) and
// V[0] (offset BC4*D4). K[1..2] and V[1..2] live in separate 4-KB slots.
static_assert(WARPS_M10_4 * BR4 * D4 == 2 * BC4 * D4, ...);
__shared__ __half KsVs_overlay[2 * BC4 * D4];           // Q | K[0] | V[0]
__shared__ __half Ks_rest[STAGES - 1][BC4 * D4];        // K[1..STAGES-1]
__shared__ __half Vs_rest[STAGES - 1][BC4 * D4];        // V[1..STAGES-1]
```

The block-wide `__syncthreads()` between Q ldmatrix and the cp.async
prologue is now load-bearing — without it, a different warp could start
cp.async into K[0] while another warp's Q ldmatrix from the same bytes is
still in flight.

### 13.5 The `cudaFuncSetAttribute` ritual

On sm_89 the per-block dynamic-shared limit defaults to 48 KB.
`cudaFuncAttributeMaxDynamicSharedMemorySize` raises it (max 99 KB on sm_89).
**On sm_89, that same attribute also gates static shared above 48 KB.** We
set it to 64 KB in `launch_flash_mma_fa2` defensively — our static footprint
is 24 KB so it isn't strictly required at this configuration, but the call
documents the FA-2 pattern (any kernel pushing STAGES higher or BC4 wider
*will* need it).

### 13.6 Numbers

Single head, D=64, RTX 4090, min-of-10 timing. Median across 5 bench runs.

| Rung | N=2048 | N=4096 | N=8192 |
|---|---:|---:|---:|
| 10.6 mma + ldmatrix + swizzle | 21.85 TF/s | 44.74 TF/s | 91.18 TF/s |
| **10.7 FA-2-shape** | **23.30 TF/s** | **48.93 TF/s** | **99.86 TF/s** |
| Δ vs 10.6                     | +6.6% | +9.4% | +9.5% |

At N=8192 that's **~60% of the 165 TF/s compute roofline** and right at the
production FA-2 / CUTLASS-shape midline (80–130 TF/s on this card).
`compute-sanitizer memcheck/synccheck/racecheck` — all clean on the M10.7
kernel (the only racecheck hazards reported are in the unrelated
`naive_softmax` reference). `max_abs` is bit-identical to 10.4/10.5/10.6
(3.80e-5 / 1.71e-5 / 1.26e-5 at N ∈ {128, 1024, 2048}) — all three levers
are pure perf, zero correctness drift.

### 13.7 What's left on the table

The remaining gap to 130 TF/s (the *upper* end of the production FA-2 band)
needs **warp specialization**: split the warps into a producer set that
runs cp.async and a consumer set that runs mma + softmax, with hand-rolled
async barriers between them. That's a structural refactor (different threads
running different code), not another knob on the same loop, so it's
out-of-scope for the 10.x ladder. Hopper's TMA + warp-group MMA makes this
much cleaner; sm_89 can do it manually but the code becomes substantially
more complex. That's where FA-2's `flash_fwd_kernel.h` and CUTLASS's
`Sm90_K64` examples spend their last 20–30%.

### 13.8 Pedagogical takeaway

The three levers individually look small. Stacked, with one structural trick
(the Q overlay) to recover the occupancy that the naive deeper-pipeline
trade would have lost, they compose into the +10% step that lands the kernel
solidly inside the production FA-2 band. The compute roofline isn't a
single bottleneck to fix — it's a *budget* shared across instruction count,
shared-memory traffic, pipeline depth, and occupancy. Production kernels
juggle all four at once; pedagogically, separating them into 10.4 → 10.5 →
10.6 → 10.7 shows you which knobs cost what.

## 14. The four shape variants

Each of these is a thin overlay on the §3 loop. They're all written on top of
the simple "one thread per Q row" base (10.0) because the shape concern is
independent of the speed concern, and because reading them next to the base
is the clearest way to see exactly what's added.

### 14.A Causal masking with tile-skip

For autoregressive (GPT-style) models the score matrix is lower-triangular:
`S[i, j] = -∞ if j > i`. The naive way is to apply the mask per-element inside
the inner loop — every score multiplication still happens, you just clobber
half of them.

The **tile-skip optimization** observes that within a Q row block, the maximum
query row is `qrow_max = blockIdx.x · BR + BR - 1`. Any K/V tile starting at
`j · BC > qrow_max` is **entirely** above the diagonal — every score in that
tile is `-∞`, so the tile contributes nothing to the running `(m, l, O)`
state. Skip it. That bounds the inner loop to `Tc_active = qrow_max / BC + 1`,
roughly halving inner work for large N. Tiles with partial overlap (the
diagonal cuts through them) still need the per-element mask; tiles fully
below the diagonal need no mask at all.

See `flash_attention_causal` in `kernels.cuh`.

### 14.B KV-cache (the inference-time form of FA)

At inference, after the prefill phase the model processes one new token at a
time. Each step:

1. Compute Q, K, V for the new token (`D` floats each per (b, h)).
2. Append the new K, V to a preallocated cache `[B, H, T_max, D]`.
3. Run attention with the new Q vs the *whole* cached K, V up to length T.

This is what `flash_attention_kvcache` does. It's structurally a degenerate FA
where `Tr = 1` (one Q row per (b, h)) — the outer Q-tile loop is gone. The
inner K/V tile loop is the same as MHA.

This is the kernel you'd call ~1000× per generated token in a deployed
transformer, so it's the **single most performance-critical attention kernel
in production inference** — more so than the prefill or training kernels in
total wall-clock at scale. Real implementations split it further by the
"paged" attention pattern (vLLM); we're showing the contiguous-cache form for
clarity.

### 14.C Grouped-Query Attention (GQA)

Llama-2/3, Mistral, and most modern open-weight inference models use GQA: the
number of query heads `H_q` is larger than the number of KV heads `H_kv`,
with `g = H_q / H_kv` query heads sharing each KV head. Mapping: query head
`h_q` reads from KV head `h_q / g`.

Why? KV cache memory is the dominant inference cost at long contexts, and KV
heads are the part of the cache that grows with sequence length. Cutting
`H_kv` by 4× cuts cache memory by 4×, with negligible quality loss in
practice.

Algorithmically GQA is identical to MHA — just a different head-index
mapping. See `flash_attention_gqa` in `kernels.cuh`.

### 14.D What about porting these to 10.2 (WMMA)?

Mechanical, not free. The MHA wrapper is one extra index calculation at the
top of the kernel — straightforward. Causal needs the tile-skip *and* a
per-tile mask: at WMMA granularity that means clobbering specific cells of
`Ssm` after `store_matrix_sync` and before the row softmax, which fits
cleanly into the existing dance. KV-cache and GQA are also pure index
remapping. Recommended as the stretch exercise.

## 15. Comparison: naive vs flash, and the ladder

`bench.cu` runs all rungs at `N ∈ {2048, 4096, 8192}, D = 64`:

| Implementation | Approach                                          | Materialized | Kernel launches |
|---|---|---|---|
| naive          | three kernels: `QK^T`, `softmax(S)`, `PV`         | yes (`S`, `P`) | 3 |
| 10.0 flash     | single fused kernel with online softmax, scalar   | no             | 1 |
| 10.1 flash     | same math, 4× rows per block                      | no             | 1 |
| 10.2 flash     | tensor-core inner matmul + softmax-on-fragments   | no             | 1 |
| 10.3 flash     | 10.2 + double-buffered cp.async K/V loads         | no             | 1 |
| 10.4 flash     | raw `mma.sync` + register-resident softmax        | no             | 1 |
| 10.5 flash     | 10.4 + `ldmatrix` fragment loads (Q `.x4`, V `.x2.trans`) | no    | 1 |
| 10.6 flash     | 10.5 + XOR-swizzled smem layout for Q/K/V         | no             | 1 |
| 10.7 flash     | 10.6 + STAGES=3 + K via `ldmatrix.x4` + Q-shared-overlay  | no    | 1 |

At small N (2048) the naive version's S/P matrices fit in L2 (17 MB) and the
wall-clock gap to flash is small. At `N = 8192` (256 MB S/P), naive blows out
of L2 and slows dramatically. flash still does the same O(N²·D) work but its
working-set fits in shared memory regardless of N — that's the production
case.

Within the flash ladder, each rung is a different optimization technique
applied to the *same* algorithm:

- 10.0 → 10.1: **block-size sweep.** Same inner code; just more rows per
  cooperative tile load. This is what M01 §"empirical block size" warns
  against treating as the optimization — it's the *first* thing to try, not
  the answer.
- 10.1 → 10.2: **switch precision + use the right hardware.** Tensor cores
  exist; FA inner matmuls fit them perfectly. ~2.4× over 10.1 at N=8192.
- 10.2 → 10.3: **overlap memory and compute.** The standard async-copy
  pattern from M08, applied on top of WMMA. ~1.27× over 10.2 at N=8192 in
  practice (on the low end of the ~1.3–1.5× projection).
- 10.3 → 10.4: **kill the softmax shared-memory round trip.** Switch from
  opaque WMMA fragments to raw `mma.sync.m16n8k16` whose lane layout is
  documented. Row softmax over S happens in registers via `__shfl_xor_sync`
  in the 4-lane t-group; per-iteration O-rescale by α is now a plain
  scalar multiply on a `float o_frag[8][4]` per lane. ~2.2× over 10.3 at
  N=8192. This is the FA-2-shape kernel.
- 10.4 → 10.5: **`ldmatrix` fragment loads.** Replace the manual `__half2`
  packing for Q (A operand) with `ldmatrix.x4` and the scalar-`__half`
  packing for V (transposed B operand) with `ldmatrix.x2.trans`. The
  instruction count drops (20 LDSMs replace 64 LDS halves) and the SASS
  is the production FA-2 shape — but on this kernel the runtime is
  essentially unchanged. The reason is bank conflicts on the V load: in
  unswizzled row-major V, the 16-lane transposed read hits a single
  bank-column across rows whether the load is `ldmatrix.x2.trans` or
  manual. The full headline win requires `ldmatrix` *plus* a swizzled
  shared-memory layout for K/V. See §11 for the full discussion.
- 10.5 → 10.6: **XOR-swizzled smem.** Permute the column index of each
  16-byte chunk by `chunk ^ (row & 7)` when writing K/V/Q to shared (and
  apply the same permutation on every read). The 16-way bank conflict on
  the V `ldmatrix.x2.trans` collapses to 2-way, and the K `__half2` and
  Q `ldmatrix.x4` paths go to 0-conflict. `ncu` bank-conflict counter:
  3.68M → 0. ~1.5× over 10.5 at N=8192 — the headline FA-2/CUTLASS
  win that 10.5 was missing. See §12.
- 10.6 → 10.7: **FA-2-shape: stack three more levers.** STAGES=3 cp.async
  pipeline (deeper latency hiding), K loaded via `ldmatrix.x4` (no
  `.trans`, since K-as-B is naturally col-major), and a Q-shared-memory
  overlay that reclaims the 4-blocks-per-SM occupancy the deeper pipeline
  would otherwise cost. ~+10% over 10.6 at N=8192, ~60% of the 165 TF/s
  compute roofline, solidly inside the production FA-2 band (80–130 TF/s
  on this card). See §13. The remaining gap to the top of the band needs
  *warp specialization* — explicit producer/consumer warps with hand-rolled
  async barriers — which is structural enough to be out-of-scope here.

---

## Exercises

> Open `starter.cu` and complete the TODOs.

1. **Inner score block.** For each thread (one Q row), compute `S_ij[c] = Σ_d
   q[d] * K_j[c][d] · scale` for `c ∈ [0, BC)`.
2. **Online softmax update.** Given the new score block, derive `m_ij`,
   `l_ij`, then update `(m, l, O)` using the recurrence above.
3. **`P · V` accumulation.** For each output dim `d`, accumulate `Σ_c p[c] *
   V_j[c][d]` into the rescaled output register.

The verification harness compares against a host-side reference attention.
For the multi-head, causal, KV-cache, and GQA variants, see `solution.cu` —
each runs a host-side reference at small sizes (B=2, H=4, N=128, D=64).

### Stretch

- **10.3 with STAGES=3 or 4.** The current 10.3 uses STAGES=2 (shared-memory
  budget). With `cudaFuncSetAttribute(.., cudaFuncAttributeMaxDynamicSharedMemorySize, ..)`
  to opt into the 100 KB-per-block limit on sm_89, STAGES=3 fits and may hide
  more of the K/V load latency. See M08 §3a for the deeper-pipeline pattern.
- **Port the four shape variants to WMMA or mma.sync.** Take `flash_attention_wmma`
  (or `flash_attention_mma`) and add the MHA / causal / KV-cache / GQA overlays
  from §13. Causal at WMMA granularity is the most interesting because the
  per-tile mask now applies to the materialized `Ssm` between
  `store_matrix_sync` and the row-softmax step. At raw `mma.sync` granularity,
  the mask is a per-lane register guard right after Q·Kᵀ — even cleaner.
- **STAGES=3 cp.async for 10.6.** 10.6 still runs the same 2-stage
  pipeline as 10.4/10.5. With `cudaFuncSetAttribute(.., cudaFuncAttribute­
  MaxDynamicSharedMemorySize, ..)` the 100 KB-per-block budget on sm_89
  fits a third K/V tile, which should hide a bit more cp.async latency.
- **Register-resident Q across the outer loop.** 10.6 reloads Q from
  shared once at kernel entry. With a small register-pressure increase
  the four Q fragments could live in registers for the full lifetime
  of the kernel, freeing shared-memory wavefronts and improving
  occupancy.
- **FlashAttention-2 loop swap.** Swap the loop order so the outer loop is
  over the K/V tile and the inner is over the Q rows. The state machine
  becomes per-output-block instead of per-Q-row. ~2× faster than v1 in
  practice, and it fixes a parallelism issue with non-causal attention.
- **Paged KV cache.** Replace the contiguous `[B, H, T_max, D]` cache with a
  block table + page pool (vLLM-style). Same kernel structure; different
  K/V index calculation.
- **Sliding-window attention.** Mistral-style — like causal but with a
  window of the last W tokens. Adds another tile-skip case.

---

## Profiler checklist

```bash
make
ncu --set full ./bench
```

What to look at, by rung:

- **10.0**: SM Busy / FMA pipe utilization should be modest (~20-30%). DRAM
  traffic should be the *flash* number, not the naive number — i.e. about
  N·D + 2·N·D·Tr bytes per row block, not N² for S/P.
- **10.1 vs 10.0**: same DRAM traffic; SM Busy goes up because the warp
  scheduler has 4× more independent rows to pick from. If 10.1 doesn't help,
  you're memory-bound rather than compute-bound — increase D (try 128) and
  re-test.
- **10.2**: now look at **Tensor Core utilization**. Should be ~30-50% on the
  FMA chart (the softmax-on-fragments dance happens on the FP32 / int pipes,
  diluting the average). DRAM throughput should be *lower* than 10.1 because
  inputs are FP16 (half the bytes per element). Watch
  "Smem→Reg" traffic — that's the O-scaling round-trip in step 4 of §8.
- **10.3 vs 10.2**: Tensor Core utilization should be a few points higher
  (memory now overlaps compute, so the SM is less idle waiting on K/V).
  Look at the `Memory Workload Analysis` view — the "Stall (Long Scoreboard)"
  metric, which is the "I'm waiting for a DRAM load" stall, should drop
  visibly. DRAM bandwidth utilization in the bench measured run goes up.
- **10.4 vs 10.3**: Tensor Core utilization should jump sharply (~2×) —
  the WMMA softmax-on-fragments and O-rescale shared-memory round-trips
  are gone, so a much larger fraction of the kernel time is spent issuing
  `mma.sync`. "Smem→Reg" traffic should drop substantially (no more O
  store/load). The "Warp Stall" breakdown should show the dominant stall
  source shift away from `__syncthreads`/`__syncwarp` toward MIO Throttle
  (the per-iter V transposed load on `Vs_cur` is now the next bottleneck).
- **causal**: at large N, "thread instructions executed" should be ~half of
  the non-causal kernel. If it isn't, the tile-skip isn't kicking in.

## Key takeaways

- The online softmax recurrence is the algorithmic centerpiece. You built it
  in M9; FA tiles it. Nothing on the optimization ladder changes the math.
- FlashAttention's win is that it avoids ever writing `S` and `P` to DRAM —
  *not* better asymptotic complexity. It's still O(N²·D) work; the
  bandwidth constant is ~2-3× smaller.
- The optimization ladder (10.0 → 10.1 → 10.2 → 10.3 → 10.4) is the same
  kind of ladder you saw in M06 (GEMM v0 → v6) and M08 (sync → cp.async).
  Same techniques, different inner loop.
- Tensor cores work great inside FA — but the **softmax** between the two
  matmuls is the awkward bit, because it needs row-wise access to the score
  fragment whose layout is opaque under WMMA. Raw `mma.sync` makes this
  fragment layout documented and per-lane addressable; that's what FA-2 and
  CUTLASS use, and it's what 10.4 does — the rung that takes us from 28
  TF/s (10.3) to 62 TF/s on this card.
- The four shape variants (MHA / causal / KV-cache / GQA) are head-index and
  masking overlays on the same inner loop. Causal masking + tile-skip cuts
  inner work nearly in half for autoregressive models. Free.
- Production FA-2/3 add tensor cores, swapped loops, and microarchitecture
  tricks (TMA, WGMMA on Hopper). The core idea is what you've now seen.
