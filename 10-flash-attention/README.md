# Module 10 — FlashAttention

**Goal:** by the end of this module you should be able to (a) explain the
*online softmax* recurrence and how it tiles, (b) implement a working
FlashAttention forward pass in CUDA at three points along the optimization
ladder — from "one thread per Q row" through warp-cooperative FP32 to a
WMMA tensor-core kernel — and (c) describe the four common shape variants
(MHA, causal, KV-cache, GQA) and the path from there to FlashAttention-2/3.

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
| 10.0   | `flash_attention`               | FP32     | per-thread scalar   | 6.5 TF/s        |
| 10.1   | `flash_attention_warp`          | FP32     | per-thread scalar   | 9.2 TF/s        |
| 10.2   | `flash_attention_wmma`          | FP16     | WMMA tensor-cores   | 22 TF/s         |
| 10.3   | `flash_attention_async_wmma`    | FP16     | WMMA + cp.async ×2  | 28 TF/s         |

(Numbers are min-of-N timing on RTX 4090, single head, D=64; see
`bench.cu`. The cuBLAS hgemm peak on this card is ~159 TF/s, so 10.3 is at
~18% of peak — the softmax-on-fragments dance is now the single biggest
cost left, and raw mma.sync is what closes the rest of the gap.)

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
that raw `mma.sync` would eliminate. (a) and (b) are 10.3.

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
which is the §11 Stretch exercise.

We use the legacy `__pipeline_memcpy_async` API from `<cuda_pipeline.h>`
(emits `cp.async.cg.shared.global ..., 16, 16`), matching M08's
`gemm_v1_async`. The modern `cuda::pipeline` wrapper is functionally
equivalent — see M08 §8.2 for the comparison.

## 10. The four shape variants

Each of these is a thin overlay on the §3 loop. They're all written on top of
the simple "one thread per Q row" base (10.0) because the shape concern is
independent of the speed concern, and because reading them next to the base
is the clearest way to see exactly what's added.

### 10.A Causal masking with tile-skip

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

### 10.B KV-cache (the inference-time form of FA)

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

### 10.C Grouped-Query Attention (GQA)

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

### 10.D What about porting these to 10.2 (WMMA)?

Mechanical, not free. The MHA wrapper is one extra index calculation at the
top of the kernel — straightforward. Causal needs the tile-skip *and* a
per-tile mask: at WMMA granularity that means clobbering specific cells of
`Ssm` after `store_matrix_sync` and before the row softmax, which fits
cleanly into the existing dance. KV-cache and GQA are also pure index
remapping. Recommended as the §11 stretch exercise.

## 11. Comparison: naive vs flash, and the ladder

`bench.cu` runs all rungs at `N ∈ {2048, 4096, 8192}, D = 64`:

| Implementation | Approach                                          | Materialized | Kernel launches |
|---|---|---|---|
| naive          | three kernels: `QK^T`, `softmax(S)`, `PV`         | yes (`S`, `P`) | 3 |
| 10.0 flash     | single fused kernel with online softmax, scalar   | no             | 1 |
| 10.1 flash     | same math, 4× rows per block                      | no             | 1 |
| 10.2 flash     | tensor-core inner matmul + softmax-on-fragments   | no             | 1 |
| 10.3 flash     | 10.2 + double-buffered cp.async K/V loads         | no             | 1 |

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
  practice (on the low end of the ~1.3–1.5× projection; closing the
  remainder to peak now requires raw `mma.sync` to eliminate the O-rescale
  shared-memory round trip — see the §11 Stretch).

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
- **Port the four shape variants to WMMA.** Take `flash_attention_wmma` and
  add the MHA / causal / KV-cache / GQA overlays from §10. Causal at WMMA
  granularity is the most interesting because the per-tile mask now applies
  to the materialized `Ssm` between `store_matrix_sync` and the row-softmax
  step.
- **FlashAttention-2 loop swap.** Swap the loop order so the outer loop is
  over the K/V tile and the inner is over the Q rows. The state machine
  becomes per-output-block instead of per-Q-row. ~2× faster than v1 in
  practice, and it fixes a parallelism issue with non-causal attention.
- **Paged KV cache.** Replace the contiguous `[B, H, T_max, D]` cache with a
  block table + page pool (vLLM-style). Same kernel structure; different
  K/V index calculation.
- **Sliding-window attention.** Mistral-style — like causal but with a
  window of the last W tokens. Adds another tile-skip case.
- **Raw `mma.sync` flash.** Replace the WMMA wrapper with raw `mma.sync.
  m16n8k16` (M07 v2_mma_sync style; M13 mma_sync_example for the lane-
  element layout). The win is that `o_frag` becomes a normal float register
  array, so step 4 of §8 (the O-scaling shared-memory round trip) goes away.
  The cost is ~50 lines of fragment-packing PTX.

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
- **causal**: at large N, "thread instructions executed" should be ~half of
  the non-causal kernel. If it isn't, the tile-skip isn't kicking in.

## Key takeaways

- The online softmax recurrence is the algorithmic centerpiece. You built it
  in M9; FA tiles it. Nothing on the optimization ladder changes the math.
- FlashAttention's win is that it avoids ever writing `S` and `P` to DRAM —
  *not* better asymptotic complexity. It's still O(N²·D) work; the
  bandwidth constant is ~2-3× smaller.
- The optimization ladder (10.0 → 10.1 → 10.2 → 10.3) is the same kind of
  ladder you saw in M06 (GEMM v0 → v6) and M08 (sync → cp.async). Same
  techniques, different inner loop.
- Tensor cores work great inside FA — but the **softmax** between the two
  matmuls is the awkward bit, because it needs row-wise access to the score
  fragment whose layout is opaque under WMMA. Raw `mma.sync` makes this less
  awkward; production kernels use it.
- The four shape variants (MHA / causal / KV-cache / GQA) are head-index and
  masking overlays on the same inner loop. Causal masking + tile-skip cuts
  inner work nearly in half for autoregressive models. Free.
- Production FA-2/3 add tensor cores, swapped loops, and microarchitecture
  tricks (TMA, WGMMA on Hopper). The core idea is what you've now seen.
