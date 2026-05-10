# Module 9 — Fused epilogues

**Goal:** by the end of this module you should be able to (a) explain *why* fusion
beats stage-by-stage processing for memory-bound kernels, (b) write a numerically
stable softmax in one CUDA kernel, (c) write a single-pass LayerNorm (both the
fast `E[x²]-μ²` form and Welford's), and (d) fuse element-wise epilogues
(bias + activation + residual) onto the back of a GEMM so the result lands in
DRAM exactly once, post-processed.

This module is short on new mechanics (you've already seen reductions and shared
memory) and long on the *pattern*: every time you find yourself writing two
kernels that read the same data, ask whether they can be one kernel.

---

## 1. Why fusion?

Most ML kernels — softmax, layernorm, GELU+bias, dropout, residual+norm — are
**memory-bound**: a tiny amount of arithmetic per byte. For these the runtime is
nearly proportional to bytes moved through DRAM. Reducing those bytes is the main
optimization knob.

A naive softmax pipeline looks like:

```
[ read x → write x_minus_max ]   kernel 1
[ read x_minus_max → write exp ] kernel 2
[ read exp → write sum-vector  ] kernel 3 (reduction)
[ read exp + sum → write y     ] kernel 4
```

Six DRAM passes (`read x`, `write x_minus_max`, `read x_minus_max`, ...). A fused
kernel folds them into one launch and as few DRAM passes as the algorithm permits:

```
[ read x → compute max → compute sum → write y ]   one kernel, 2 reads + 1 write
```

That's a 2× bandwidth reduction, plus four kernel launches saved (each ~5 µs from
Module 1, so meaningful at small batch sizes).

## 2. Numerically stable softmax

Direct `exp(x_i) / sum(exp(x_j))` overflows for any reasonably large input. The
standard trick is to subtract the row max:

```
y_i = exp(x_i - m) / sum_j exp(x_j - m)        where m = max_j x_j
```

This is exact (mathematically) and never produces an exp argument larger than 0,
so it's safe in any precision.

Three passes through the input row:

```
1. find m = max(x)               (block reduction → max)
2. compute s = sum(exp(x - m))   (block reduction → sum)
3. write y_i = exp(x_i - m) / s
```

Each pass is one block walking a row. Reductions across the block use the
`block_reduce_*` helpers from Module 5 (copied into `kernels.cuh`).

### Online softmax (required)

Three passes touch the input row three times. **Online softmax** collapses the
first two into a single pass via a running max + rescaled running sum:

```
m_new = max(m, x)
l_new = l · exp(m - m_new) + exp(x - m_new)
m = m_new;  l = l_new
```

The same recurrence powers FlashAttention in Module 10 — it's the single most
important streaming-statistics pattern in modern GPU kernels. `softmax_online`
in `kernels.cuh` walks `x` once per thread, then combines per-thread `(m, l)`
pairs across the block by (a) reducing `m` to a global row max and (b) rescaling
each thread's local `l` by `exp(m_local - row_max)` and summing.

### Shared-buffer reuse hazard

`block_reduce_sum` and `block_reduce_max` both stash their output in `smem[0]`
and exit through `__syncthreads()`. The same `smem[N_WARPS]` scratch buffer can
safely be reused across back-to-back reductions (sum then sum-of-squares; max
then sum) **only because** the trailing `__syncthreads()` inside each helper
guarantees every thread has read the previous result before the next call's
warp leaders write `smem[lane]`. If you ever inline the reduction and forget
that final sync, you get a silent race on the warp scratch slots.

This is exactly the kind of hazard ncu's `compute-sanitizer --tool=racecheck`
catches.

## 3. LayerNorm

```
y_i = γ_i · (x_i - μ) / σ + β_i
```

with `μ = E[x]`, `σ² = E[x²] - μ²`. The clean trick is to compute `μ` and `σ`
**in a single pass** by maintaining `sum(x)` and `sum(x²)` simultaneously. Two
reductions, but only one read of the input row. After the reductions, every
thread knows `μ` and `1/σ` and writes its piece of the output.

```cpp
float s = 0, sq = 0;
for (int i = tid; i < cols; i += BLK) { s += x[i]; sq += x[i] * x[i]; }
s  = block_reduce_sum(s);
sq = block_reduce_sum(sq);

float mean = s / cols;
float var  = sq / cols - mean * mean;
float rstd = rsqrtf(var + eps);

for (int i = tid; i < cols; i += BLK)
    y[i] = γ[i] * (x[i] - mean) * rstd + β[i];
```

Compare against the "unfused" version which would launch separate kernels for
`mean`, `var`, `normalize`, `affine` — four launches and four DRAM passes for the
same answer.

### Welford's algorithm (required)

The `E[x²] - μ²` form catastrophically loses precision for very large rows
when the mean is large compared to the variance — a classic textbook example.
**Welford's online variance** maintains a running `(count, mean, M2)` triple:

```
n_new   = n + 1
δ       = x - mean
mean   += δ / n_new
M2     += δ · (x - mean_new)
```

After one pass, `var = M2 / n`. Welford generalizes to combining two partial
triples (mean₁, M2₁, n₁) + (mean₂, M2₂, n₂) — that combine rule is what lets us
do Welford in parallel via warp shuffles. See `layernorm_welford` in
`kernels.cuh`.

For the typical hidden dimensions in a transformer (≤8K, modest mean), the
naive form is fine. You should still know Welford because (a) it's the same
shape as online softmax — running statistic via associative combine — and (b)
it's the right answer for any sequence-length-dependent statistic where rows
can grow unbounded.

### LayerNorm + residual-add (required)

A textbook transformer block is

```
y = LN(x + sublayer(x))
```

so the LayerNorm input is `x + residual`. Fusing the residual add saves one
read+write of an `(rows × cols)` tensor — at typical dimensions
(4096 × 4096 fp32 = 64 MB), that's a 25-30% wall-clock win on a 1 TB/s GPU just
from removing one round-trip. `layernorm_residual_add_v0` in `kernels.cuh`
takes (x, residual), writes both the post-add `sum_out` (becomes the residual
for the *next* sublayer) and the post-norm `y_out` (fed into the sublayer).

## 4. GEMM epilogues

The other massive fusion lever is **GEMM epilogues** — folding bias addition,
activation functions, residual adds, and even output dtype conversion into the
GEMM kernel itself, so the result lands in DRAM exactly once, post-processed.

Two implementations side by side, both built on M06's v6 warp-tiled GEMM:

- `gemm_bias_gelu_v0`: standard GEMM (writes raw `C` to DRAM) followed by a
  separate `bias_gelu_epilogue` kernel that reads `C`, adds bias, applies GELU,
  writes back. Two kernels, two DRAM round-trips for the GEMM output.
- `gemm_bias_gelu_v1`: **fused.** After the MMA inner loop, before the float4
  store, each thread loads its `TN` bias values into registers, computes
  `gelu(c + bias)`, and stores the activated result. The GEMM result *never
  visits DRAM in raw form*. One kernel, one DRAM write.

The activation here is the tanh-approximation GELU (matches PyTorch's
`gelu(approximate='tanh')`):

```
gelu(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

This is exactly the technique behind cuBLASLt's epilogue API and torch's
`_efficient_addmm`. For a 4096×4096 GEMM the unfused version moves an extra
64 MB through DRAM versus the fused version — small relative to the K-summed
matmul traffic, but free, and it scales with the post-pass complexity (matters
much more if you stack `bias → gelu → dropout → cast-to-fp16`).

## 5. FP16 / BF16 variants

LayerNorm and softmax are bandwidth-bound, so halving the input/output
precision halves the runtime almost exactly. `layernorm_fused_fp16` in
`kernels.cuh` takes `__half` inputs and outputs but keeps the reductions in
FP32 — the standard "store small, compute wide" pattern.

This is *not* the same as Tensor-Core FP16 GEMM (Module 7); LayerNorm has no
matmul, so there's nothing for the tensor cores to do. It's just narrower
loads and stores.

---

## Exercises

> Open `starter.cu` and complete the TODOs.

1. **Stable softmax (3-pass).** One block per row. Three passes: max, sum,
   normalize. Use `block_reduce_max` and `block_reduce_sum`.
2. **Fused layernorm.** One block per row. Single pass to compute `sum(x)` and
   `sum(x²)` simultaneously, then a second pass to write `y = γ·(x-μ)·rstd + β`.

The bench compares both against unfused versions implemented as separate kernel
launches.

### Stretch

- **`__half` softmax.** The included FP16 LayerNorm shows the pattern; do the
  same for softmax.
- **GEMM + bias + GELU + residual.** Add a residual add to the v1 epilogue.
  Fits in registers; no shared-memory cost.
- **FlashAttention-2 inner softmax.** Re-derive the online softmax recurrence
  from scratch — Module 10 will lean on it heavily.
- **Mamba block fusion (deferred / TODO).** Fuse selective scan + linear
  projections + activation into one kernel, layered on the LN+residual
  pattern from this module. Builds on M05 §5.3's parametric scan stretch;
  ties into M12 Project E. Estimated: 1–2 days after the M05 stretch lands.

---

## Profiler checklist

```bash
make
ncu --set full ./bench
```

Look at:

- **DRAM throughput** — fused versions should be near peak; unfused versions
  split the work across 3-4 launches, each independently bandwidth-bound but
  accumulating more total bytes.
- **Kernel launches in `nsys`** — nsys timeline shows fewer kernels = fewer
  launch-overhead gaps.
- **Stall reasons** — for these kernels the dominant stall should be "Long
  Scoreboard" (DRAM); if you see "Wait" (`__syncthreads`), it's because the
  block reduction is dominating, which usually means the row is short relative
  to block size.
- **Race detection** — `compute-sanitizer --tool=racecheck ./bench` should be
  clean. If you ever delete the trailing `__syncthreads()` in the block
  reductions, this is what catches it.

## Key takeaways

- Fusion is the simplest massive win for memory-bound kernels: combine kernels
  that read the same data.
- Numerically stable softmax: subtract the row max before exponentiating.
- Online softmax (running max + rescaled running sum) collapses two passes into
  one and is the algorithmic core of FlashAttention.
- Single-pass mean/variance: track `sum(x)` and `sum(x²)`; or use Welford for
  numerical stability at very large rows.
- LayerNorm + residual-add is a cheap, important fusion in every transformer
  block.
- GEMM epilogues fuse bias / activation / cast onto the back of a matmul; the
  raw GEMM result never visits DRAM. cuBLASLt's API is exactly this.
