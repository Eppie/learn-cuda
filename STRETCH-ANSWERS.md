# Stretch-exercise hints

One-paragraph hints for the "Stretch" sections across modules. The point is
to give you enough to unstick yourself, not to short-circuit the exercise.
If a module isn't listed here, it doesn't currently have a "Stretch" section
worth hinting at (M04 has two main exercises rather than stretches; M12's
projects A–E are themselves the stretches).

When a hint says "see the module README," the relevant section is the same
one the exercise pointed at — re-read it; the answer is usually right there.

---

## Module 01 — Execution model

### Stretch #4 — saxpy bandwidth
Already hinted in the module README. Same DRAM traffic per element as vector
add (read 2 floats + write 1 = 12 B/elem). The multiply-add is hidden in
DRAM-bound territory and the GPU folds `a*x + b` into one FFMA, so the
instruction-issue cost barely changes either. Expect bandwidth identical to
plain vector add. If saxpy reports slightly higher throughput, you're seeing
FFMA-vs-FADD instruction-issue headroom, not extra compute being "free".

### Stretch #5 — block size 48
Already hinted in the module README. Warps are 32 threads and indivisible.
Block size 48 = 2 warps with the second warp half-masked, so 33% of the
issued lanes never do useful work. Compounds with occupancy: 48 threads/block
× N blocks/SM gives a different resident-warp count than 64 threads/block ×
N blocks/SM. Measure both and look at
`smsp__warps_active.avg.pct_of_peak_sustained_active`.

---

## Module 02 — Memory coalescing

### Stretch #4 — `float4` with the strided pattern
Vectorization helps when the underlying access pattern is *almost* coalesced
and you're paying for instruction-issue overhead. Strided access has the
opposite problem — each lane's load lands in a different cache line, so
asking for 16 bytes per lane multiplies the number of distinct lines touched
by 4. You can fetch 4× more *useful* data per warp, but you're also
fetching 4× as many sectors total, so efficiency stays roughly the same and
absolute throughput rises only because instruction count drops. Profile
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request` — it's still
~32 sectors/request, just issued 4× less often.

### Stretch #5 — L2 effects
This is now in the main bench, not stretch. Run `./bench` and confirm
`copy_scalar` reports markedly higher bandwidth at `N=1M` (4 MB working
set, fully L2-resident) than at `N=64M` (256 MB, DRAM-bound). The crossover
sits around 18 MB — that's roughly 72 MB L2 / 4, since each iteration does
read + write and the previous launch's writes plus this launch's reads need
to coexist in cache.

---

## Module 03 — Shared memory & tiling

### Stretch #7 — count bank conflicts on the unpadded transpose
Already hinted in the module README. Run
`ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./bench`
on the unpadded `transpose_shared` kernel — expect a substantial non-zero
count (the read-back column access is 32-way conflicting). On
`transpose_shared_padded` it should be ~0. The throughput gap between the
two is *exactly* the cost of those serializations. The padded version's
bandwidth approaches the M02 `copy_scalar` ceiling because there's no
remaining shared-memory contention.

---

## Module 04 — Profiling

M04's exercises are the canonical "diagnose me" puzzle (single bottleneck)
and the multi-bottleneck kernel — both required, no stretch section. The
"lab using earlier modules" section asks you to re-profile M02 and M03
binaries and confirm the numbers match the explanations; do that if you
want extra practice without a puzzle.

---

## Module 05 — Reductions, scans, warp shuffles

### Stretch #10 — atomicAdd partial sums into a single output
Replace the second-kernel finish with `atomicAdd(out, blockResult)` from
the first kernel. Easy to write; performance is dominated by contention.
With `gridDim.x = 256` the contention is mild and the simpler code is
often a win; with `gridDim.x = 65536` the atomics serialize and you lose.
Profile `l2_request_lookups`-style counters, or just compare wall-clock
against the two-kernel pattern at varying grid sizes. Fine for ≤256 blocks,
not above.

### Stretch #11 — max reduction
Same skeleton as sum, two changes: the operator (`fmaxf` instead of `+`)
and the identity element (`-INFINITY` instead of `0.0f`). Watch the
out-of-range padding — initializing those slots to `0` will silently
produce wrong answers; use `-INFINITY`. The warp-shuffle helper becomes
`v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset))`.

### Stretch #12 — grid-level reduction with `cooperative_groups::this_grid().sync()`
The kernel structure is in the module README §8. The trap is the launch
side: you must use `cudaLaunchCooperativeKernel`, not `<<<...>>>`. Also the
grid has to fit on the device simultaneously (RTX 4090 caps at ~2048
cooperative blocks given launch resources), and you need
`cudaDeviceProp.cooperativeLaunch == 1` (true on Pascal+, but not under WSL2
in some setups — guard the test). Compared to the two-kernel pattern, the
cooperative version saves one launch but pays for the grid sync.

### Stretch #13 — block-level Hillis-Steele scan
Build it on top of your warp-scan helper. Each warp scans its 32 elements;
warp leader writes the warp's prefix-total to shared memory; first warp
scans those 32 totals; every warp adds the appropriate prefix-total back
to its lanes. Two `__syncthreads`. Compare against your Blelloch from
exercise 8: HS will be ~10–20% faster at BLK=256 because the constant
factor is smaller, and Blelloch's work-optimality only matters for much
larger block sizes.

### Stretch (deferred) — Mamba selective scan
This is a multi-hour project and the answer isn't a hint. See the inline
note in the module README §5.3 plus
[`12-capstone/PROJECT-E.md`](12-capstone/PROJECT-E.md) for the full project
spec. Short version: swap the Blelloch combine from `+` to the parametric
affine combine `(a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)`; reuse the same
up-sweep / down-sweep machinery; discretize Δ via `Ā = exp(Δ·A)`,
`B̄ ≈ Δ·B`. Reference implementation: `state-spaces/mamba` repo,
`selective_scan_cuda`.

---

## Module 06 — GEMM journey

### Stretch — autotune `BM/BN/BK/WM/WN/WMITER/WNITER/TM/TN`
There's no single best answer; tile sizes interact with shared-memory
budget, register pressure, occupancy, and the specific (M, N, K). Sweep
in a script and watch (a) achieved TFLOPs in the bench, (b) achieved
occupancy in `ncu`, (c) `launch__registers_per_thread`. The bigger the
tile, the more register pressure and the less occupancy — but more ILP
per K-iter. The crossover varies; on RTX 4090 at 4096³, the v6 default
(BM=BN=128, BK=16, TM=4, TN=8, WMITER=WNITER=2) is close to optimal.

### Stretch — non-square shapes (e.g. M=N=4096, K=1024)
Wider tiles help when K is short because you're K-summing fewer iterations
and want each iteration to do more work. v5b often beats v6 at very small
K because v6's sub-tile iteration overhead (loop-counter and per-sub-tile
setup) only pays for itself across many K-iters. Try BK=32 or BK=64 at
small K — the bench is set up to take command-line K, just edit and
re-run.

### Stretch — mixed-precision accumulator (forward-ref to M07)
This is the M07 entry point, not a M06 exercise. M07 §4 covers FP16 inputs
with FP32 accumulators; M07 §6 has the table of (input dtype, acc dtype)
combinations and when each is appropriate.

### Stretch — split-K (forward-ref)
When M and N are small but K is huge (e.g. M=N=512, K=65536), the v6 grid
has too few blocks to fill the GPU. Split K into chunks across multiple
blocks (each block accumulates a partial result over its K-slice), then
reduce the partials in a second kernel. Production cuBLAS does this
automatically when it detects the geometry; for hand-rolled kernels you
launch with a 3D grid `(N/BN, M/BM, K_splits)` and a per-block atomic
add (or a separate reduction kernel) at the end.

### Stretch — warp-spec / specialized warps (forward-ref to M08)
Already covered in M08's `mbarrier` section. Designate 1–2 warps as
loaders doing only `cp.async` + commit; the rest do only MMA. Pair them
through an mbarrier per stage. This is the CUTLASS warp-specialized GEMM
pattern. Don't implement it as a M06 stretch — do it in M08 where the
machinery is already there.

---

## Module 07 — Tensor Cores

### Stretch — swizzled shared memory
Already partially done for you: `gemm_v1_wmma_swizzled` in `kernels.cuh` is
the working swizzled version. The exercise is to compare against your v0:
run `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./bench`
and watch the conflict count drop ~4× on the As loads. The why is in
module §3 (lane→column mapping with stride-32-byte rows landing in only 4
of the 8 bank groups; padding to LDAs = 24 halves visits all 8).

### Stretch — raw `mma.sync` with FP16 accumulator
Read §5 and the m16n8k16 lane-element table. Then read `gemm_v2_mma_sync`
in `kernels.cuh`. Modify the accumulator dtype: change the PTX shape from
`.f32.f16.f16.f32` to `.f16.f16.f16.f16`, halve the C/D fragment register
count (from 4 fp32 to 2 b32-of-half2), and update the load/store of the
accumulator. The rest of the kernel (A/B fragment loading, MMA inner
loop) is unchanged.

### Stretch — FP16 accumulator (WMMA)
Already implemented as `gemm_v0_wmma_fp16acc` in `kernels.cuh`. Read it
and run the bench; expect ~2× speedup over FP32 acc on Ada at the cost
of accumulator precision. Watch the verify pass — at K=4096 the FP16-acc
result is at the edge of the tolerance.

### Stretch — `half2` writes
Switch the FP16-acc path to write FP16 output with `half2` for two-element
vectorized stores. `*reinterpret_cast<half2*>(&C[i]) = *reinterpret_cast<const half2*>(&c[0])`.
Half the store instructions, same total bytes; small win on already-fast
path.

### Stretch — cuBLAS comparison with `CUBLAS_COMPUTE_32F_FAST_16F`
Set the cuBLAS compute type to `CUBLAS_COMPUTE_32F_FAST_16F` (TF32-style)
when calling `cublasSgemmEx` / `cublasGemmEx`. Same FP32 inputs as Module 6,
but Tensor-Core accelerated via TF32. Expect roughly 2× speedup over
Module 6's cuBLAS FP32 baseline; some accuracy bits lost (TF32 has 10
mantissa bits vs 23 for FP32), within typical ML tolerance.

---

## Module 08 — Async copy & pipelining

### Stretch — modern API rewrite
Take your working `__pipeline_memcpy_async` kernel and rewrite it using
`cuda::pipeline` + `cuda::memcpy_async`. The reference is `gemm_v_modern`
in `kernels.cuh`. Inspect the PTX of both with `nvcc --ptx`; they should
emit the same `cp.async.cg.shared.global` instruction. The point of the
stretch is to verify the modern API doesn't add overhead — it's a typed
wrapper, not a heavier abstraction. CUTLASS / FlashAttention build on the
modern form because it composes with `cuda::barrier`.

### Stretch — mbarrier-based stage tracking
Replace `__pipeline_wait_prior` with per-stage `cuda::barrier` arrive/wait;
have one warp produce while the other three consume. Module §5 has the
sketch. On Ada this is a stylistic exercise (it emits the same `cp.async`);
on Hopper the mbarrier path is a TMA prerequisite (TMA writes its
completion phase directly to an mbarrier slot). Useful for understanding
how CUTLASS's warp-specialized GEMM divides labor.

### Stretch — compare against cuBLAS at varying problem sizes
The bench's scaling study (M=N=K ∈ {512, 1024, 2048, 4096, 8192}) makes
this concrete. At small K, cuBLAS pays a startup overhead that shows up as
visible launch-cost; at large K, cuBLAS's pipelined kernel pulls ahead by
~20-30%. Your double-buffered kernel narrows the gap most at the middle
sizes.

### Stretch — profile the difference
`ncu` will show *Issue Active* (warp issued an instruction this cycle)
climbing as you add stages — that's the metric of "compute and load are
overlapping." Specifically watch
`smsp__warps_eligible.avg.per_cycle_active` (eligible warps per cycle):
synchronous version reports a low number (warps stuck on `__syncthreads`
after loads); double-buffered reports higher (some warps doing MMA while
others wait on cp.async).

---

## Module 09 — Fused epilogues

### Stretch — `__half` softmax
The included `layernorm_fused_fp16` shows the pattern for LN; do the same
for softmax. Inputs and outputs as `__half`; reductions in FP32; cast at
the boundary. Pattern is "store small, compute wide." Expect
~2× speedup on a bandwidth-bound kernel because the input/output bytes
halve. Watch numerical stability — `expf(x_minus_max)` evaluated in FP16
underflows for large negative x_minus_max; keep the exp in FP32.

### Stretch — GEMM + bias + GELU + residual
Add a residual add to `gemm_bias_gelu_v1`. The residual is one more
input pointer; in the epilogue, after computing `gelu(c + bias)`, add
the residual element. Fits in registers; no shared-memory cost. The
input bandwidth grows by one read of `[M, N]`, but you save the entire
output round-trip you'd otherwise pay for the standalone residual-add
kernel.

### Stretch — FlashAttention-2 inner softmax (re-derive online softmax)
Re-derive the online softmax recurrence from scratch. Given a stream of
scores arriving in batches `(x_1, …, x_BC)` and a running pair `(m, l)`,
prove that the update
`m' = max(m, max(x_*))`, `l' = l·exp(m - m') + Σ exp(x_i - m')` keeps
`l = Σ exp(x_j - m)` as an invariant. M10 leans on this; understanding
the proof makes the FA-2 loop swap make sense.

### Stretch — Mamba block fusion
Deferred / TODO; see [`12-capstone/PROJECT-E.md`](12-capstone/PROJECT-E.md).
Short version: layer the M05 stretch (parametric scan) underneath the
LN+residual pattern from this module. Estimated 1–2 days. Don't tackle as
a single-session stretch.

---

## Module 10 — FlashAttention

> Note: "FP16 + Tensor Cores in the inner matmul" and "cp.async double-
> buffering on top of WMMA" used to be stretch exercises here. They've
> been promoted to the main ladder as **M10.2** (`flash_attention_wmma`)
> and **M10.3** (`flash_attention_async_wmma`). Read those sections of
> M10's README first if you're targeting Project A's higher tiers in M12.

### Stretch — FlashAttention-2 loop swap
Swap the loop order so the outer loop is over the K/V tile and the inner
is over the Q rows. The state machine becomes per-output-block instead of
per-Q-row. The win is for non-causal kernels (where FA-1's outer-Q loop
underutilizes the GPU), and ~2× faster wall-clock on long sequences.
Reference: the FA-2 paper (Tri Dao 2023). The proof of correctness is the
same online-softmax invariant from M09's stretch.

### Stretch — paged KV cache (vLLM-style)
Replace the contiguous `[B, H, T_max, D]` cache with a block table + page
pool. Same kernel structure; different K/V index calculation: instead of
`K[b, h, t, d]`, look up `block_table[b][t / BLOCK_SIZE]` to get a page
index, then `K_pool[page_idx, t % BLOCK_SIZE, d]`. The win is memory
fragmentation — without paging you preallocate `T_max` per request, with
paging you allocate pages as the sequence grows.

### Stretch — sliding-window attention (Mistral-style)
Like causal but with a window of the last W tokens. Adds another tile-skip
case: a tile is skippable when its column range is entirely outside
`[qrow - W, qrow]`. The implementation is one extra check in the inner
loop's per-tile mask logic. Test on `N = 4096, W = 256`; verify against a
host-side reference.

---

## Module 11 — Low-latency

### Exercise #5 (stretch) — 4-stream H2D / kernel / D2H pipeline
Three concurrent streams: stream 1 issues `cudaMemcpyAsync(H2D)` for the
next batch, stream 2 runs the kernel on the current batch, stream 3
issues `cudaMemcpyAsync(D2H)` for the previous batch. With 3 stages and
N batches, you pay max(H2D, kernel, D2H) per batch instead of the sum.
Use `cudaStreamWaitEvent` to express "stream 2's kernel waits for stream
1's H2D to finish" (M11 §3). Profile with `nsys` to see the timeline
overlap; the screenshot is more convincing than the bench number.

### Exercise #6 (stretch) — GPU-to-GPU producer/consumer ring
Replace the host-CPU producer in `ring_buffer.cu` with a second
persistent producer kernel on a different stream. The ring is now in
device memory (still pinned-equivalent for inter-stream visibility);
both kernels poll their respective `head` / `tail` indices via
`__threadfence` / atomic operations. The building block for inter-kernel
pipelines on one device. Trickier than it looks — the producer-consumer
visibility needs `__threadfence_block` (same SM) or `__threadfence`
(different SMs); cross-stream visibility on Ada requires the slow
`__threadfence_system` because Ada lacks the proper inter-stream fence.
On Hopper, `cuda::atomic_ref` with `memory_scope_device` is cheaper.

### Section 7.5 exercise — Green Context with 16 SMs
Take `ring_buffer.cu` and modify the launch to use a Green Context bound
to 16 SMs. You'll need the driver API (`cu*`); module §7.1 has the full
setup sequence. Verify with an in-kernel `%smid` print (module §7.2 has
the inline asm) that only those 16 SMs ever serve work items. Hint: call
`cuInit(0)` once at startup, then mix `cu*` and `cuda*` freely. Expect
zero performance change vs the regular kernel — Green Contexts are about
*isolation*, not speed; the win shows up when you also run a batch
workload on the other 112 SMs and measure tail-latency on the hot path.

---

## Module 12 — Capstone

The capstone projects A–E *are* the stretches. There aren't sub-stretches
worth hinting at separately:

- **Project A stretches** (whole-tile causal skip, FP16, WMMA inner matmul)
  are listed with their performance tiers in
  [`12-capstone/README.md` § Project A](12-capstone/README.md). Tile-skip
  is the easy 1.6× win (no FP16 conversions, no WMMA setup); start there.
- **Project B** is itself a 15–25 hour stretch over the core path. See the
  scaffold in `12-capstone/README.md`.
- **Project C / D / E** each have their own performance criteria and
  starting-point notes inline. Project E is a multi-week post-course
  project; see [`12-capstone/PROJECT-E.md`](12-capstone/PROJECT-E.md).

---

## Module 13 — PTX appendix

M13 is a reference, not an exercise module. It doesn't have a Stretch
section. Read the relevant sub-section when M07/M08/M11's forward-refs
point you there.
