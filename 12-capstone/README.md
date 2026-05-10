# Module 12 — Capstone

You've built up the parts. This module is about putting them together into one
end-to-end kernel that resembles something a real ML system would actually call.

There are three capstone projects below; pick the one that fits your interest.
Each builds on existing modules with concrete, scoped extensions. The repo ships
**working starter code for project A**, since it's the most direct continuation of
Module 10 and it's the one most students should do. Project B now ships a
**partial scaffold** — see the dedicated section.

---

## Project A — Multi-head causal FlashAttention (recommended)

**What you build:** extend Module 10's single-head FlashAttention forward to
support (a) batched multi-head attention with shape `[B, H, N, D]`, and (b) causal
masking (no attention to future tokens). This is the actual signature used by
GPT-style transformers.

**Why this:** it's the smallest possible step from "FlashAttention exists in this
repo" to "FlashAttention I could plug into a real GPT". Multi-head is just an outer
loop / grid dim; causal masking is two lines in the inner loop. After this you'll
have the structure of every GPU attention kernel in production, minus Tensor
Cores and async copy.

**Files in this module:**
```
12-capstone/
  README.md         # this file
  Makefile
  kernels.cuh       # working solution: multi-head causal FA
  solution.cu       # verifies vs host attention (small B, H, N)
  starter.cu        # TODO scaffold of the multi-head + causal extensions
  bench.cu          # times the kernel on B=32, H=12, N=2048, D=64
                    #   (a 124M-token forward pass; see TODO-USER for cuDNN ref)
```

**The two changes vs Module 10:**

1. **Batch + head dimensions.** Launch grid is 3D: `(N/BR, H, B)` blocks. Inside,
   add `(b * H + h) * N * D` to all four input/output pointers before doing
   exactly the Module 10 algorithm.
2. **Causal mask.** In the inner-loop score computation, set
   `s[c] = -INFINITY` whenever `kcol > qrow`. The `exp(-INF) = 0` makes those
   entries contribute nothing to the softmax. (Do this *before* computing
   `m_ij = max(s)` so that `m_ij` reflects only the unmasked entries.)

**Stretch within Project A:**

- **Skip whole tiles** when the entire `BC × BR` block is below the diagonal —
  a free 2× speedup on the causal case (the lower triangle is ~half the matrix).
- **FP16 inputs.** Convert Q, K, V, O to `__half`. The kernel structure doesn't
  change much; the inner-loop arithmetic is what gets faster (Module 7).
- **WMMA inside the inner matmuls.** Use `mma_sync` for the `Q · K^T` and
  `P · V` matmuls in the inner loop. This is the real production architecture of
  FlashAttention-1.

### Performance success criteria — RTX 4090 (sm_89)

These targets are calibrated against the RTX 4090's specs (about 82 TFLOPs FP32,
330 TFLOPs FP16 dense tensor-core, 1 TB/s HBM). All numbers assume the bench
shape `B=32, H=12, N=2048, D=64`. Effective FLOPs counted as `4 · B · H · N² · D`
(the standard MHA FLOP count; halve for `causal=true` since half the tile area
is masked).

| Tier | Implementation | Target effective TFLOPs/s | Realism |
|------|---------------|---------------------------|---------|
| **Passing**  | FP32, multi-head + causal, no tile-skip | ≥ 8 TF/s | Baseline; what `kernels.cuh` ships should hit. |
| **Solid**    | FP32 + tile-skip stretch | ≥ 14 TF/s on `causal=true` | ~1.6× speedup over no-skip. |
| **Strong**   | FP16 inputs, FP32 accumulator (no tensor cores) | ≥ 25 TF/s | Half the bandwidth, same arithmetic. Reachable directly by porting M10.2 (`flash_attention_wmma`) to MHA + causal. |
| **Production-shaped** | FP16 + WMMA inner matmul + `cp.async` | ≥ 50 TF/s | Reachable directly by porting M10.3 (`flash_attention_async_wmma`) to MHA + causal — single-head 10.3 hits 26 TF/s, multi-head should match. |
| **Stretch (advanced)** | FP16 + raw `mma.sync` + `cp.async` | ≥ 90 TF/s | Single-head M10.4 (`flash_attention_mma`) hits ~59 TF/s; porting to MHA + causal + adding `ldmatrix.x2.trans` for the V-as-B load should clear 90 TF/s. The remaining gap to cuBLAS+cuDNN is CUTLASS-quality scheduling. |

These are *output* TFLOPs (useful work), not raw HMMA peak. Don't be alarmed if
your `nsys`/`ncu` profile reports a different number for "tensor-core utilization"
— that's the metric for the matmul portion alone, not the whole attention kernel
(which spends real time on softmax + masking).

### Reference comparison

`bench.cu` includes a `TODO-USER` for an apples-to-apples cuBLAS+cuDNN MHA
reference. The full cuDNN frontend MHA descriptor (`cudnnMultiHeadAttnForward`)
is a fairly involved API to set up — backends, weight matrices, KV-cache
layout, fused dropout, etc. — and a faithful comparison is its own ~200-line
exercise.

If you want to wire it up, the docs to start from are:
- cuDNN frontend: `cudnn_frontend::graph::Graph` with `SDPA` operation node.
- The descriptor-API form: `cudnnMultiHeadAttnForward` with weight buffers
  set to identity (so it computes plain attention with no projections).

For a faster sanity comparison, you can also:
1. Time the pieces yourself: `cublasSgemmStridedBatched` for `Q @ K^T` and
   `P @ V` (separate kernels — slower than the fused FA, that's the point).
2. Take the ratio. If your fused FA isn't beating the unfused two-GEMM
   reference by ≥ 2×, the fusion isn't pulling its weight and there's a bug.

---

## Project B — Low-latency inference pipeline

**What you build:** a CUDA-Graph-wrapped pipeline that runs a small transformer
inference step (input projection + GEMM + softmax + output projection) end-to-end
with single-digit microsecond latency per step. You're combining Module 11's
launch-overhead reduction with Module 6/9's GEMM/softmax kernels.

**Why this:** speaks directly to your HFT interest. Real low-latency systems care
not about peak throughput but about per-step overhead — a CUDA Graph that captures
"input arrives → kernels execute → result is in pinned memory" can hit ~10 µs
per step for small models.

**Time estimate: 15–25 hours.** This is a real systems-integration project that
combines five modules' worth of techniques. Budget accordingly.

### Hard dependencies (don't start B until you've completed these)

| Required from | What | Why |
|---------------|------|-----|
| **M06 v6**    | Working warp-tiled GEMM with verify | The two projections are GEMMs |
| **M07 v0** *or* **M10.2** | At least one WMMA kernel working — either `gemm_v0_wmma` (M07) or `flash_attention_wmma` (M10.2). The latter is closer to what Project A needs and is a faster on-ramp to the Strong / Production tiers. | Strong tier (≥25 TF/s) and above need WMMA inside the inner matmuls |
| **M09 fused** | LayerNorm + softmax fused kernels  | Activation between projections |
| **M11 events**       | `cudaEventRecord` / `cudaStreamWaitEvent` | Stream-sync for the pipeline |
| **M11 persistent**   | Working persistent-kernel demo | One option for ultra-low-latency |
| **M11 ring buffer**  | SPSC ring + producer pattern | The "input arrives" half |

If any one of these isn't solid, fix it in its own module first. The capstone is
not the place to debug a broken GEMM.

### What ships in this directory for Project B

There's a partial scaffold below — `inference_pipeline.cu` (you'll create it
from the template in `kernels.cuh`'s commented-out section, see below). The
scaffold wires up the kernel sequence + graph capture, but leaves the
individual kernel bodies as `TODO`s.

```
inference_pipeline.cu (you create from the scaffold)
  ┌─ kernel sequence wired up:
  │    h_in  →  [pinned host -> device async memcpy]
  │           →  [input projection GEMM: M06 v6]
  │           →  [softmax: M09 fused]
  │           →  [output projection GEMM: M06 v6]
  │           →  [pinned device -> host async memcpy]  →  h_out
  ├─ wrap the whole sequence in a CUDA graph (M11)
  └─ measure per-step latency with and without graph capture
```

The scaffold below has the orchestration + graph capture in place. The bodies
are `// TODO: invoke your M06 v6 / M09 softmax kernel here` calls — you fill in
the launch lines.

### Project B scaffold

Save this as `inference_pipeline.cu` in this directory and link against your
own `kernels.cuh` from M06/M09:

<details>
<summary>Click to expand the full scaffold (~120 lines)</summary>

```cpp
// Module 12 Project B — low-latency inference pipeline scaffold.
//
// Pipeline: pinned input → input proj GEMM → softmax → output proj GEMM
//                       → pinned output. Wrap in a CUDA graph; measure.
//
// You provide:
//   - launch_gemm_v6(A, B, C, M, N, K, stream)       from M06
//   - launch_softmax_rows(X, M, N, stream)            from M09
//
// We provide: orchestration, graph capture, latency measurement.

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_utils.h"
// #include "../06-gemm/kernels.cuh"        // for launch_gemm_v6
// #include "../09-fused-epilogues/kernels.cuh"  // for launch_softmax_rows

constexpr int M_BATCH    = 1;        // single inference step
constexpr int D_MODEL    = 512;
constexpr int D_FF       = 2048;     // feedforward intermediate
constexpr int ITERS      = 1000;
constexpr int WARMUP     = 100;

static void run_pipeline(cudaStream_t stream,
                         float* d_in, float* d_w_in, float* d_intermed,
                         float* d_w_out, float* d_out) {
    // TODO 1: launch input projection GEMM.
    //   d_intermed[M x D_FF] = d_in[M x D_MODEL] @ d_w_in[D_MODEL x D_FF]
    //   launch_gemm_v6(d_in, d_w_in, d_intermed,
    //                  M_BATCH, D_FF, D_MODEL, stream);

    // TODO 2: launch fused softmax (or layernorm) over each row of d_intermed.
    //   launch_softmax_rows(d_intermed, M_BATCH, D_FF, stream);

    // TODO 3: launch output projection GEMM.
    //   d_out[M x D_MODEL] = d_intermed[M x D_FF] @ d_w_out[D_FF x D_MODEL]
    //   launch_gemm_v6(d_intermed, d_w_out, d_out,
    //                  M_BATCH, D_MODEL, D_FF, stream);
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("Project B: pinned -> proj1 -> softmax -> proj2 -> pinned\n");
    std::printf("Shape: M=%d  D_MODEL=%d  D_FF=%d\n\n",
                M_BATCH, D_MODEL, D_FF);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- pinned host buffers (M11) ----
    float *h_in_pinned, *h_out_pinned;
    CUDA_CHECK(cudaHostAlloc(&h_in_pinned,  M_BATCH * D_MODEL * sizeof(float),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out_pinned, M_BATCH * D_MODEL * sizeof(float),
                             cudaHostAllocDefault));

    // ---- device tensors ----
    float *d_in, *d_w_in, *d_intermed, *d_w_out, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,       M_BATCH  * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_in,     D_MODEL  * D_FF    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_intermed, M_BATCH  * D_FF    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_out,    D_FF     * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,      M_BATCH  * D_MODEL * sizeof(float)));

    // (Initialize buffers here. Skipping for brevity.)

    // ============================================================
    // Path 1: stream replay (no graph). Each iteration enqueues
    // 5 kernel launches.
    // ============================================================
    auto step_no_graph = [&]() {
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in_pinned,
                                   M_BATCH * D_MODEL * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        run_pipeline(stream, d_in, d_w_in, d_intermed, d_w_out, d_out);
        CUDA_CHECK(cudaMemcpyAsync(h_out_pinned, d_out,
                                   M_BATCH * D_MODEL * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    };

    for (int i = 0; i < WARMUP; ++i) step_no_graph();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < ITERS; ++i) step_no_graph();
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_no_graph = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_no_graph, e0, e1));

    // ============================================================
    // Path 2: capture into a CUDA graph, then replay.
    // ============================================================
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    step_no_graph();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < WARMUP; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_graph = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_graph, e0, e1));

    std::printf("\nResults (us per inference step):\n");
    std::printf("  no-graph  (5 launches/step):  %.2f us\n",
                (double)ms_no_graph * 1000 / ITERS);
    std::printf("  graph     (1 driver call):    %.2f us\n",
                (double)ms_graph    * 1000 / ITERS);
    std::printf("  speedup:                       %.2fx\n",
                (double)ms_no_graph / ms_graph);

    // Cleanup omitted for brevity.
    return 0;
}
```

</details>

### Project B success criteria — RTX 4090

| Step | Target |
|------|--------|
| **Passing** | Pipeline runs end-to-end without errors; per-step graph latency ≤ 25 µs. |
| **Solid**   | Per-step graph latency ≤ 12 µs. |
| **Strong**  | Per-step graph latency ≤ 8 µs and graph speedup over stream-replay ≥ 3×. |
| **Stretch** | Persistent-kernel variant (no `cudaGraphLaunch` per step) at ≤ 5 µs RT. |

The graph speedup over no-graph is the headline number — it's measuring how
much of your latency was launch overhead.

---

## Project C — Strided batched GEMM

**What you build:** extend Module 6 v6 (or v5) to handle `N` independent
small GEMMs of identical shape, given as strided pointers. This is what cuBLAS's
`cublasSgemmStridedBatched` does.

**Why this:** strided batched GEMM shows up everywhere — multi-head Q/K/V
projections, batched linear layers, parallel small models. Single-kernel batching
beats N separate kernel launches dramatically when the matrices are small.

**Where to start:** Module 6 v6, take `B` (batch) as a third grid dimension.
Each block already has `(blockIdx.x, blockIdx.y) → (M, N)` tile; add
`blockIdx.z → batch` and offset all pointers by `b * stride_*`.

**Concrete deliverable:** for `B = 64, M = N = K = 256` (small per-batch),
compare against:
- 64 separate launches of v6.
- A single batched-grid v6.
- `cublasSgemmStridedBatched`.

**Performance criteria** — single-batched-grid v6 should beat 64 separate
launches by ≥ 5× (this is mostly killing launch overhead, like Project B).
Hitting cuBLAS strided-batched is harder; budget aim is ≥ 60% of cuBLAS.

---

## Project D — Megakernel inference pipeline (stretch beyond Project B)

> **Estimated time:** 5–10 hours on top of Project B.
> **Hard dependencies:** Project B working at the "Solid" tier or better.

The megakernel variant of Project B. Take Project B's three-kernel sequence
(GEMM → softmax → GEMM) wrapped in a CUDA Graph, and **collapse it into a
single kernel** that switches phases internally based on a work-item descriptor
(see Module 11 §6 for the megakernel pattern).

The win: cross-phase state in shared memory means no global round-trip
between phases. The cost: shared memory and registers sized for the worst
phase, no library composition (no cuBLAS / cuDNN inside).

**Performance criteria** — RTX 4090, batch=1, hidden=4096:
- Megakernel beats Project B's CUDA-Graph version by ≥ 20% on per-token
  latency at small batch sizes (where launch+boundary cost dominates).
- At batch ≥ 32, expect Project B to win — note this honestly in your
  writeup.

This is the "all the launches eliminated" endpoint of M11's ladder.

---

## Project E — Mamba inference megakernel (post-course)

> **Estimated time:** 1–2 weeks. The most ambitious capstone option.
> **Hard dependencies:** M05 scan ladder, M09 fused epilogues, M11 §6
> megakernel + §7 Green Contexts. M07 (tensor cores) for stretch goals.

Implement the inference path of Mamba's selective state-space model as a
**single persistent megakernel** processing one token's recurrence step per
work-item. Compare per-token latency against M10 flash-attention-causal at
sequence lengths *N* ∈ {128, 1024, 8192}. Pulls together every module in
the course; touches an architecture that's genuinely different from
attention.

**Full project specification:** [`PROJECT-E.md`](PROJECT-E.md). It's a
multi-week project, presented as a post-course milestone rather than an
in-budget capstone option — but the design notes (phase decomposition,
state persistence between tokens, single-Green-Context partitioning) make
it a worthwhile read even if you don't implement it.

---

## Recap of techniques the capstone touches

| module | technique                                      |
|--------|------------------------------------------------|
| 1–3    | execution model, coalescing, shared memory     |
| 4      | profiling discipline                           |
| 5      | warp shuffles, hierarchical reductions         |
| 6      | block tiling → register tiling → vectorization |
| 7      | Tensor Cores via WMMA / `mma.sync`             |
| 8      | `cp.async`, software pipelining                |
| 9      | kernel fusion for memory-bound work            |
| 10     | online softmax, FlashAttention                 |
| 11     | CUDA Graphs, persistent kernels, pinned mem    |
| 13     | PTX appendix — read SASS to verify the win     |

The capstone won't use *all* of these — pick the ones the project demands. The
point of the course was to give you the tools and the intuition for *which* tool
fits *which* problem. The capstone is the test of that.

---

## Where to go after this course

- **CUTLASS / cuTe.** The library NVIDIA uses internally for GEMM-shaped kernels.
  Once you understand Modules 6–8, CUTLASS's tile abstractions become readable.
- **TVM / Triton / Mojo.** Higher-level kernel-authoring DSLs that compile down
  to the same patterns we wrote by hand. Worth understanding the tradeoffs.
- **FlashAttention 2 / 3.** The papers, then the source. With Module 10 in
  your head, both are tractable reads.
- **Hopper-specific (H100).** The Tensor Memory Accelerator (TMA), `wgmma`,
  cluster launch semantics. None of this exists on Ada (`sm_89`); it's only worth
  studying if you'll have an H100 in front of you.
- **GPU systems papers.** Page Migration on UM, GPUDirect RDMA, MPI+CUDA.
  Relevant once you're scaling beyond one GPU.

---

## Closing

The path from module 1 (vector add at 950 GB/s) to module 12 (an attention kernel
that scales to long contexts) is the whole arc. You now understand, from
first principles, what every line of a production GPU kernel is doing and why.

The next step is to write your own.
