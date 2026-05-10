# Wave 2 follow-ups (deferred decisions and additions)

> **What this file is:** the design-decision log from a substantial overhaul
> pass on this course. Most items have already landed in the relevant module
> READMEs; the genuinely-deferred work (M10.3 cp.async-flash implementation,
> Mamba PROJECT-E, online-softmax shared-header lift) is called out
> explicitly. Useful for understanding *why* parts of the curriculum are
> structured the way they are; not necessary reading for someone taking the
> course.

---

## A. Module-structure decisions

### A.1 No renumbering — use subsection ladders for fat modules

Several modules grew during Wave 1 (M11 especially: events + streams + graphs
+ persistent + ring + plus Wave 2 additions below). The temptation was to
split them into M11a/M11b/M11c. **Decision: don't.** Renumbering invalidates
every cross-reference in every other module's README and the top-level table.

Instead, follow M06's `6.0 → 6.6` precedent: a module can have explicit
sub-version sections that read like mini-modules but live under one number.

**M11 subsection structure** (target after Wave 2):
- M11.0 — Streams, events, pinned memory, async copies (the building blocks)
- M11.1 — Reducing launch overhead: CUDA Graphs (capture + replay)
- M11.2 — Eliminating per-item launches: persistent kernels, doorbell, ring buffer
- M11.3 — Eliminating launches across the pipeline: megakernel pattern
- M11.4 — Hardware partitioning: Green Contexts, %smid hack, stream priorities
- M11.5 — Beyond Ada: GPUDirect RDMA, DSMEM, MIG (one-paragraph each)

**M09 subsection structure** (already 9 kernels after A4):
- M09.0 — Reduction-pattern epilogues: softmax, online softmax, LayerNorm (Welford), LN+residual
- M09.1 — GEMM-fused epilogues: GEMM+bias+GELU (post-pass v0, in-loop v1)
- M09.2 — Mixed-precision: FP16 LayerNorm

### A.2 The framing: "eliminate launches" is the climax, not "reduce overhead"

M11's README (A5's draft) is currently structured as "ways to reduce
per-launch cost." That undersells it. The real ladder is:

1. **Reduce per-launch overhead** — CUDA Graphs amortize.
2. **Eliminate launches per work-item** — persistent kernels do.
3. **Eliminate launches across the entire pipeline** — megakernels do.

(Static SM partitioning is orthogonal — it's about *who runs where*, not *when
launches happen*. It composes with all three.)

Wave 2: rewrite M11 §1 (intro) and §goals around this ladder.

---

## B. Topics to add to M11

### B.1 Megakernel pattern
A single kernel that switches "phases" internally based on a work-item
descriptor (read from a ring buffer or a fixed schedule). What would have
been `kernel_a → kernel_b → kernel_c` becomes one persistent kernel doing
all three in turn, dispatched from inside the kernel.

Trade-offs:
- **Pro**: zero kernel-boundary cost, no scheduler round-trip, predictable scheduling, can hold cross-phase state in registers/shared.
- **Con**: huge register pressure (sized for the worst phase), poor utilization if phases are unbalanced, instruction cache pressure, harder to compose with libraries (no cuBLAS megakernel), ugly code.

**Implementation sketch** for M11.3:
```cuda
__global__ void megakernel(WorkQueue* wq, ...) {
    while (true) {
        WorkItem w = wq->dequeue();
        if (w.type == STOP) break;
        switch (w.type) {
            case GEMM:        gemm_phase(w.args);    break;
            case SOFTMAX:     softmax_phase(w.args); break;
            case GEMM_OUT:    gemm_out_phase(w.args); break;
        }
        __threadfence_system();
    }
}
```

Companion exercise: take Project B's three-kernel inference pipeline (GEMM →
softmax → GEMM) and rewrite as one megakernel. Measure: what does it cost to
size shared memory and registers for the worst-case phase?

### B.2 SM partitioning — CUDA Green Contexts (CUDA 12.4+, Ada-supported)

The modern, real answer:

```cuda
CUgreenCtx green;
CUdevResource resource;
cuDeviceGetDevResource(dev, &resource, CU_DEV_RESOURCE_TYPE_SM);
// split SMs (e.g., 8 of 128 for low-latency, 120 for batch)
CUdevResource lowlat, batch;
cuDevSmResourceSplitByCount(&lowlat, &count, &resource, &batch, ...);
cuGreenCtxCreate(&green, lowlat, dev, CU_GREEN_CTX_DEFAULT_STREAM);
// kernels launched into `green`'s primary stream only run on the low-lat SMs
```

Use case: dedicate 8 SMs to a persistent low-latency kernel responding to
market events, leave 120 for batch ML inference. They never compete for warp
slots. Fits the HFT thread.

Companion exercise: port the M11.2 ring-buffer demo into a Green Context
allocated only 8 SMs; verify with `%smid` reads inside the kernel that only
those 8 SMs ever serve work items.

### B.3 SM partitioning — `%smid` hack (educational warm-up before Green Contexts)

```cuda
__device__ uint smid() {
    uint id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}
__global__ void filter_by_smid(uint mask) {
    if (((1u << smid()) & mask) == 0) return;  // exit early
    // real work
}
```

Educational: shows what Green Contexts do at hardware level. Wasteful: you
launch enough blocks to cover all SMs, most exit immediately. Useful as a
backward-compat fallback or a teaching beat.

### B.4 Stream priorities
`cudaStreamCreateWithPriority` — preemption-style priority, not partitioning.
Higher-priority work can preempt lower-priority. Useful adjunct for low
latency. One short subsection in M11.4.

### B.5 GPUDirect RDMA — explainer (no code)
Already in A5's draft. Strengthen the ladder framing: "doorbell + persistent
+ Green Context is the local CPU↔GPU low-latency story; GPUDirect RDMA is
how the NIC bypasses CPU entirely. The NIC writes directly into the same
host-pinned ring buffer the persistent kernel polls."

### B.6 DSMEM forward-ref (Hopper-only)
One paragraph in M11.5. Distributed Shared Memory across blocks within a
thread-block cluster (sm_90+, H100 et seq.). Lets blocks read/write each
other's `__shared__` over the SM-to-SM network — useful for cross-block
matmul tile cooperation, distributed softmax/allreduce within a cluster.
*Not* a per-launch-latency primitive in the M11 sense; it's a parallelism /
data-locality feature. Mention as "what evolves out of M11 patterns on Hopper."

---

## C. Topics to add to M07 (Tensor cores) — for A3 to consume

### C.1 Hopper successor §
A3's M07 README should have a small "Beyond Ada" section noting:
- Hopper adds Thread Block Clusters + DSMEM (cross-block shared memory)
- Hopper adds TMA (`cp.async.bulk.tensor`) — async tensor loads with descriptors
- Hopper's 4th-gen tensor cores add FP8 support (already there on Ada too)

### C.2 WMMA fragment-layout caveat
A7 flagged: the m16n16k16 D-fragment layout in
`viz/wmma-fragment-layout.html` is rendered as a 2× tile of the documented
m16n8k16 layout. `wmma::fragment` is officially opaque, so the visualization
is *illustrative*, not authoritative. M07 README should acknowledge this in
the section that links to the viz.

---

## D. Mamba — TODOs across modules (research/post-course)

Mamba (selective state-space models) is a natural extension of the M05 scan
ladder + M09 fused-epilogue + M10 flash thread. Decision: **TODO sections
only — do not implement in this pass.**
Mamba's hot-path is exactly the curriculum's gravitational center
(parametric scan + SRAM-resident state + per-token recurrence), so the
hooks fit cleanly even without code.

### D.1 M05 stretch §scan — selective scan
After Hillis-Steele/Blelloch land, add stretch:
> "Mamba's selective scan replaces the scalar `+` combine with a parametric
> affine combine `(A_t, B_t · u_t)` where the recurrence is
> `x_t = A_t · x_{t-1} + B_t · u_t`. For diagonal A, the combine is
> element-wise: `(a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)`. Implement single-warp
> first via shuffle, then block-level using your Blelloch scaffold. Discretize
> Δ via `Ā = exp(Δ·A)`, `B̄ ≈ Δ·B`."
Estimated: 4-6 hours after Blelloch is solid.

### D.2 M09 stretch §fused — full Mamba block
Stretch problem: fuse selective-scan + linear projections + activation into
one kernel, layered on the LN+residual pattern from A4's work.
Estimated: 1-2 days.

### D.3 M10 §1 callback
One paragraph in M10's intro:
> "Attention scales O(N²) compute, O(N) state per layer. State-space models
> like Mamba scale O(N) compute, O(1) state per token. Different shape, same
> 'keep state in SRAM' design ethic. See M12 Project E."

### D.4 M12 Project E — Mamba inference megakernel (TODO file)
Create `12-capstone/PROJECT-E.md` with:
- Goal: selective scan + projections as a single persistent kernel processing
  one token's recurrence step.
- Target: per-token latency comparison vs M10 flash-attention causal at
  N ∈ {128, 1024, 8192}.
- Stretch: dual-mode kernel switching between training (chunked parallel
  scan) and inference (sequential recurrence) via work-item descriptor.
- Citations: Mamba paper, Mamba-2 / SSD, the official selective-scan kernel.
- Honest scope: 1-2 weeks; post-course.

### D.5 M05 §6 forward-ref to Mamba
Already planned to forward-ref M9; add Mamba pointer too.

---

## E. Coordination cleanup from Wave 1

### E.1 Online-softmax shared-header lift
A4 wrote the recurrence inline in M09 (because A2 hadn't published the M05
primitives yet when A4 needed them). Once A2 is done:
- Lift the recurrence to a header (suggested: `common/online_softmax.cuh` or
  `05-reductions/online_softmax.cuh`).
- Replace the inline copies in M09's softmax_online and the M10 flash
  variants with `#include`-and-call.
- Mechanical change; A4 documented the structure for exactly this.

### E.2 M03 ↔ M07 swizzling cross-ref
A1 will write M03's swizzling forward-ref pointing to "Module 7 §3 'Swizzled
shared memory'". A3 must create that exact anchor. Coherence pass verifies.

### E.3 M04 profiler-counters table cited downstream
A1 will create the table. A2-A5 should cite it from their profiler
checklists. If they don't (because they finished before A1), coherence pass
adds the citations.

### E.4 M13 anchors cited from M07/M08/M11
A6 publishes anchors like:
- `## __threadfence and PTX memory ordering` (cited from M11)
- `## mma.sync vs WMMA` (cited from M07)
- `## cp.async PTX form` (cited from M08)
A5 already placed forward-refs assuming these names. Coherence pass verifies.

### E.5 M06 v6 ↔ M07 evolution
A2 publishes final v6; A3 wraps M07's WMMA kernel around v6's launcher
(replace inner FMA loop with WMMA fragments). A3 cannot start until A2
lands. Coherence pass verifies M07 actually evolves from v6 (same BM/BN/BK
template params, same warp tiling, just different inner loop).

---

## F. Bench / verification items for Wave 2

When I (the orchestrator) do the perf bench sweep:

1. **Re-measure M11 doorbell-RTT cleanly** (idle GPU, no concurrent agents).
   A5's measurement was 1.8-5 µs; expect the low end (~2 µs) on a clean run.
2. **M06 v6 perf check** — must hit ≥75% of cuBLAS at M=N=K=4096 (target 80%).
3. **M07 WMMA perf check** — must hit ≥85% of cuBLAS hgemm at M=N=K=4096.
4. **M13 cache_hints** — verify three *distinct* `ld.global.{ca,cg,nc}.f32`
   appear in the regenerated PTX.
5. **All bench targets** — run, capture in `BENCH-RESULTS.md` with:
   columns: kernel, time (ms min over 10), GFLOP/s or GB/s, % of peak, % of
   reference (cuBLAS where applicable).
6. **Build the whole tree** at `make` from repo root.
7. **`compute-sanitizer --tool=memcheck ./solution`** clean for every module.
8. **Visual smoke test** — open each `viz/*.html`, confirm controls render and
   update.

---

## G. Items emerging late that didn't make the original plan

- **DSMEM forward-ref** (M07, M11) — emerged late in review (Hopper-only,
  but worth signposting).
- **Megakernel section** (M11.3) — emerged from a reframing of the M11
  ladder around "eliminate launches" rather than "reduce per-launch cost."
- **Green Contexts SM partitioning** (M11.4) — emerged late in review; new
  in CUDA 12.4 and Ada-supported, so this is current and worth covering.
- **Mamba TODOs** — emerged late in review as a natural extension of M5/M9/M10.
- **Module-structure decision** to use subsections instead of renumbering —
  emerged from M11/M09 outgrowing original sizes.
