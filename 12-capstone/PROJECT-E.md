# Capstone Project E — Mamba inference megakernel

> **Status:** post-course / TODO. This is a 1-2 week project, larger than
> Projects A or C. It's the most ambitious option in the capstone and
> ties together M05 (scans), M09 (fused epilogues), M10 (state-in-SRAM
> design ethic), and M11 (megakernel + persistent kernels).

## Goal

Implement the inference path of Mamba's selective state-space model as a
**single persistent megakernel** that processes one token's recurrence step
per work-item. Compare per-token latency against M10's flash-attention
causal kernel at sequence lengths *N* ∈ {128, 1024, 8192}.

## Why this project exists

Mamba is the most prominent recent example of a model architecture whose
hot kernel is *not* attention. It replaces the O(N²) attention pattern
with an O(N) parametric scan, and it claims competitive quality on long-
context language modeling. The CUDA-side thesis of the original paper is
"keep the recurrent state in SRAM, never materialize between time steps" —
exactly the design ethic of FlashAttention, applied to a different inner
operation.

This project is interesting because:
- It's a **selective scan** — the M05 scan ladder (Hillis-Steele,
  Blelloch) is the foundation, but the combine operation is parametric
  (matrix composition for diagonal A: `(a₂, b₂) ∘ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)`).
- It fuses the scan with linear projections and an activation, mirroring
  M09's fused epilogues.
- Inference is per-token — constant state per token, no KV cache. Natural
  fit for M11's megakernel and persistent-kernel patterns.
- It's a published architecture with public reference implementations,
  so success is measurable.

## Scope

### Required (MVP)

1. **Discretize Mamba-1's continuous SSM** for diagonal A:
   - Inputs: parameters Δ, A (diagonal), B, C, D; sequence input `u[t]`.
   - Discretized: `Ā = exp(Δ·A)`, `B̄ ≈ Δ·B` (or the official
     zero-order-hold form: `B̄ = (Ā - I) · A⁻¹ · B`).
   - Recurrence: `x[t] = Ā · x[t-1] + B̄ · u[t]`, output `y[t] = C·x[t] + D·u[t]`.

2. **Implement the per-token recurrence step** as a single persistent
   kernel that processes one token at a time, using M11's ring-buffer
   pattern to dispatch tokens.

3. **Wire up the linear projections** (input proj, output proj) as
   additional phases in the megakernel — no kernel boundary between
   recurrence step and projections.

4. **Verify against a reference** Python+PyTorch implementation (the
   official `mamba-ssm` package, or the `state-spaces/mamba` repo's
   selective_scan).

5. **Bench** per-token latency at N ∈ {128, 1024, 8192}, batch size 1
   (single sequence, latency-bound regime). Compare to M10
   `flash_attention_causal` at the same shapes.

### Stretch

6. **Dual-mode kernel** that can switch between training (chunked
   parallel scan) and inference (sequential recurrence) via a work-item
   descriptor. Training uses your M05 Blelloch scan with the parametric
   combine; inference uses the per-token recurrence above.

7. **Multi-head SSM** (Mamba-2 / SSD shape). Heads share the input
   sequence but have independent (Δ, A, B, C). Naturally fits a
   warp-per-head decomposition inside one block.

8. **Selective scan for non-diagonal A** (full Mamba-1). The combine
   becomes 2×2 matrix multiply per step, tiles get bigger, register
   pressure jumps. This is what the official kernel does; it's a real
   exercise in matching the published design.

9. **Quantized inference path** (INT8 or FP8 activations) — Mamba's
   latency story is amplified at low precision. M07 (tensor cores +
   FP8 forward-ref) is the prerequisite.

## Comparison targets

On RTX 4090, single sequence, batch=1, model dim 1024 (typical for
small Mamba-1):

| N | Mamba mega-kernel target | Flash attention causal target | Notes |
|---|---|---|---|
| 128   | < 30 µs / token | ~20 µs / token  | FA wins at very short context |
| 1024  | < 50 µs / token | ~80 µs / token  | Mamba should pull ahead |
| 8192  | < 100 µs / token | ~300 µs / token | Mamba wins big — O(1) state |

Numbers are educational targets, not authoritative. The official
selective-scan kernel from `mamba-ssm` is *the* reference for "what
this should hit" — it uses raw `mma.sync` on Hopper TMA where available
and a tile-based schedule on Ada.

## Design notes (where to start)

### Phase decomposition for the megakernel

```cuda
enum MambaPhase {
    INPUT_PROJ,      // Linear: u' = W_in · u
    SELECTIVE_SCAN,  // x[t] = Ā·x[t-1] + B̄·u'[t], y[t] = C·x[t] + D·u'[t]
    OUTPUT_PROJ,     // Linear: out = W_out · y
    STOP,
};
```

Each token's processing dispatches three work items in order. The
megakernel keeps the recurrent state `x` in registers across phases for
that token, so there's zero global round-trip between INPUT_PROJ and
SELECTIVE_SCAN, or between SELECTIVE_SCAN and OUTPUT_PROJ.

### Cross-token state

The recurrent state `x` for token *t* needs to persist into token *t+1*.
Options:

- **Per-block in shared memory.** Works if one block handles the whole
  sequence. Tightest latency but limits to single-block-per-sequence.
- **Per-block in pinned host memory.** Survives across blocks but
  pays a host round-trip per token — defeats the purpose.
- **Per-block in DRAM with persistence between work items.** The
  megakernel reads the state at INPUT_PROJ entry and writes back at
  OUTPUT_PROJ exit. One read + one write per token.

For the MVP: single-block-per-sequence + state in shared memory.
Stretch: multi-block via DSMEM (Hopper) or DRAM (Ada).

### Don't reinvent the recurrence math

Use the official paper notation (Δ, A, B, C, D); it's standard and
matches every reference implementation. The discretization step (Δ →
Ā, B̄) is an O(D) operation per token — fold it into INPUT_PROJ.

## Reading list

1. **Original Mamba paper:** Gu & Dao, "Mamba: Linear-Time Sequence
   Modeling with Selective State Spaces" (Dec 2023). Section 4 is the
   CUDA design.
2. **Mamba-2 / SSD:** Dao & Gu, "Transformers are SSMs: Generalized
   Models and Efficient Algorithms Through Structured State Space
   Duality" (May 2024). Connects SSMs to attention; informs the
   multi-head stretch.
3. **Reference kernel:** `state-spaces/mamba` on GitHub — the
   `selective_scan_cuda` directory. Read the `.cu` files; the
   structure of their megakernel is a good template.
4. **Flash linear attention** (`fla-org/flash-linear-attention`) —
   covers the broader family of linear-attention / SSM kernels with
   shared design patterns.

## Definition of done

- The MVP passes correctness vs `mamba-ssm`'s `selective_scan_fn` at
  D=1024, N ∈ {128, 1024, 8192}, max_abs_err < 1e-3 (FP32).
- Per-token latency measured at all three N values, recorded in
  `BENCH-RESULTS.md`.
- The megakernel runs on a single Green Context with 8–16 SMs (M11.7.1)
  and the rest of the GPU continues to serve unrelated work without
  visible degradation.
- A paragraph in `README.md` summarizing where the design wins and where
  it loses vs FA.

## Scope warnings

Things this project is **not**:

- A re-implementation of Mamba-2 or SSD. Stick with Mamba-1 diagonal-A
  for the MVP; promote to harder variants only after the MVP works.
- A training kernel. The MVP is inference only. The chunked parallel
  scan for training is a stretch (#6) and is its own multi-day project.
- A multi-GPU / NCCL story. Single-GPU only.
- A model-quality experiment. We're benchmarking the *kernel*, not
  training a model.
