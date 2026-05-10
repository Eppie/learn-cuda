# Interactive visualizations

Self-contained HTML+JS files. No build step, no server, no external
dependencies — open them directly in a browser:

```bash
xdg-open viz/thread-memory-map.html       # Linux
open      viz/thread-memory-map.html       # macOS
# or paste the file path into your browser's URL bar
```

Vanilla JS + SVG, ≤ 200 lines per file. Shared helpers (sliders, color
palettes, etc.) live in `lib.js`; every viz includes it via
`<script src="lib.js"></script>`.

## Index

| File | Module | What it shows |
|---|---|---|
| [`thread-memory-map.html`](thread-memory-map.html) | M02 | 32 lanes mapping to cache lines under different stride / load-width / AoS-vs-SoA settings; bytes-used percentage updates live |
| [`bank-conflicts.html`](bank-conflicts.html) | M03 | 32×32 vs 32×33 vs XOR-swizzled tiles colored by bank 0–31, with bank-hit histogram and serialization factor |
| [`warp-shuffle-butterfly.html`](warp-shuffle-butterfly.html) | M05 | Animated 5-round `__shfl_down_sync` and `__shfl_xor_sync`, lane→lane arrows, step/play controls |
| [`roofline.html`](roofline.html) | M04 | RTX 4090 log-log roofline (FP32 / Tensor / DRAM BW); click to drop a kernel and see bound-regime + likely stall reasons |
| [`gemm-tile-hierarchy.html`](gemm-tile-hierarchy.html) | M06 | Block tile → warp tile → thread tile → MMA fragment, sliders for every dimension, validates threads-per-warp = 32 |
| [`wmma-fragment-layout.html`](wmma-fragment-layout.html) | M07 | 16×16 and 16×8 tiles with cells labeled `lane.elt`, three fragment views (A / B / C). m16n16k16 view is illustrative since `wmma::fragment` is opaque |
| [`cp-async-pipeline.html`](cp-async-pipeline.html) | M08 | Timeline of 2-stage `cp.async` pipeline over 8 K-iterations; animated `wait_prior` boundary |
| [`online-softmax.html`](online-softmax.html) | M09 / M10 | 12 random elements arrive one at a time; running `(m, s)` updates with rescale steps highlighted |
| [`flash-attention-tiles.html`](flash-attention-tiles.html) | M10 | Q in registers, K/V tiles sliding from DRAM, per-tile S / m / l / O updating; fully driven by the online-softmax recurrence on synthetic Q,K,V |
| [`persistent-doorbell.html`](persistent-doorbell.html) | M11 | Host/kernel timeline + 7-stage state machine with PCIe latency labels; cumulative round-trip tally |
| [`cuda-graph-replay.html`](cuda-graph-replay.html) | M11 | Sequential vs CUDA-Graph timelines; sliders for kernel count / kernel duration / launch overhead; computes savings % |
| [`ptx-sass-aligned.html`](ptx-sass-aligned.html) | M13 | Hover-aligned PTX↔SASS pairs for a saxpy-shape example; shows `mad → IMAD`, `fma.rn → FFMA`, address arithmetic absorbed into `LDG` |

## Notes for instructors / readers

- **Standalone.** The HTML files don't depend on the rest of the repo and can
  be hosted from any static web root if you want to share a link. They use
  only relative paths to `lib.js`.
- **Mobile-friendly-ish.** Layouts work on phone screens for casual viewing;
  parameter exploration is easier on a desktop.
- **No analytics, no fonts, no external requests.** Everything is local.
- **Keep the source readable.** The point is for a learner to read both the
  README *and* the visualization JS — the JS is teaching content too. Don't
  minify.

## Adding a new visualization

1. Drop a new `.html` file in this directory.
2. Include `lib.js` for shared utilities (look at any existing file for the
   pattern). The exposed helpers: `header()`, `slider()`, `dropdown()`,
   `button()`, `legend()`, `injectBaseStyles()`, plus the `BANK_PALETTE` and
   `STATUS` color sets.
3. Keep the file under ~500 lines. Each viz should communicate one idea
   well; for multi-idea exploration, ship two visualizations.
4. Add an entry to the table above and to the relevant module's README.
