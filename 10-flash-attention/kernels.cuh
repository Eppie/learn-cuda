#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>      // M10.3: __pipeline_memcpy_async / commit / wait_prior
#include <mma.h>
#include <cmath>

#include "online_softmax.cuh"   // ms_pair, online_softmax_combine_with_factors

constexpr int BR = 32;     // rows of Q per block (also threads per block) — M10.0
constexpr int BC = 32;     // rows of K/V per inner tile                    — M10.0
constexpr int D  = 64;     // head dim (compile-time)

// ============================================================================
// FlashAttention-1 forward, FP32, single head.
// One block per Q row block (BR rows). One thread per Q row.
//
// IMPORTANT pedagogical simplification: real FlashAttention uses *one warp per
// ~16 Q rows* with a tensor-core inner matmul. The "one thread per Q row"
// shape we use here makes the algorithm easy to read but is severely
// under-utilizing the SM — every thread does its own D-element dot product
// against each K column with no cooperative reuse. The math is identical;
// the mapping is what evolves into production FA.
// ============================================================================
__global__ void flash_attention(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int N) {
    int row = blockIdx.x * BR + threadIdx.x;

    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    float q[D];
    float o[D];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (row < N) {
        for (int d = 0; d < D; ++d) q[d] = Q[row * D + d];
        for (int d = 0; d < D; ++d) o[d] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; ++j) {
        // Cooperative tile load: BR threads, BC*D floats each for K and V.
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int c = idx / D;
            int d = idx % D;
            int kcol = j * BC + c;
            Ks[c][d] = (kcol < N) ? K[kcol * D + d] : 0.0f;
            Vs[c][d] = (kcol < N) ? V[kcol * D + d] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            float s[BC];
            float m_ij = -INFINITY;
            for (int c = 0; c < BC; ++c) {
                int kcol = j * BC + c;
                if (kcol >= N) {
                    s[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[d] * Ks[c][d];
                    s[c] = dot * scale;
                }
                if (s[c] > m_ij) m_ij = s[c];
            }

            // Tile-local softmax pieces: P_unscaled[c] = exp(s[c] - m_ij),
            // l_ij = sum P_unscaled.  The full FA combine then folds these into
            // the running (m_state, l_state) and emits α (apply to old O) and β
            // (apply to the new tile's P · V contribution).
            float l_ij = 0.0f;
            float p[BC];
            for (int c = 0; c < BC; ++c) {
                p[c] = expf(s[c] - m_ij);
                l_ij += p[c];
            }

            os_combine_factors r = online_softmax_combine_with_factors(
                {m_state, l_state}, {m_ij, l_ij});

            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                for (int c = 0; c < BC; ++c) pv += p[c] * Vs[c][d];
                o[d] = r.alpha * o[d] + r.beta * pv;
            }

            m_state = r.p.m;
            l_state = r.p.s;
        }

        __syncthreads();
    }

    if (row < N) {
        float inv_l = 1.0f / l_state;
        for (int d = 0; d < D; ++d) O[row * D + d] = o[d] * inv_l;
    }
}

inline void launch_flash(const float* Q, const float* K, const float* V, float* O,
                         int N) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR);
    flash_attention<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// Multi-head FlashAttention.
//
// Tensors are laid out [B, H, N, D] row-major (head dim is the contiguous
// axis). The kernel is the same as the single-head version with one extra
// indirection: blockIdx.y = b*H + h selects which (B, H) slice we attend in.
//
// Grid: (Tr, B*H), block: BR threads.
// ============================================================================
__global__ void flash_attention_mha(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    int B, int H, int N) {
    int bh   = blockIdx.y;        // 0 .. B*H - 1
    int row  = blockIdx.x * BR + threadIdx.x;

    // Slice base for this (b, h).
    size_t slice_off = (size_t)bh * N * D;
    const float* Qb = Q + slice_off;
    const float* Kb = K + slice_off;
    const float* Vb = V + slice_off;
    float*       Ob = O + slice_off;

    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    float q[D];
    float o[D];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (row < N) {
        for (int d = 0; d < D; ++d) q[d] = Qb[row * D + d];
        for (int d = 0; d < D; ++d) o[d] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; ++j) {
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int c = idx / D;
            int d = idx % D;
            int kcol = j * BC + c;
            Ks[c][d] = (kcol < N) ? Kb[kcol * D + d] : 0.0f;
            Vs[c][d] = (kcol < N) ? Vb[kcol * D + d] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            float s[BC];
            float m_ij = -INFINITY;
            for (int c = 0; c < BC; ++c) {
                int kcol = j * BC + c;
                if (kcol >= N) {
                    s[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[d] * Ks[c][d];
                    s[c] = dot * scale;
                }
                if (s[c] > m_ij) m_ij = s[c];
            }

            float l_ij = 0.0f;
            float p[BC];
            for (int c = 0; c < BC; ++c) {
                p[c] = expf(s[c] - m_ij);
                l_ij += p[c];
            }

            os_combine_factors r = online_softmax_combine_with_factors(
                {m_state, l_state}, {m_ij, l_ij});

            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                for (int c = 0; c < BC; ++c) pv += p[c] * Vs[c][d];
                o[d] = r.alpha * o[d] + r.beta * pv;
            }

            m_state = r.p.m;
            l_state = r.p.s;
        }

        __syncthreads();
    }

    if (row < N) {
        float inv_l = 1.0f / l_state;
        for (int d = 0; d < D; ++d) Ob[row * D + d] = o[d] * inv_l;
    }
}

inline void launch_flash_mha(const float* Q, const float* K, const float* V,
                             float* O, int B, int H, int N) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR, B * H);
    flash_attention_mha<<<grid, block>>>(Q, K, V, O, B, H, N);
}

// ============================================================================
// Causal multi-head FlashAttention with tile-skip.
//
// Causal mask: query token at row i may only attend to keys at columns j ≤ i.
// Within a Q row block, the maximum query row is qrow_max = blockIdx.x*BR + BR-1.
// If j*BC > qrow_max, the *entire* tile is in the future — skip it. (This is
// the "tile-skip optimization": for large N it cuts inner work nearly in half.)
//
// For partial-overlap tiles (qrow_max in the middle of the tile), we apply
// the per-element mask `s[c] = -INF if kcol > qrow`. The fully-overlapped
// tiles (qrow_min ≥ kcol_max of that tile) are processed with no mask
// branches at all.
// ============================================================================
__global__ void flash_attention_causal(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int B, int H, int N) {
    int bh   = blockIdx.y;
    int row  = blockIdx.x * BR + threadIdx.x;
    int qrow_min  = blockIdx.x * BR;
    int qrow_max  = qrow_min + BR - 1;

    size_t slice_off = (size_t)bh * N * D;
    const float* Qb = Q + slice_off;
    const float* Kb = K + slice_off;
    const float* Vb = V + slice_off;
    float*       Ob = O + slice_off;

    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    float q[D];
    float o[D];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (row < N) {
        for (int d = 0; d < D; ++d) q[d] = Qb[row * D + d];
        for (int d = 0; d < D; ++d) o[d] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    int Tc = (N + BC - 1) / BC;
    // Tile j touches kcol ∈ [j*BC, j*BC + BC). The tile is fully past the
    // diagonal if j*BC > qrow_max — skip the loop entirely from that j on.
    int Tc_active = min(Tc, (qrow_max / BC) + 1);

    for (int j = 0; j < Tc_active; ++j) {
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int c = idx / D;
            int d = idx % D;
            int kcol = j * BC + c;
            Ks[c][d] = (kcol < N) ? Kb[kcol * D + d] : 0.0f;
            Vs[c][d] = (kcol < N) ? Vb[kcol * D + d] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            // Tile-mask classification:
            //   tile_kcol_max = j*BC + BC - 1
            //   - if tile_kcol_max <= qrow_min: fully in-bounds, no mask
            //   - if j*BC > row: fully masked (this thread's row, not the
            //     block) → skip the per-element loop entirely
            //   - else: partial-overlap, apply per-element mask
            int tile_kcol_min = j * BC;
            int tile_kcol_max = j * BC + BC - 1;

            if (tile_kcol_min > row) {
                // Every score for this thread is masked out; m_ij = -INF, so
                // alpha = 1 and the (m, l, O) state is unchanged. No work.
            } else {
                float s[BC];
                float m_ij = -INFINITY;
                bool full_overlap = (tile_kcol_max <= row) && (tile_kcol_max < N);
                for (int c = 0; c < BC; ++c) {
                    int kcol = j * BC + c;
                    bool valid = (kcol < N) && (full_overlap || kcol <= row);
                    if (!valid) {
                        s[c] = -INFINITY;
                    } else {
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) dot += q[d] * Ks[c][d];
                        s[c] = dot * scale;
                    }
                    if (s[c] > m_ij) m_ij = s[c];
                }

                float l_ij = 0.0f;
                float p[BC];
                for (int c = 0; c < BC; ++c) {
                    p[c] = expf(s[c] - m_ij);
                    l_ij += p[c];
                }

                os_combine_factors r = online_softmax_combine_with_factors(
                    {m_state, l_state}, {m_ij, l_ij});

                for (int d = 0; d < D; ++d) {
                    float pv = 0.0f;
                    for (int c = 0; c < BC; ++c) pv += p[c] * Vs[c][d];
                    o[d] = r.alpha * o[d] + r.beta * pv;
                }

                m_state = r.p.m;
                l_state = r.p.s;
            }
        }

        __syncthreads();
    }

    if (row < N) {
        // It's possible (for the very first row of a block, with row=0) that
        // l_state is still 0 — but row 0 sees k=0 in tile 0, so l_state ≥ 1.
        float inv_l = 1.0f / l_state;
        for (int d = 0; d < D; ++d) Ob[row * D + d] = o[d] * inv_l;
    }
}

inline void launch_flash_causal(const float* Q, const float* K, const float* V,
                                float* O, int B, int H, int N) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR, B * H);
    flash_attention_causal<<<grid, block>>>(Q, K, V, O, B, H, N);
}

// ============================================================================
// FlashAttention with KV-cache (autoregressive inference).
//
// Inference scenario: the model has already seen T_past tokens (their K, V are
// cached). We have *one new query token* per (b, h) and want to attend it
// against K[0..T-1] and V[0..T-1] where T = T_past + 1 (the current token's
// own K/V is already in the cache at index T_past).
//
// Shapes:
//   Q : [B, H,    1, D]   — one query row per (b,h)
//   K : [B, H, T_max, D]  — preallocated cache; only first T entries valid
//   V : [B, H, T_max, D]
//   O : [B, H,    1, D]
//
// One block per (b, h). One thread per output element of D, conceptually —
// but to reuse the existing kernel shape we keep BR threads per block and
// have each thread process D/BR output dims. BR=32, D=64 → 2 dims per thread.
// (Strictly: this is a degenerate FA where Tr=1; the inner loop over K/V
// tiles is the same.)
// ============================================================================
__global__ void flash_attention_kvcache(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ O,
                                        int B, int H, int T_max, int T) {
    int bh = blockIdx.x;          // 0 .. B*H - 1
    int tid = threadIdx.x;

    const float* Qb = Q + (size_t)bh * D;                // 1 query row per (b,h)
    const float* Kb = K + (size_t)bh * T_max * D;
    const float* Vb = V + (size_t)bh * T_max * D;
    float*       Ob = O + (size_t)bh * D;

    __shared__ float q_s[D];
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    // Load Q (just D floats).
    for (int idx = tid; idx < D; idx += BR) q_s[idx] = Qb[idx];
    __syncthreads();

    // Per-thread accumulator: each thread owns D/BR output dims.
    constexpr int DT = D / BR;     // dims per thread (2 for D=64, BR=32)
    static_assert(D % BR == 0, "D must be a multiple of BR for this kernel");

    float o_t[DT];
    for (int i = 0; i < DT; ++i) o_t[i] = 0.0f;
    float m_state = -INFINITY;
    float l_state = 0.0f;

    const float scale = 1.0f / sqrtf((float)D);
    int Tc = (T + BC - 1) / BC;

    for (int j = 0; j < Tc; ++j) {
        // Cooperative load of BC × D floats from K[j*BC..] and V[j*BC..].
        for (int idx = tid; idx < BC * D; idx += BR) {
            int c = idx / D;
            int d = idx % D;
            int kcol = j * BC + c;
            Ks[c][d] = (kcol < T) ? Kb[kcol * D + d] : 0.0f;
            Vs[c][d] = (kcol < T) ? Vb[kcol * D + d] : 0.0f;
        }
        __syncthreads();

        // All threads in the block compute the *same* score vector s[BC]
        // because there's only one query row. We could parallelize across
        // BC, but since we already syncthreads we can let every thread
        // recompute (cheap). Cleaner: thread c (c < BC) computes s[c],
        // then we shuffle/share. Simplest approach: every thread does it
        // — D=64 dot product per c, BC=32 c's, BR=32 threads — fine for
        // pedagogy, not optimal.
        float s[BC];
        float m_ij = -INFINITY;
        for (int c = 0; c < BC; ++c) {
            int kcol = j * BC + c;
            if (kcol >= T) {
                s[c] = -INFINITY;
            } else {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += q_s[d] * Ks[c][d];
                s[c] = dot * scale;
            }
            if (s[c] > m_ij) m_ij = s[c];
        }

        float l_ij = 0.0f;
        float p[BC];
        for (int c = 0; c < BC; ++c) {
            p[c] = expf(s[c] - m_ij);
            l_ij += p[c];
        }

        os_combine_factors r = online_softmax_combine_with_factors(
            {m_state, l_state}, {m_ij, l_ij});

        // Each thread accumulates its DT output dims.
        for (int i = 0; i < DT; ++i) {
            int d = tid * DT + i;
            float pv = 0.0f;
            for (int c = 0; c < BC; ++c) pv += p[c] * Vs[c][d];
            o_t[i] = r.alpha * o_t[i] + r.beta * pv;
        }

        m_state = r.p.m;
        l_state = r.p.s;

        __syncthreads();
    }

    float inv_l = 1.0f / l_state;
    for (int i = 0; i < DT; ++i) {
        int d = tid * DT + i;
        Ob[d] = o_t[i] * inv_l;
    }
}

inline void launch_flash_kvcache(const float* Q, const float* K, const float* V,
                                 float* O, int B, int H, int T_max, int T) {
    dim3 block(BR);
    dim3 grid(B * H);
    flash_attention_kvcache<<<grid, block>>>(Q, K, V, O, B, H, T_max, T);
}

// ============================================================================
// Grouped-Query Attention (GQA, Llama-style).
//
// num_q_heads = H_q, num_kv_heads = H_kv, with H_q % H_kv == 0. Group size
// g = H_q / H_kv: each KV head is shared by `g` consecutive query heads.
// Mapping: query head h_q → KV head h_q / g.
//
// Q : [B, H_q,  N, D]
// K : [B, H_kv, N, D]
// V : [B, H_kv, N, D]
// O : [B, H_q,  N, D]
//
// Saves memory and bandwidth on the KV cache during inference (the dominant
// cost at long contexts). Algorithmically identical to MHA — just a
// different head→head index mapping.
// ============================================================================
__global__ void flash_attention_gqa(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    int B, int H_q, int H_kv, int N) {
    int bh_q = blockIdx.y;             // 0 .. B*H_q - 1
    int b    = bh_q / H_q;
    int h_q  = bh_q % H_q;
    int g    = H_q / H_kv;
    int h_kv = h_q / g;
    int bh_kv = b * H_kv + h_kv;

    int row = blockIdx.x * BR + threadIdx.x;

    size_t q_slice  = (size_t)bh_q  * N * D;
    size_t kv_slice = (size_t)bh_kv * N * D;
    const float* Qb = Q + q_slice;
    const float* Kb = K + kv_slice;
    const float* Vb = V + kv_slice;
    float*       Ob = O + q_slice;

    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];

    float q[D];
    float o[D];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (row < N) {
        for (int d = 0; d < D; ++d) q[d] = Qb[row * D + d];
        for (int d = 0; d < D; ++d) o[d] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; ++j) {
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int c = idx / D;
            int d = idx % D;
            int kcol = j * BC + c;
            Ks[c][d] = (kcol < N) ? Kb[kcol * D + d] : 0.0f;
            Vs[c][d] = (kcol < N) ? Vb[kcol * D + d] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            float s[BC];
            float m_ij = -INFINITY;
            for (int c = 0; c < BC; ++c) {
                int kcol = j * BC + c;
                if (kcol >= N) {
                    s[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[d] * Ks[c][d];
                    s[c] = dot * scale;
                }
                if (s[c] > m_ij) m_ij = s[c];
            }

            float l_ij = 0.0f;
            float p[BC];
            for (int c = 0; c < BC; ++c) {
                p[c] = expf(s[c] - m_ij);
                l_ij += p[c];
            }

            os_combine_factors r = online_softmax_combine_with_factors(
                {m_state, l_state}, {m_ij, l_ij});

            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                for (int c = 0; c < BC; ++c) pv += p[c] * Vs[c][d];
                o[d] = r.alpha * o[d] + r.beta * pv;
            }

            m_state = r.p.m;
            l_state = r.p.s;
        }

        __syncthreads();
    }

    if (row < N) {
        float inv_l = 1.0f / l_state;
        for (int d = 0; d < D; ++d) Ob[row * D + d] = o[d] * inv_l;
    }
}

inline void launch_flash_gqa(const float* Q, const float* K, const float* V,
                             float* O, int B, int H_q, int H_kv, int N) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR, B * H_q);
    flash_attention_gqa<<<grid, block>>>(Q, K, V, O, B, H_q, H_kv, N);
}

// ============================================================================
// M10.1 — Warp-cooperative FlashAttention, FP32.
//
// The pedagogical kernel above ("one thread per Q row") leaves the warp idle:
// each thread does its own D-element dot product against every K column with
// no cross-lane reuse. M10.1 keeps the simple "one thread per Q row" inner
// math but bumps the block size to 128 threads (4 warps), so each block
// services BR1 = 128 Q rows. The K/V tile is loaded once per block and reused
// across 4× more rows — the biggest available win at FP32 without going to
// tensor cores.
//
// What changes vs M10.0:
//   * Block size 32 → 128 (4 warps).
//   * Rows per block 32 → 128 (BR1=128).
//   * The cooperative K/V tile load now spreads BC1*D1 = 32*64 = 2048 floats
//     across 128 threads — 16 floats/thread, 4 vector-2 loads (could be
//     vectorised further; left as an exercise).
//   * Inner per-row math is identical: each thread does s[c] = q · K[c, :]
//     for c ∈ [0, BC1), then online-softmax + PV.
//
// The "warp-cooperative" name is slightly aspirational at FP32 — at FP32 the
// FMA pipe is the bottleneck, not the shared-memory or shuffle paths, so the
// simple "more rows per K-tile load" change is the lever that actually moves
// performance. The genuine warp-cooperation pattern (lanes parallelising D)
// is what M10.2 does on tensor cores.
// ============================================================================
constexpr int BR1 = 128;
constexpr int BC1 = 32;
constexpr int D1  = 64;
constexpr int M10_1_THREADS = BR1;       // 128 threads per block, one per row

__global__ void flash_attention_warp(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     int N) {
    int row = blockIdx.x * BR1 + threadIdx.x;

    __shared__ float Ks[BC1][D1];
    __shared__ float Vs[BC1][D1];

    float q[D1];
    float o[D1];
    float m_state = -INFINITY;
    float l_state = 0.0f;

    if (row < N) {
        #pragma unroll
        for (int d = 0; d < D1; ++d) q[d] = Q[row * D1 + d];
        #pragma unroll
        for (int d = 0; d < D1; ++d) o[d] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D1);
    int Tc = (N + BC1 - 1) / BC1;

    for (int j = 0; j < Tc; ++j) {
        // Cooperative tile load: 128 threads, BC1 * D1 = 2048 floats each tile.
        // 16 floats/thread; we use a strided scalar loop here for simplicity.
        for (int idx = threadIdx.x; idx < BC1 * D1; idx += M10_1_THREADS) {
            int c = idx / D1;
            int d = idx % D1;
            int kcol = j * BC1 + c;
            Ks[c][d] = (kcol < N) ? K[kcol * D1 + d] : 0.0f;
            Vs[c][d] = (kcol < N) ? V[kcol * D1 + d] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            float s[BC1];
            float m_ij = -INFINITY;
            #pragma unroll
            for (int c = 0; c < BC1; ++c) {
                int kcol = j * BC1 + c;
                if (kcol >= N) {
                    s[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D1; ++d) dot += q[d] * Ks[c][d];
                    s[c] = dot * scale;
                }
                if (s[c] > m_ij) m_ij = s[c];
            }

            float l_ij = 0.0f;
            float p[BC1];
            #pragma unroll
            for (int c = 0; c < BC1; ++c) {
                p[c] = __expf(s[c] - m_ij);
                l_ij += p[c];
            }

            os_combine_factors r = online_softmax_combine_with_factors(
                {m_state, l_state}, {m_ij, l_ij});

            #pragma unroll
            for (int d = 0; d < D1; ++d) {
                float pv = 0.0f;
                #pragma unroll
                for (int c = 0; c < BC1; ++c) pv += p[c] * Vs[c][d];
                o[d] = r.alpha * o[d] + r.beta * pv;
            }

            m_state = r.p.m;
            l_state = r.p.s;
        }

        __syncthreads();
    }

    if (row < N) {
        float inv_l = 1.0f / l_state;
        #pragma unroll
        for (int d = 0; d < D1; ++d) O[row * D1 + d] = o[d] * inv_l;
    }
}

inline void launch_flash_warp(const float* Q, const float* K, const float* V,
                              float* O, int N) {
    dim3 block(M10_1_THREADS);                // 128 threads = 4 warps
    dim3 grid((N + BR1 - 1) / BR1);
    flash_attention_warp<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// M10.2 — WMMA FlashAttention. FP16 inputs (Q, K, V), FP32 accumulators,
// FP32 output. The inner Q · Kᵀ and P · V matmuls run on tensor cores.
//
// Block / fragment shape:
//   BR2 = 16, BC2 = 16, D2 = 64.   (16×16×16 is the canonical fp16-input WMMA tile.)
//   WARPS_M10_2 = 4 warps per block. Each warp owns one BR2×D2 = 16×64 Q-tile,
//   so a block services BR2_BLOCK = 64 Q rows.
//   Grid = (Tr / WARPS_M10_2) where Tr = N / BR2.
//
// Per-warp objects (all opaque WMMA fragments, kept in registers):
//   q_frag[D2/16=4]    matrix_a 16×16 row_major  — Q tile, kept across all tiles.
//   o_frag[D2/16=4]    accumulator 16×16 float    — running output.
//   s_frag             accumulator 16×16 float    — score tile, recomputed each iter.
//
// Per-warp shared scratch (the only place we leave fragment-land):
//   Ssm  : 16 × 16 FP32         — S after store_matrix_sync, for softmax
//   Psm  : 16 × 16 FP16         — P after exp, fed to the second mma_sync
//   Osm  : 16 × 64 FP32         — scratch for the per-row α-scaling of O
//   row state per warp           — m, l, alpha per Q row (16 rows × 3 floats)
//
// Block-shared: Ks, Vs (16 × 64 FP16 each). All 4 warps share these tile loads,
// which is the whole point of going wide — the load cost is amortised 4×.
//
// Online-softmax-on-fragments dance, per K/V tile (each warp does this on its
// own per-warp scratch, in parallel):
//   1. s_frag = 0;  for each kk in [0, D2, 16): s_frag += Q[:, kk:kk+16] · K[kk:kk+16, :]
//      via `wmma::mma_sync`. K is row-major in shared memory; reading as
//      `matrix_b col_major` gives Kᵀ — exactly what `S = Q · Kᵀ` needs.
//   2. store_matrix_sync(Ssm, s_frag): materialise S to shared row-major.
//   3. Per-row softmax on Ssm: lane r ∈ [0,16) handles its row; lanes 16..31
//      are idle in this section (BC2=16 is small enough that warp-parallel
//      reduction inside the row isn't worth the code complexity).
//   4. Write P = exp(S - m_new) to Psm as FP16.
//   5. Scale o_frag by α[row]: store_matrix_sync each o_frag → Osm, scale
//      row-wise, load_matrix_sync back. (o_frag's layout is opaque, so this
//      shared-memory round-trip is the WMMA-API way; raw mma.sync would let
//      us scale per-register.)
//   6. WMMA: o_frag[d] += P · V[:, d*16:(d+1)*16]  for d in [0, 4).
//   7. Update m_state, l_state per row.
// ============================================================================

constexpr int BR2 = 16;
constexpr int BC2 = 32;                            // 2 WMMA "n" tiles wide
constexpr int D2  = 64;
constexpr int FRAGS_D2     = D2 / 16;              // 4
constexpr int FRAGS_BC2    = BC2 / 16;             // 2
constexpr int WARPS_M10_2  = 4;
constexpr int BR2_BLOCK    = BR2 * WARPS_M10_2;    // 64 Q rows per block
constexpr int M10_2_THREADS = WARPS_M10_2 * 32;    // 128

using namespace nvcuda;

__global__ void flash_attention_wmma(const __half* __restrict__ Q,
                                     const __half* __restrict__ K,
                                     const __half* __restrict__ V,
                                     float*        __restrict__ O,
                                     int N) {
    int warp_id = threadIdx.x >> 5;     // 0 .. WARPS_M10_2-1
    int lane    = threadIdx.x & 31;

    int row_base = blockIdx.x * BR2_BLOCK + warp_id * BR2;

    // Block-shared K/V tiles (loaded once per outer iter, shared across warps).
    __shared__ __half Ks[BC2 * D2];
    __shared__ __half Vs[BC2 * D2];

    // Per-warp scratch for softmax / O-scaling round trips.
    // Osm doubles as the Q staging buffer at startup (first 8 KB per warp,
    // viewed as fp16). Q is no longer needed after q_frag is loaded.
    __shared__ float  Ssm_blk     [WARPS_M10_2 * BR2 * BC2];
    __shared__ __half Psm_blk     [WARPS_M10_2 * BR2 * BC2];
    __shared__ float  Osm_blk     [WARPS_M10_2 * BR2 * D2];
    __shared__ float  m_state_blk [WARPS_M10_2 * BR2];
    __shared__ float  l_state_blk [WARPS_M10_2 * BR2];
    __shared__ float  alpha_blk   [WARPS_M10_2 * BR2];

    float*  Ssm     = &Ssm_blk    [warp_id * BR2 * BC2];
    __half* Psm     = &Psm_blk    [warp_id * BR2 * BC2];
    float*  Osm     = &Osm_blk    [warp_id * BR2 * D2];
    float*  m_state_sm = &m_state_blk[warp_id * BR2];
    float*  l_state_sm = &l_state_blk[warp_id * BR2];
    float*  alpha_sm   = &alpha_blk  [warp_id * BR2];
    __half* Qstage  = reinterpret_cast<__half*>(Osm);   // reuse Osm bytes

    // Initialize per-row state.
    if (lane < BR2) {
        m_state_sm[lane] = -INFINITY;
        l_state_sm[lane] = 0.0f;
    }

    // ---- Stage Q for this warp into Qstage, then load 4 frags from shared ----
    for (int idx = lane; idx < BR2 * D2; idx += 32) {
        int r = idx / D2;
        int d = idx % D2;
        int row = row_base + r;
        Qstage[r * D2 + d] = (row < N) ? Q[row * D2 + d] : __float2half(0.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[FRAGS_D2];
    #pragma unroll
    for (int kk = 0; kk < FRAGS_D2; ++kk) {
        wmma::load_matrix_sync(q_frag[kk], &Qstage[kk * 16], D2);
    }

    // ---- O accumulator fragments (registers) ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[FRAGS_D2];
    #pragma unroll
    for (int kk = 0; kk < FRAGS_D2; ++kk) wmma::fill_fragment(o_frag[kk], 0.0f);

    const float scale = 1.0f / sqrtf((float)D2);
    int Tc = (N + BC2 - 1) / BC2;

    for (int j = 0; j < Tc; ++j) {
        // ---- 1. BLOCK-COOPERATIVE tile load: K, V (16 × 64 FP16) ----
        // 1024 halfs per tile; with 128 threads that's 8 halfs/thread.
        for (int idx = threadIdx.x; idx < BC2 * D2; idx += M10_2_THREADS) {
            int c = idx / D2;
            int d = idx % D2;
            int kcol = j * BC2 + c;
            __half kv = (kcol < N) ? K[kcol * D2 + d] : __float2half(0.0f);
            __half vv = (kcol < N) ? V[kcol * D2 + d] : __float2half(0.0f);
            Ks[c * D2 + d] = kv;
            Vs[c * D2 + d] = vv;
        }
        __syncthreads();

        // ---- 2. S = Q · Kᵀ via WMMA ----
        // K is row-major BC2 × D2; matrix_b col_major reads it as Kᵀ.
        // S is BR2 × BC2 = 16 × 32 = FRAGS_BC2 fragments wide.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[FRAGS_BC2];
        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) wmma::fill_fragment(s_frag[n], 0.0f);

        #pragma unroll
        for (int kk = 0; kk < FRAGS_D2; ++kk) {
            #pragma unroll
            for (int n = 0; n < FRAGS_BC2; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
                // K[n*16..n*16+16 rows, kk*16..kk*16+16 cols] → matrix_b col_major
                // gives the (kk*16..)-block of cols of Kᵀ — 16 rows of Q dim,
                // 16 cols of K-row-index. Pointer: row n*16, col kk*16 of Ks.
                wmma::load_matrix_sync(k_frag, &Ks[(n * 16) * D2 + kk * 16], D2);
                wmma::mma_sync(s_frag[n], q_frag[kk], k_frag, s_frag[n]);
            }
        }

        // Scale S by 1/sqrt(D).
        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) {
            #pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; ++i) s_frag[n].x[i] *= scale;
        }

        // ---- 3. Materialize S, mask OOB cols, run per-row softmax ----
        // Ssm is row-major BR2 × BC2; each s_frag covers cols [n*16, n*16+16).
        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) {
            wmma::store_matrix_sync(&Ssm[n * 16], s_frag[n], BC2, wmma::mem_row_major);
        }
        __syncwarp();

        if (j == Tc - 1) {
            // Mask OOB K columns to -INFINITY.
            for (int idx = lane; idx < BR2 * BC2; idx += 32) {
                int c = idx % BC2;
                int kcol = j * BC2 + c;
                if (kcol >= N) Ssm[idx] = -INFINITY;
            }
            __syncwarp();
        }

        // Lane r ∈ [0, 16) handles row r.
        if (lane < BR2) {
            int r = lane;
            float* row_S = &Ssm[r * BC2];

            float m_ij = -INFINITY;
            #pragma unroll
            for (int c = 0; c < BC2; ++c) m_ij = fmaxf(m_ij, row_S[c]);

            float m_state = m_state_sm[r];
            float m_new   = fmaxf(m_state, m_ij);
            float alpha   = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);

            float l_ij = 0.0f;
            #pragma unroll
            for (int c = 0; c < BC2; ++c) {
                float p = __expf(row_S[c] - m_new);
                Psm[r * BC2 + c] = __float2half(p);
                l_ij += p;
            }

            l_state_sm[r] = alpha * l_state_sm[r] + l_ij;
            m_state_sm[r] = m_new;
            alpha_sm[r]   = alpha;
        }
        __syncwarp();

        // ---- 4. Scale o_frag by α[row]: WMMA fragments → shared → fragments ----
        #pragma unroll
        for (int kk = 0; kk < FRAGS_D2; ++kk) {
            wmma::store_matrix_sync(&Osm[kk * 16], o_frag[kk], D2, wmma::mem_row_major);
        }
        __syncwarp();

        for (int idx = lane; idx < BR2 * D2; idx += 32) {
            int r = idx / D2;
            Osm[idx] *= alpha_sm[r];
        }
        __syncwarp();

        #pragma unroll
        for (int kk = 0; kk < FRAGS_D2; ++kk) {
            wmma::load_matrix_sync(o_frag[kk], &Osm[kk * 16], D2, wmma::mem_row_major);
        }

        // ---- 5. O += P · V via WMMA, per-d-chunk ----
        // P is BR2 × BC2 = 16 × 32 = FRAGS_BC2 wide.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag[FRAGS_BC2];
        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) {
            wmma::load_matrix_sync(p_frag[n], &Psm[n * 16], BC2);
        }

        #pragma unroll
        for (int kk = 0; kk < FRAGS_D2; ++kk) {
            #pragma unroll
            for (int n = 0; n < FRAGS_BC2; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
                // V[n*16..n*16+16 rows, kk*16..kk*16+16 cols], row-major stride D2.
                wmma::load_matrix_sync(v_frag, &Vs[(n * 16) * D2 + kk * 16], D2);
                wmma::mma_sync(o_frag[kk], p_frag[n], v_frag, o_frag[kk]);
            }
        }

        __syncthreads();    // before next iter overwrites Ks/Vs (block-shared)
    }

    // ---- Final: divide each row of O by l_state[row], write FP32 output ----
    #pragma unroll
    for (int kk = 0; kk < FRAGS_D2; ++kk) {
        wmma::store_matrix_sync(&Osm[kk * 16], o_frag[kk], D2, wmma::mem_row_major);
    }
    __syncwarp();

    for (int idx = lane; idx < BR2 * D2; idx += 32) {
        int r = idx / D2;
        int d = idx % D2;
        int row = row_base + r;
        if (row < N) {
            float inv_l = (l_state_sm[r] > 0.0f) ? (1.0f / l_state_sm[r]) : 0.0f;
            O[row * D2 + d] = Osm[idx] * inv_l;
        }
    }
}

inline void launch_flash_wmma(const __half* Q, const __half* K, const __half* V,
                              float* O, int N) {
    dim3 block(M10_2_THREADS);                  // 128 threads = 4 warps
    dim3 grid((N + BR2_BLOCK - 1) / BR2_BLOCK);
    flash_attention_wmma<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// M10.3 — cp.async + WMMA FlashAttention (stretch in README §9).
//
// Same algorithm as M10.2; the only change is that the K/V tile loads are
// double-buffered with cp.async so that the next tile's DRAM->SMEM transfer
// is in flight while the current tile is being consumed by the WMMA dance.
//
// Reference structure: M08 `gemm_v1_async` (legacy `__pipeline_memcpy_async`
// API, STAGES-deep). We pick STAGES = 2 here: the extra SMEM cost is small
// (8 KB on top of M10.2's ~38 KB) and it fits in the default 48 KB without
// `cudaFuncSetAttribute`. STAGES >= 3 is left as further stretch.
//
// Pipeline shape (per block):
//
//   issue_loads(K[0], V[0] -> Ks[0], Vs[0])
//   __pipeline_commit();
//
//   for j in [0, Tc):
//       if j + 1 < Tc:
//           issue_loads(K[j+1], V[j+1] -> Ks[(j+1)%2], Vs[(j+1)%2])
//       __pipeline_commit();
//       __pipeline_wait_prior(STAGES - 1);   // oldest commit done
//       __syncthreads();
//
//       // ... M10.2 inner body on Ks[j%2], Vs[j%2] ...
//
//       __syncthreads();   // WAR fence: next iter's cp.async writes a buffer
//                          // that aliases this iter's compute reads under
//                          // STAGES=2 (see M08 §3a).
//
// Out-of-bounds K/V rows in the final tile: cp.async loads from a *clamped*
// source row (min(kcol, N-1)), giving finite garbage values. M10.2's existing
// post-softmax mask sets the corresponding S columns to -INF, so the
// resulting P entries are 0 and the garbage Vs values get multiplied out.
// (The garbage values must be finite, not NaN/inf, because 0 * NaN = NaN.)
// ============================================================================

constexpr int STAGES_M10_3 = 2;
constexpr int M10_3_THREADS = M10_2_THREADS;       // reuse 128 (4 warps)

// 16-byte cp.async = 8 halves per transfer. With BC2 * D2 = 32 * 64 = 2048
// halves per tile and 128 threads, that's 2 transfers per thread.
constexpr int M10_3_VEC_HALVES = 8;

__device__ __forceinline__ void m10_3_cp_async_16(void* smem_dst, const void* gmem_src) {
    // 16-byte cp.async. Identical to what `__pipeline_memcpy_async(.., 16)`
    // emits: `cp.async.cg.shared.global ..., 16, 16` (skip L1, cache at L2).
    __pipeline_memcpy_async(smem_dst, gmem_src, sizeof(int4));
}

__global__ void flash_attention_async_wmma(const __half* __restrict__ Q,
                                           const __half* __restrict__ K,
                                           const __half* __restrict__ V,
                                           float*        __restrict__ O,
                                           int N) {
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    int row_base = blockIdx.x * BR2_BLOCK + warp_id * BR2;

    // Double-buffered K/V tiles.
    __shared__ __half Ks[STAGES_M10_3][BC2 * D2];
    __shared__ __half Vs[STAGES_M10_3][BC2 * D2];

    // M10.2's per-warp scratch (unchanged).
    __shared__ float  Ssm_blk     [WARPS_M10_2 * BR2 * BC2];
    __shared__ __half Psm_blk     [WARPS_M10_2 * BR2 * BC2];
    __shared__ float  Osm_blk     [WARPS_M10_2 * BR2 * D2];
    __shared__ float  m_state_blk [WARPS_M10_2 * BR2];
    __shared__ float  l_state_blk [WARPS_M10_2 * BR2];
    __shared__ float  alpha_blk   [WARPS_M10_2 * BR2];

    float*  Ssm        = &Ssm_blk    [warp_id * BR2 * BC2];
    __half* Psm        = &Psm_blk    [warp_id * BR2 * BC2];
    float*  Osm        = &Osm_blk    [warp_id * BR2 * D2];
    float*  m_state_sm = &m_state_blk[warp_id * BR2];
    float*  l_state_sm = &l_state_blk[warp_id * BR2];
    float*  alpha_sm   = &alpha_blk  [warp_id * BR2];
    __half* Qstage     = reinterpret_cast<__half*>(Osm);   // reuse Osm bytes

    if (lane < BR2) {
        m_state_sm[lane] = -INFINITY;
        l_state_sm[lane] = 0.0f;
    }

    // ---- Stage Q for this warp into Qstage, then load 4 frags ----
    for (int idx = lane; idx < BR2 * D2; idx += 32) {
        int r = idx / D2;
        int d = idx % D2;
        int row = row_base + r;
        Qstage[r * D2 + d] = (row < N) ? Q[row * D2 + d] : __float2half(0.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[FRAGS_D2];
    #pragma unroll
    for (int kk = 0; kk < FRAGS_D2; ++kk) {
        wmma::load_matrix_sync(q_frag[kk], &Qstage[kk * 16], D2);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[FRAGS_D2];
    #pragma unroll
    for (int kk = 0; kk < FRAGS_D2; ++kk) wmma::fill_fragment(o_frag[kk], 0.0f);

    const float scale = 1.0f / sqrtf((float)D2);
    int Tc = (N + BC2 - 1) / BC2;

    // ------------------------------------------------------------------
    // cp.async issue helper.
    //
    // Each tile is BC2*D2 = 2048 halves = 4096 B. With 128 threads * 16 B
    // (= 8 halves) per transfer we cover 128 * 16 = 2048 B per pass; so we
    // need 2 passes per tile (= 2 transfers per thread for K, 2 for V).
    //
    // For OOB rows in the final tile we clamp the source row to N-1 (still
    // a valid global address); the post-softmax -INF mask zeroes the
    // resulting P column so the garbage value never reaches the output.
    // ------------------------------------------------------------------
    auto issue_tile = [&](int stage, int j, bool may_be_oob) {
        // 2048 halves / (128 threads * 8 halves) = 2 passes per K (and 2 per V).
        // We unroll the per-thread pass count explicitly so the compiler can
        // schedule the cp.async instructions tightly.
        constexpr int N_TRANSFERS_PER_TILE = (BC2 * D2) / M10_3_VEC_HALVES;  // 256
        constexpr int VECS_PER_ROW         = D2 / M10_3_VEC_HALVES;          // 8
        constexpr int PASSES               = N_TRANSFERS_PER_TILE / M10_3_THREADS;  // 2

        #pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            int t = p * M10_3_THREADS + threadIdx.x;
            int row_in_tile = t / VECS_PER_ROW;       // 0..BC2-1
            int col_in_row  = (t % VECS_PER_ROW) * M10_3_VEC_HALVES;
            int kcol = j * BC2 + row_in_tile;
            int src_row = may_be_oob ? ((kcol < N) ? kcol : (N - 1)) : kcol;

            __half*       k_smem = &Ks[stage][row_in_tile * D2 + col_in_row];
            __half*       v_smem = &Vs[stage][row_in_tile * D2 + col_in_row];
            const __half* k_gmem = &K[src_row * D2 + col_in_row];
            const __half* v_gmem = &V[src_row * D2 + col_in_row];

            m10_3_cp_async_16(k_smem, k_gmem);
            m10_3_cp_async_16(v_smem, v_gmem);
        }
    };

    // ---- Prologue: kick off tile 0 ----
    if (Tc > 0) issue_tile(0, 0, /*may_be_oob=*/(Tc == 1));
    __pipeline_commit();

    int compute_stage = 0;
    int load_stage    = 1 % STAGES_M10_3;

    for (int j = 0; j < Tc; ++j) {
        // Kick off the NEXT tile (if any) into load_stage.
        if (j + 1 < Tc) {
            issue_tile(load_stage, j + 1, /*may_be_oob=*/(j + 2 == Tc));
        }
        __pipeline_commit();

        // Wait for the tile we're about to consume.
        __pipeline_wait_prior(STAGES_M10_3 - 1);
        __syncthreads();

        // ---- Identical M10.2 inner body on Ks[compute_stage], Vs[compute_stage] ----
        __half* Ks_cur = Ks[compute_stage];
        __half* Vs_cur = Vs[compute_stage];

        // S = Q · Kᵀ
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[FRAGS_BC2];
        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) wmma::fill_fragment(s_frag[n], 0.0f);

        #pragma unroll
        for (int kk = 0; kk < FRAGS_D2; ++kk) {
            #pragma unroll
            for (int n = 0; n < FRAGS_BC2; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
                wmma::load_matrix_sync(k_frag, &Ks_cur[(n * 16) * D2 + kk * 16], D2);
                wmma::mma_sync(s_frag[n], q_frag[kk], k_frag, s_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) {
            #pragma unroll
            for (int i = 0; i < s_frag[n].num_elements; ++i) s_frag[n].x[i] *= scale;
        }

        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) {
            wmma::store_matrix_sync(&Ssm[n * 16], s_frag[n], BC2, wmma::mem_row_major);
        }
        __syncwarp();

        if (j == Tc - 1) {
            for (int idx = lane; idx < BR2 * BC2; idx += 32) {
                int c = idx % BC2;
                int kcol = j * BC2 + c;
                if (kcol >= N) Ssm[idx] = -INFINITY;
            }
            __syncwarp();
        }

        if (lane < BR2) {
            int r = lane;
            float* row_S = &Ssm[r * BC2];

            float m_ij = -INFINITY;
            #pragma unroll
            for (int c = 0; c < BC2; ++c) m_ij = fmaxf(m_ij, row_S[c]);

            float m_state = m_state_sm[r];
            float m_new   = fmaxf(m_state, m_ij);
            float alpha   = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);

            float l_ij = 0.0f;
            #pragma unroll
            for (int c = 0; c < BC2; ++c) {
                float p = __expf(row_S[c] - m_new);
                Psm[r * BC2 + c] = __float2half(p);
                l_ij += p;
            }

            l_state_sm[r] = alpha * l_state_sm[r] + l_ij;
            m_state_sm[r] = m_new;
            alpha_sm[r]   = alpha;
        }
        __syncwarp();

        // Scale o_frag by α[row] via the shared-memory round trip.
        // On iter 0, o_frag is identically zero (filled above) and alpha is 0
        // by construction (m_state == -INFINITY); the scaling is a no-op.
        // Skipping it saves the store + scale + load round trip on the first
        // iteration of every block — a small but free win.
        if (j > 0) {
            #pragma unroll
            for (int kk = 0; kk < FRAGS_D2; ++kk) {
                wmma::store_matrix_sync(&Osm[kk * 16], o_frag[kk], D2, wmma::mem_row_major);
            }
            __syncwarp();

            for (int idx = lane; idx < BR2 * D2; idx += 32) {
                int r = idx / D2;
                Osm[idx] *= alpha_sm[r];
            }
            __syncwarp();

            #pragma unroll
            for (int kk = 0; kk < FRAGS_D2; ++kk) {
                wmma::load_matrix_sync(o_frag[kk], &Osm[kk * 16], D2, wmma::mem_row_major);
            }
        }

        // O += P · V
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag[FRAGS_BC2];
        #pragma unroll
        for (int n = 0; n < FRAGS_BC2; ++n) {
            wmma::load_matrix_sync(p_frag[n], &Psm[n * 16], BC2);
        }

        #pragma unroll
        for (int kk = 0; kk < FRAGS_D2; ++kk) {
            #pragma unroll
            for (int n = 0; n < FRAGS_BC2; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag, &Vs_cur[(n * 16) * D2 + kk * 16], D2);
                wmma::mma_sync(o_frag[kk], p_frag[n], v_frag, o_frag[kk]);
            }
        }

        // WAR fence: next iter's cp.async into load_stage may alias this
        // iter's compute_stage buffer (STAGES=2). Block-wide barrier so all
        // warps finished reading Ks_cur / Vs_cur before any cp.async store
        // can overwrite them.
        __syncthreads();

        compute_stage = (compute_stage + 1) % STAGES_M10_3;
        load_stage    = (load_stage    + 1) % STAGES_M10_3;
    }

    // Final O normalization (identical to M10.2).
    #pragma unroll
    for (int kk = 0; kk < FRAGS_D2; ++kk) {
        wmma::store_matrix_sync(&Osm[kk * 16], o_frag[kk], D2, wmma::mem_row_major);
    }
    __syncwarp();

    for (int idx = lane; idx < BR2 * D2; idx += 32) {
        int r = idx / D2;
        int d = idx % D2;
        int row = row_base + r;
        if (row < N) {
            float inv_l = (l_state_sm[r] > 0.0f) ? (1.0f / l_state_sm[r]) : 0.0f;
            O[row * D2 + d] = Osm[idx] * inv_l;
        }
    }
}

inline void launch_flash_async_wmma(const __half* Q, const __half* K, const __half* V,
                                    float* O, int N) {
    // Sanity: shared memory footprint is well under the 48 KB default.
    //   Ks: 2 * 32 * 64 * 2 B  = 8192 B
    //   Vs: 2 * 32 * 64 * 2 B  = 8192 B
    //   Ssm_blk:  4 * 16 * 32 * 4 B = 8192 B
    //   Psm_blk:  4 * 16 * 32 * 2 B = 4096 B
    //   Osm_blk:  4 * 16 * 64 * 4 B = 16384 B
    //   state arrays: 4 * 16 * 4 * 3 B = 768 B
    //   total ~ 45.6 KB    (no cudaFuncSetAttribute needed)
    dim3 block(M10_3_THREADS);
    dim3 grid((N + BR2_BLOCK - 1) / BR2_BLOCK);
    flash_attention_async_wmma<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// Naive reference: three kernels, materialized N×N attention matrix.
// (Single-head only — used by bench.cu for the headline comparison. Multi-head
// references for the new variants live in solution.cu as host-side functions.)
// ============================================================================
__global__ void naive_qkt(const float* Q, const float* K, float* S, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += Q[row * D + d] * K[col * D + d];
        S[row * N + col] = dot * (1.0f / sqrtf((float)D));
    }
}

__global__ void naive_softmax(float* S, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    float* x = S + row * N;

    constexpr int BLK = 256;
    __shared__ float smem[BLK / 32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += BLK) m = fmaxf(m, x[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < BLK / 32) ? smem[lane] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) smem[0] = m;
    }
    __syncthreads();
    float row_max = smem[0];

    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLK) s += expf(x[i] - row_max);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) smem[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < BLK / 32) ? smem[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) smem[0] = s;
    }
    __syncthreads();
    float inv = 1.0f / smem[0];

    for (int i = threadIdx.x; i < N; i += BLK) x[i] = expf(x[i] - row_max) * inv;
}

__global__ void naive_pv(const float* P, const float* V, float* O, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && dim < D) {
        float dot = 0.0f;
        for (int c = 0; c < N; ++c) dot += P[row * N + c] * V[c * D + dim];
        O[row * D + dim] = dot;
    }
}

inline void launch_naive(const float* Q, const float* K, const float* V, float* O,
                         float* S_buf, int N) {
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        naive_qkt<<<grid, block>>>(Q, K, S_buf, N);
    }
    naive_softmax<<<N, 256>>>(S_buf, N);
    {
        dim3 block(D, 16);
        dim3 grid(1, (N + 15) / 16);
        naive_pv<<<grid, block>>>(S_buf, V, O, N);
    }
}
