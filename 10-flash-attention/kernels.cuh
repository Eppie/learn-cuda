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
// M10.4 — Raw mma.sync FlashAttention (FP16 inputs, FP32 accumulators).
//
// What changes vs M10.3 (cp.async + WMMA):
//
// M10.3 hits ~28 TF/s. The remaining cost (over the cuBLAS-hgemm peak of
// ~160 TF/s on this card) is the softmax / O-rescale shared-memory round
// trip: WMMA fragments have an opaque layout, so to do row-softmax on S we
// must `store_matrix_sync(Ssm, s_frag)` and to rescale O by alpha we must
// `store_matrix_sync(Osm, o_frag)`, scale, then `load_matrix_sync` back.
// Both round-trips are pure overhead with no math content.
//
// Raw `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` exposes its
// per-lane register layout exactly (PTX ISA §"MMA .m16n8k16 with .f16
// floating-point type"). With that, we can:
//   * Do row-softmax over S in registers via `__shfl_xor_sync` within the
//     4-lane t-group that holds each row.
//   * Rescale the per-lane O accumulator by alpha with a plain scalar
//     multiply on registers — no shared trip at all.
//   * Repack P → A-fragment in registers (P-chunk-n's D layout maps
//     directly onto the A-fragment of the next mma).
//
// Fragment layout summary (m16n8k16, FP16 A/B, FP32 D, A row-major, B col-major):
//
//   Lane (q, t) where q = laneIdx / 4, t = laneIdx % 4.
//
//   D / C [16 × 8] FP32: 4 floats per lane, addressing
//       d[0] = (q,   2t)     d[1] = (q,   2t+1)
//       d[2] = (q+8, 2t)     d[3] = (q+8, 2t+1)
//     So each row r ∈ [0,16) is owned by 4 lanes (one t-group). For r ∈
//     [0,8) it lives in d[0,1] of lanes with q=r; for r ∈ [8,16) in d[2,3]
//     of lanes with q=r-8.
//
//   A [16 × 16] FP16: 4 b32 regs per lane (a0..a3), packing 8 FP16:
//       a0 = pair (q,   2t..2t+1)   a1 = pair (q+8, 2t..2t+1)
//       a2 = pair (q,   2t+8..2t+9) a3 = pair (q+8, 2t+8..2t+9)
//
//   B [16 × 8] FP16, col-major: 2 b32 regs per lane (b0, b1):
//       b0 = pair (2t..2t+1,   g)  b1 = pair (2t+8..2t+9, g)
//
// Tile shape (chosen to match M10.3 for an apples-to-apples comparison):
//
//   BR = 16 (rows of Q per warp, one mma.m16n8k16 A-tile row)
//   BC = 32 (cols of K/V per inner tile, 4 mma n-tiles wide)
//   D  = 64 (4 mma k-tiles in the Q · K^T reduction; 8 mma n-tiles in P · V)
//   WARPS = 4 per block → BR_BLOCK = 64 Q rows per block
//   STAGES = 2 cp.async double-buffer (reused from M10.3)
//
// Per-warp register state (each lane):
//   o_frag[8][4]   FP32 — running output O, 16x64 (8 n-tiles × 4 elems/lane)
//   m_state[2]     FP32 — running max for the 2 rows this lane owns (q, q+8)
//   l_state[2]     FP32 — running sum for the 2 rows this lane owns
//
// Per-tile transient register state (each lane):
//   s_frag[4][4]   FP32 — S = Q·K^T scaled, 16x32
//   p_packed[2][4] b32  — P FP16 packed as A-fragment for P·V, 16x32 split
//                         into 2 chunks of 16 (each is an A-input for mma)
//
// Online softmax in registers, per K-tile:
//   1. Compute S = Q · K^T with 16 mma.sync (4 k-chunks × 4 n-chunks).
//      Scale S by 1/sqrt(D).
//   2. Mask OOB cols (last tile): for each lane, set s[n_chunk][i] = -INF
//      where the corresponding (q', col) has col ≥ N.
//   3. Per-row max: for each of the 2 rows this lane owns (q and q+8),
//      reduce across the 4 t-group lanes via __shfl_xor_sync(., ., 1) then
//      __shfl_xor_sync(., ., 2). All 4 lanes in the group end with the same
//      row_max.
//   4. Combine with running (m_state, l_state) → (m_new, alpha) per row.
//   5. Apply alpha to the corresponding O slot in registers (no shared!):
//        o_frag[d_chunk][0,1] *= alpha[row q]      // row q
//        o_frag[d_chunk][2,3] *= alpha[row q+8]    // row q+8
//   6. Compute P = exp(S - m_new), per-element. Pack into A-fragment for
//      the next mma. Reduce row-sum of P across the t-group → l_ij.
//   7. l_state = alpha * l_state + l_ij.
//   8. O += P · V with 16 mma.sync (2 p-chunks × 8 d-chunks).
//
// At the end, divide each lane's O slots by its l_state.
//
// We keep the cp.async double-buffer pipeline from M10.3 since it's
// orthogonal to the softmax change and free perf.
// ============================================================================

constexpr int BR4         = 16;
constexpr int BC4         = 32;
constexpr int D4          = 64;
constexpr int WARPS_M10_4 = 4;
constexpr int BR4_BLOCK   = BR4 * WARPS_M10_4;          // 64
constexpr int M10_4_THREADS = WARPS_M10_4 * 32;         // 128
constexpr int STAGES_M10_4 = 2;

constexpr int N_CHUNKS_S = BC4 / 8;     // 4 (mma n-tiles wide in S)
constexpr int K_CHUNKS_S = D4 / 16;     // 4 (mma k-tiles in Q·K^T)
constexpr int D_CHUNKS_O = D4 / 8;      // 8 (mma n-tiles wide in O)
constexpr int P_CHUNKS   = BC4 / 16;    // 2 (P split into A-frag chunks for P·V)

// pack two halves into a b32 (low = x, high = y).
__device__ __forceinline__ uint32_t pack_halves(__half x, __half y) {
    uint32_t lo = *reinterpret_cast<uint16_t*>(&x);
    uint32_t hi = *reinterpret_cast<uint16_t*>(&y);
    return lo | (hi << 16);
}
__device__ __forceinline__ uint32_t pack_half2(__half2 v) {
    return *reinterpret_cast<uint32_t*>(&v);
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
}

__device__ __forceinline__ void m10_4_cp_async_16(void* smem_dst, const void* gmem_src) {
    __pipeline_memcpy_async(smem_dst, gmem_src, sizeof(int4));
}

__global__ void flash_attention_mma(const __half* __restrict__ Q,
                                    const __half* __restrict__ K,
                                    const __half* __restrict__ V,
                                    float*        __restrict__ O,
                                    int N) {
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int q_lane  = lane >> 2;    // 0..7
    int t_lane  = lane & 3;     // 0..3

    int row_base = blockIdx.x * BR4_BLOCK + warp_id * BR4;

    // The 2 rows of D / C / softmax-output this lane owns are q_lane and q_lane+8.
    int row_top = row_base + q_lane;       // for d[0], d[1]
    int row_bot = row_base + q_lane + 8;   // for d[2], d[3]

    // Block-shared: K/V double-buffered tiles.
    __shared__ __half Ks[STAGES_M10_4][BC4 * D4];
    __shared__ __half Vs[STAGES_M10_4][BC4 * D4];

    // Per-warp shared: Q (16×64 fp16). Loaded once at kernel entry.
    __shared__ __half Qs_blk[WARPS_M10_4][BR4 * D4];
    __half* Qs = Qs_blk[warp_id];

    // ---- Stage Q for this warp ----
    for (int idx = lane; idx < BR4 * D4; idx += 32) {
        int r = idx / D4;
        int d = idx % D4;
        int row = row_base + r;
        Qs[r * D4 + d] = (row < N) ? Q[row * D4 + d] : __float2half(0.0f);
    }
    __syncwarp();

    // ---- Pre-load Q A-fragments for all 4 k-chunks ----
    // q_frag[k_chunk][0..3] hold a0..a3 per the layout above.
    uint32_t q_frag[K_CHUNKS_S][4];
    #pragma unroll
    for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
        // Read 4 __half2 per lane. Q rows q_lane and q_lane+8, cols
        // kk*16+2t..kk*16+2t+1 and kk*16+2t+8..kk*16+2t+9.
        __half2 a01 = *reinterpret_cast<__half2*>(
            &Qs[q_lane * D4 + kk * 16 + 2 * t_lane]);
        __half2 a23 = *reinterpret_cast<__half2*>(
            &Qs[(q_lane + 8) * D4 + kk * 16 + 2 * t_lane]);
        __half2 a45 = *reinterpret_cast<__half2*>(
            &Qs[q_lane * D4 + kk * 16 + 2 * t_lane + 8]);
        __half2 a67 = *reinterpret_cast<__half2*>(
            &Qs[(q_lane + 8) * D4 + kk * 16 + 2 * t_lane + 8]);
        q_frag[kk][0] = pack_half2(a01);
        q_frag[kk][1] = pack_half2(a23);
        q_frag[kk][2] = pack_half2(a45);
        q_frag[kk][3] = pack_half2(a67);
    }

    // ---- O accumulator: 8 d-chunks × 4 floats per lane ----
    float o_frag[D_CHUNKS_O][4];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        o_frag[dc][0] = 0.0f; o_frag[dc][1] = 0.0f;
        o_frag[dc][2] = 0.0f; o_frag[dc][3] = 0.0f;
    }

    // ---- Per-row state (2 rows per lane) ----
    float m_state[2] = {-INFINITY, -INFINITY};
    float l_state[2] = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf((float)D4);
    int Tc = (N + BC4 - 1) / BC4;

    // ---- cp.async issuance helper for one tile ----
    // BC4 * D4 = 32 * 64 = 2048 halves = 4096 B per K-tile (and per V-tile).
    // With 128 threads * 16 B per cp.async = 2048 B per pass → 2 passes per tile.
    auto issue_tile = [&](int stage, int j, bool may_be_oob) {
        constexpr int VEC_HALVES   = 8;             // 16 B = 8 fp16
        constexpr int VECS_PER_ROW = D4 / VEC_HALVES;       // 8
        constexpr int N_TRANSFERS  = (BC4 * D4) / VEC_HALVES; // 256
        constexpr int PASSES       = N_TRANSFERS / M10_4_THREADS; // 2

        #pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            int t = p * M10_4_THREADS + threadIdx.x;
            int row_in_tile = t / VECS_PER_ROW;
            int col_in_row  = (t % VECS_PER_ROW) * VEC_HALVES;
            int kcol = j * BC4 + row_in_tile;
            int src_row = may_be_oob ? ((kcol < N) ? kcol : (N - 1)) : kcol;

            __half*       k_smem = &Ks[stage][row_in_tile * D4 + col_in_row];
            __half*       v_smem = &Vs[stage][row_in_tile * D4 + col_in_row];
            const __half* k_gmem = &K[src_row * D4 + col_in_row];
            const __half* v_gmem = &V[src_row * D4 + col_in_row];

            m10_4_cp_async_16(k_smem, k_gmem);
            m10_4_cp_async_16(v_smem, v_gmem);
        }
    };

    // ---- Prologue: kick off tile 0 ----
    if (Tc > 0) issue_tile(0, 0, /*may_be_oob=*/(Tc == 1));
    __pipeline_commit();

    int compute_stage = 0;
    int load_stage    = 1 % STAGES_M10_4;

    for (int j = 0; j < Tc; ++j) {
        // Next-tile prefetch.
        if (j + 1 < Tc) {
            issue_tile(load_stage, j + 1, /*may_be_oob=*/(j + 2 == Tc));
        }
        __pipeline_commit();

        __pipeline_wait_prior(STAGES_M10_4 - 1);
        __syncthreads();

        __half* Ks_cur = Ks[compute_stage];
        __half* Vs_cur = Vs[compute_stage];

        // ---------------- S = Q · K^T (in registers) ----------------
        // s_frag[n_chunk][i]: lane (q, t) holds D-output of mma n=n_chunk:
        //   s[n_chunk][0] = (q, 2t)      s[n_chunk][1] = (q, 2t+1)
        //   s[n_chunk][2] = (q+8, 2t)    s[n_chunk][3] = (q+8, 2t+1)
        // s S-column indexed by (n_chunk, 2t/2t+1) → S col = n_chunk*8 + 2t/2t+1.
        float s_frag[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            s_frag[n][0] = 0.0f; s_frag[n][1] = 0.0f;
            s_frag[n][2] = 0.0f; s_frag[n][3] = 0.0f;
        }

        #pragma unroll
        for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
            #pragma unroll
            for (int n = 0; n < N_CHUNKS_S; ++n) {
                // B = K^T 16×8: lane holds b0 = K[n*8 + g, kk*16 + 2t..2t+1],
                //                          b1 = K[n*8 + g, kk*16 + 2t+8..2t+9].
                int k_row = n * 8 + q_lane;
                __half2 b0v = *reinterpret_cast<__half2*>(
                    &Ks_cur[k_row * D4 + kk * 16 + 2 * t_lane]);
                __half2 b1v = *reinterpret_cast<__half2*>(
                    &Ks_cur[k_row * D4 + kk * 16 + 2 * t_lane + 8]);
                uint32_t b0 = pack_half2(b0v);
                uint32_t b1 = pack_half2(b1v);

                mma_m16n8k16(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
                             q_frag[kk][0], q_frag[kk][1], q_frag[kk][2], q_frag[kk][3],
                             b0, b1,
                             s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
            }
        }

        // ---- Scale by 1/sqrt(D), apply OOB mask ----
        // Lane (q, t)'s S col for slot i ∈ [0,3]:
        //   i=0,2: col = n_chunk*8 + 2*t
        //   i=1,3: col = n_chunk*8 + 2*t + 1
        // Mask only on the last tile.
        bool last_tile = (j == Tc - 1);
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            int col0 = n * 8 + 2 * t_lane;
            int col1 = col0 + 1;
            int kcol0 = j * BC4 + col0;
            int kcol1 = j * BC4 + col1;
            bool m0 = last_tile && (kcol0 >= N);
            bool m1 = last_tile && (kcol1 >= N);
            s_frag[n][0] = m0 ? -INFINITY : (s_frag[n][0] * scale);
            s_frag[n][1] = m1 ? -INFINITY : (s_frag[n][1] * scale);
            s_frag[n][2] = m0 ? -INFINITY : (s_frag[n][2] * scale);
            s_frag[n][3] = m1 ? -INFINITY : (s_frag[n][3] * scale);
        }

        // ---------------- Row max via __shfl_xor in the t-group ----------------
        // Each row of S is held across 4 lanes (same q_lane, t_lane = 0..3).
        // For each row, the 8 values held by the 4 lanes (2 per lane per
        // n_chunk × 4 n_chunks = 8 per lane) are reduced.
        // row 0 = row "top" = q_lane; uses s_frag[*][0,1]
        // row 1 = row "bot" = q_lane + 8; uses s_frag[*][2,3]

        float row_max_top = -INFINITY;
        float row_max_bot = -INFINITY;
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            row_max_top = fmaxf(row_max_top, fmaxf(s_frag[n][0], s_frag[n][1]));
            row_max_bot = fmaxf(row_max_bot, fmaxf(s_frag[n][2], s_frag[n][3]));
        }
        // XOR-reduce across the 4-lane t-group.
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 1));
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 2));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 1));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 2));

        // Online softmax combine (two rows per lane).
        float m_new_top = fmaxf(m_state[0], row_max_top);
        float m_new_bot = fmaxf(m_state[1], row_max_bot);
        float alpha_top = (m_state[0] == -INFINITY) ? 0.0f : __expf(m_state[0] - m_new_top);
        float alpha_bot = (m_state[1] == -INFINITY) ? 0.0f : __expf(m_state[1] - m_new_bot);

        // ---------------- Apply alpha to per-lane O (REGISTER-RESIDENT) ----------------
        // o_frag[dc][0,1] are row "top" (q_lane). [2,3] are row "bot" (q_lane+8).
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
            o_frag[dc][0] *= alpha_top;
            o_frag[dc][1] *= alpha_top;
            o_frag[dc][2] *= alpha_bot;
            o_frag[dc][3] *= alpha_bot;
        }

        // ---------------- Compute P = exp(S - m_new), per-element row-sum ----------------
        float l_ij_top = 0.0f;
        float l_ij_bot = 0.0f;
        // p_vals[n_chunk][i]: P in fp32 form.
        float p_vals[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            float p0 = (s_frag[n][0] == -INFINITY) ? 0.0f : __expf(s_frag[n][0] - m_new_top);
            float p1 = (s_frag[n][1] == -INFINITY) ? 0.0f : __expf(s_frag[n][1] - m_new_top);
            float p2 = (s_frag[n][2] == -INFINITY) ? 0.0f : __expf(s_frag[n][2] - m_new_bot);
            float p3 = (s_frag[n][3] == -INFINITY) ? 0.0f : __expf(s_frag[n][3] - m_new_bot);
            p_vals[n][0] = p0; p_vals[n][1] = p1;
            p_vals[n][2] = p2; p_vals[n][3] = p3;
            l_ij_top += p0 + p1;
            l_ij_bot += p2 + p3;
        }
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 1);
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 2);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 1);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 2);

        l_state[0] = alpha_top * l_state[0] + l_ij_top;
        l_state[1] = alpha_bot * l_state[1] + l_ij_bot;
        m_state[0] = m_new_top;
        m_state[1] = m_new_bot;

        // ---------------- Repack P into A-fragment layout for P·V ----------------
        // P is 16 × BC4 = 16 × 32. Split into BC4/16 = 2 chunks of 16 cols each.
        // Each chunk is an mma.A operand 16×16. For chunk pc:
        //   - P_cols [pc*16 .. pc*16+15] map to S n_chunks {2*pc, 2*pc+1}.
        //   - Lane (q, t) for the A-fragment: 4 b32 regs (8 fp16).
        //     a0 = pair (q,   2t..2t+1)   ← from n_chunk = 2*pc, slots p[0], p[1]
        //     a1 = pair (q+8, 2t..2t+1)   ← from n_chunk = 2*pc, slots p[2], p[3]
        //     a2 = pair (q,   2t+8..2t+9) ← from n_chunk = 2*pc+1, slots p[0], p[1]
        //     a3 = pair (q+8, 2t+8..2t+9) ← from n_chunk = 2*pc+1, slots p[2], p[3]
        //
        // So the D-fragment layout of the S mma's n_chunks {2pc, 2pc+1} maps
        // directly onto the A-fragment of the next mma. NO cross-lane shuffle.
        uint32_t p_frag[P_CHUNKS][4];
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            int n0 = 2 * pc;
            int n1 = 2 * pc + 1;
            p_frag[pc][0] = pack_halves(__float2half(p_vals[n0][0]), __float2half(p_vals[n0][1]));
            p_frag[pc][1] = pack_halves(__float2half(p_vals[n0][2]), __float2half(p_vals[n0][3]));
            p_frag[pc][2] = pack_halves(__float2half(p_vals[n1][0]), __float2half(p_vals[n1][1]));
            p_frag[pc][3] = pack_halves(__float2half(p_vals[n1][2]), __float2half(p_vals[n1][3]));
        }

        // ---------------- O += P · V ----------------
        // V is 32×64. Reduce dim = 32 (P col / V row). Split into 2 chunks of 16.
        // Output dim = 64. Split into 8 chunks of 8 (mma n).
        //
        // For each (p_chunk pc, output chunk dc):
        //   A = p_frag[pc] (16×16)
        //   B = V[pc*16..pc*16+15 rows, dc*8..dc*8+7 cols] read as col-major
        //   Accumulate into o_frag[dc] in place.
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
                int v_col = dc * 8 + q_lane;
                int v_base = pc * 16;
                // B fragment for m16n8k16: lane (q, t) holds:
                //   b0 = pair (k=2t..2t+1, n=g)
                //   b1 = pair (k=2t+8..2t+9, n=g)
                // n = q_lane → output col index within chunk = q_lane.
                // k = reduce row → V row in chunk = pc*16 + k.
                __half b0_lo = Vs_cur[(v_base + 2 * t_lane    ) * D4 + v_col];
                __half b0_hi = Vs_cur[(v_base + 2 * t_lane + 1) * D4 + v_col];
                __half b1_lo = Vs_cur[(v_base + 2 * t_lane + 8) * D4 + v_col];
                __half b1_hi = Vs_cur[(v_base + 2 * t_lane + 9) * D4 + v_col];
                uint32_t b0 = pack_halves(b0_lo, b0_hi);
                uint32_t b1 = pack_halves(b1_lo, b1_hi);

                mma_m16n8k16(o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3],
                             p_frag[pc][0], p_frag[pc][1], p_frag[pc][2], p_frag[pc][3],
                             b0, b1,
                             o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3]);
            }
        }

        // WAR fence: under STAGES=2, next iter's cp.async writes the buffer
        // we just consumed.
        __syncthreads();

        compute_stage = (compute_stage + 1) % STAGES_M10_4;
        load_stage    = (load_stage    + 1) % STAGES_M10_4;
    }

    // ---------------- Normalize and write O ----------------
    // Lane (q, t) owns:
    //   o_frag[dc][0,1] for row row_top, cols dc*8 + 2t and 2t+1
    //   o_frag[dc][2,3] for row row_bot, cols dc*8 + 2t and 2t+1
    float inv_l_top = (l_state[0] > 0.0f) ? (1.0f / l_state[0]) : 0.0f;
    float inv_l_bot = (l_state[1] > 0.0f) ? (1.0f / l_state[1]) : 0.0f;

    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        int col0 = dc * 8 + 2 * t_lane;
        int col1 = col0 + 1;
        if (row_top < N) {
            O[row_top * D4 + col0] = o_frag[dc][0] * inv_l_top;
            O[row_top * D4 + col1] = o_frag[dc][1] * inv_l_top;
        }
        if (row_bot < N) {
            O[row_bot * D4 + col0] = o_frag[dc][2] * inv_l_bot;
            O[row_bot * D4 + col1] = o_frag[dc][3] * inv_l_bot;
        }
    }
}

inline void launch_flash_mma(const __half* Q, const __half* K, const __half* V,
                             float* O, int N) {
    dim3 block(M10_4_THREADS);
    dim3 grid((N + BR4_BLOCK - 1) / BR4_BLOCK);
    flash_attention_mma<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// M10.5 — Raw mma.sync + ldmatrix fragment loads (the FA-2 / CUTLASS shape).
//
// Identical algorithm and tile shape to M10.4. Only the shared→register
// fragment load paths change:
//   - Q (A operand of S = Q · K^T): replaced 4 manual __half2 loads per lane
//     per k-chunk with one `ldmatrix.sync.aligned.m8n8.x4.shared.b16`.
//   - V (B operand of O += P · V): replaced 4 scalar __half loads per lane
//     per (p_chunk, d_chunk) with one `ldmatrix.sync.aligned.m8n8.x2.trans
//     .shared.b16`. The `.trans` variant transposes each 8×8 tile during the
//     load, turning V's row-major rows into the col-major-fragment layout
//     mma.sync wants on the B side.
//
// K is left on the manual __half2 path: K^T as B is read with a layout where
// the manual code is already two __half2 reads per lane (clean and fast), so
// it's not the slow load. V is the win here.
//
// Lane → ldmatrix pointer mapping (32 lanes, x4):
//   lane L provides one row address. Tile (L/8) holds 8 rows; the lane's row
//   within that tile is (L%8). For a 16×16 A fragment laid out in shared as
//   16 rows × 16 fp16 (row-major, 32 B/row), the four 8×8 tiles are:
//     tile 0 (a0): rows  0..7,  cols 0..7
//     tile 1 (a1): rows  8..15, cols 0..7
//     tile 2 (a2): rows  0..7,  cols 8..15
//     tile 3 (a3): rows  8..15, cols 8..15
//   After the load, lane (q=L/4, t=L%4) holds in r0..r3 the 2-fp16 pairs
//   {tile, row q, cols 2t..2t+1} — exactly the mma m16n8k16 A layout.
//
// For ldmatrix.x2.trans on V (B fragment for P·V): B is 16 rows × 8 cols
// (col-major). With .trans, ldmatrix transposes each of the 2 input 8×8
// tiles read row-major from shared. The 2 tiles correspond to B rows 0..7
// and rows 8..15. Lane L (0..15) provides &V_tile[(pc*16 + L) * D4 + dc*8]
// (a 16-byte aligned row pointer); lanes 16..31's addresses are unused
// (we still pass a valid address for safety). Lane (q, t) ends up with
// r0 = pair (rows 2t..2t+1, col q), r1 = pair (rows 2t+8..2t+9, col q).
// ============================================================================

// Issue ldmatrix.x4 on an A-fragment in shared memory laid out as 16 rows ×
// 16 fp16 starting at `tile_base`. Lane L provides the row pointer for its
// tile. Outputs r0..r3 follow the mma m16n8k16 A layout described above.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            const __half* tile_base,
                                            int row_stride_halves) {
    int lane = threadIdx.x & 31;
    int tile = lane >> 3;          // 0..3 = which 8×8 tile this lane addresses
    int row  = lane & 7;           // 0..7 = row within that tile
    // Tile (col_block, row_block): tile 0 = (rows 0-7, cols 0-7), tile 1 = (rows 8-15, cols 0-7),
    // tile 2 = (rows 0-7, cols 8-15), tile 3 = (rows 8-15, cols 8-15).
    int row_block = tile & 1;      // 0 or 1 → rows 0-7 or rows 8-15
    int col_block = tile >> 1;     // 0 or 1 → cols 0-7 or cols 8-15
    const __half* row_addr = tile_base
        + (row_block * 8 + row) * row_stride_halves
        + col_block * 8;
    uint32_t smem_int = __cvta_generic_to_shared(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        :  "r"(smem_int));
}

// Issue ldmatrix.x2.trans on a B-fragment in shared memory laid out as 16
// rows × 8 fp16 starting at `tile_base` (row-major in shared). With .trans,
// lane (q=L/4, t=L%4) gets r0 = (rows 2t..2t+1, col q), r1 = (rows 2t+8..2t+9,
// col q) — the mma m16n8k16 B layout.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1,
                                                  const __half* tile_base,
                                                  int row_stride_halves) {
    int lane = threadIdx.x & 31;
    // For x2: lanes 0..7 provide rows of tile 0 (B rows 0..7), lanes 8..15
    // provide rows of tile 1 (B rows 8..15). Lanes 16..31 are unused; we
    // pass them lane 0's address for safety.
    int eff = lane & 15;           // 0..15
    const __half* row_addr = tile_base + eff * row_stride_halves;
    uint32_t smem_int = __cvta_generic_to_shared(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        :  "r"(smem_int));
}

__global__ void flash_attention_mma_ldmatrix(const __half* __restrict__ Q,
                                             const __half* __restrict__ K,
                                             const __half* __restrict__ V,
                                             float*        __restrict__ O,
                                             int N) {
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int q_lane  = lane >> 2;    // 0..7
    int t_lane  = lane & 3;     // 0..3

    int row_base = blockIdx.x * BR4_BLOCK + warp_id * BR4;

    int row_top = row_base + q_lane;
    int row_bot = row_base + q_lane + 8;

    __shared__ __half Ks[STAGES_M10_4][BC4 * D4];
    __shared__ __half Vs[STAGES_M10_4][BC4 * D4];

    __shared__ __half Qs_blk[WARPS_M10_4][BR4 * D4];
    __half* Qs = Qs_blk[warp_id];

    // ---- Stage Q for this warp ----
    for (int idx = lane; idx < BR4 * D4; idx += 32) {
        int r = idx / D4;
        int d = idx % D4;
        int row = row_base + r;
        Qs[r * D4 + d] = (row < N) ? Q[row * D4 + d] : __float2half(0.0f);
    }
    __syncwarp();

    // ---- Pre-load Q A-fragments for all 4 k-chunks via ldmatrix.x4 ----
    uint32_t q_frag[K_CHUNKS_S][4];
    #pragma unroll
    for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
        // 16×16 A fragment lives at Qs[0..15 rows, kk*16..kk*16+15 cols].
        ldmatrix_x4(q_frag[kk][0], q_frag[kk][1], q_frag[kk][2], q_frag[kk][3],
                    &Qs[kk * 16], D4);
    }

    // ---- O accumulator ----
    float o_frag[D_CHUNKS_O][4];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        o_frag[dc][0] = 0.0f; o_frag[dc][1] = 0.0f;
        o_frag[dc][2] = 0.0f; o_frag[dc][3] = 0.0f;
    }

    float m_state[2] = {-INFINITY, -INFINITY};
    float l_state[2] = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf((float)D4);
    int Tc = (N + BC4 - 1) / BC4;

    auto issue_tile = [&](int stage, int j, bool may_be_oob) {
        constexpr int VEC_HALVES   = 8;
        constexpr int VECS_PER_ROW = D4 / VEC_HALVES;
        constexpr int N_TRANSFERS  = (BC4 * D4) / VEC_HALVES;
        constexpr int PASSES       = N_TRANSFERS / M10_4_THREADS;

        #pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            int t = p * M10_4_THREADS + threadIdx.x;
            int row_in_tile = t / VECS_PER_ROW;
            int col_in_row  = (t % VECS_PER_ROW) * VEC_HALVES;
            int kcol = j * BC4 + row_in_tile;
            int src_row = may_be_oob ? ((kcol < N) ? kcol : (N - 1)) : kcol;

            __half*       k_smem = &Ks[stage][row_in_tile * D4 + col_in_row];
            __half*       v_smem = &Vs[stage][row_in_tile * D4 + col_in_row];
            const __half* k_gmem = &K[src_row * D4 + col_in_row];
            const __half* v_gmem = &V[src_row * D4 + col_in_row];

            m10_4_cp_async_16(k_smem, k_gmem);
            m10_4_cp_async_16(v_smem, v_gmem);
        }
    };

    if (Tc > 0) issue_tile(0, 0, /*may_be_oob=*/(Tc == 1));
    __pipeline_commit();

    int compute_stage = 0;
    int load_stage    = 1 % STAGES_M10_4;

    for (int j = 0; j < Tc; ++j) {
        if (j + 1 < Tc) {
            issue_tile(load_stage, j + 1, /*may_be_oob=*/(j + 2 == Tc));
        }
        __pipeline_commit();

        __pipeline_wait_prior(STAGES_M10_4 - 1);
        __syncthreads();

        __half* Ks_cur = Ks[compute_stage];
        __half* Vs_cur = Vs[compute_stage];

        // ---------------- S = Q · K^T ----------------
        // K (as B = K^T) load stays on the manual __half2 path — already
        // a clean two-loads-per-lane pattern (K^T's B layout matches
        // contiguous 32-bit groups in row-major K).
        float s_frag[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            s_frag[n][0] = 0.0f; s_frag[n][1] = 0.0f;
            s_frag[n][2] = 0.0f; s_frag[n][3] = 0.0f;
        }

        #pragma unroll
        for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
            #pragma unroll
            for (int n = 0; n < N_CHUNKS_S; ++n) {
                int k_row = n * 8 + q_lane;
                __half2 b0v = *reinterpret_cast<__half2*>(
                    &Ks_cur[k_row * D4 + kk * 16 + 2 * t_lane]);
                __half2 b1v = *reinterpret_cast<__half2*>(
                    &Ks_cur[k_row * D4 + kk * 16 + 2 * t_lane + 8]);
                uint32_t b0 = pack_half2(b0v);
                uint32_t b1 = pack_half2(b1v);

                mma_m16n8k16(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
                             q_frag[kk][0], q_frag[kk][1], q_frag[kk][2], q_frag[kk][3],
                             b0, b1,
                             s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
            }
        }

        // ---- Scale + OOB mask ----
        bool last_tile = (j == Tc - 1);
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            int col0 = n * 8 + 2 * t_lane;
            int col1 = col0 + 1;
            int kcol0 = j * BC4 + col0;
            int kcol1 = j * BC4 + col1;
            bool m0 = last_tile && (kcol0 >= N);
            bool m1 = last_tile && (kcol1 >= N);
            s_frag[n][0] = m0 ? -INFINITY : (s_frag[n][0] * scale);
            s_frag[n][1] = m1 ? -INFINITY : (s_frag[n][1] * scale);
            s_frag[n][2] = m0 ? -INFINITY : (s_frag[n][2] * scale);
            s_frag[n][3] = m1 ? -INFINITY : (s_frag[n][3] * scale);
        }

        // ---- Row max ----
        float row_max_top = -INFINITY;
        float row_max_bot = -INFINITY;
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            row_max_top = fmaxf(row_max_top, fmaxf(s_frag[n][0], s_frag[n][1]));
            row_max_bot = fmaxf(row_max_bot, fmaxf(s_frag[n][2], s_frag[n][3]));
        }
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 1));
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 2));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 1));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 2));

        float m_new_top = fmaxf(m_state[0], row_max_top);
        float m_new_bot = fmaxf(m_state[1], row_max_bot);
        float alpha_top = (m_state[0] == -INFINITY) ? 0.0f : __expf(m_state[0] - m_new_top);
        float alpha_bot = (m_state[1] == -INFINITY) ? 0.0f : __expf(m_state[1] - m_new_bot);

        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
            o_frag[dc][0] *= alpha_top;
            o_frag[dc][1] *= alpha_top;
            o_frag[dc][2] *= alpha_bot;
            o_frag[dc][3] *= alpha_bot;
        }

        float l_ij_top = 0.0f;
        float l_ij_bot = 0.0f;
        float p_vals[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            float p0 = (s_frag[n][0] == -INFINITY) ? 0.0f : __expf(s_frag[n][0] - m_new_top);
            float p1 = (s_frag[n][1] == -INFINITY) ? 0.0f : __expf(s_frag[n][1] - m_new_top);
            float p2 = (s_frag[n][2] == -INFINITY) ? 0.0f : __expf(s_frag[n][2] - m_new_bot);
            float p3 = (s_frag[n][3] == -INFINITY) ? 0.0f : __expf(s_frag[n][3] - m_new_bot);
            p_vals[n][0] = p0; p_vals[n][1] = p1;
            p_vals[n][2] = p2; p_vals[n][3] = p3;
            l_ij_top += p0 + p1;
            l_ij_bot += p2 + p3;
        }
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 1);
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 2);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 1);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 2);

        l_state[0] = alpha_top * l_state[0] + l_ij_top;
        l_state[1] = alpha_bot * l_state[1] + l_ij_bot;
        m_state[0] = m_new_top;
        m_state[1] = m_new_bot;

        // ---- Repack P into A-fragment layout (same as M10.4) ----
        uint32_t p_frag[P_CHUNKS][4];
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            int n0 = 2 * pc;
            int n1 = 2 * pc + 1;
            p_frag[pc][0] = pack_halves(__float2half(p_vals[n0][0]), __float2half(p_vals[n0][1]));
            p_frag[pc][1] = pack_halves(__float2half(p_vals[n0][2]), __float2half(p_vals[n0][3]));
            p_frag[pc][2] = pack_halves(__float2half(p_vals[n1][0]), __float2half(p_vals[n1][1]));
            p_frag[pc][3] = pack_halves(__float2half(p_vals[n1][2]), __float2half(p_vals[n1][3]));
        }

        // ---------------- O += P · V (V load via ldmatrix.x2.trans) ----------------
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
                // 16 rows × 8 cols sub-tile of V at (rows pc*16..pc*16+15, cols dc*8..dc*8+7).
                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1, &Vs_cur[pc * 16 * D4 + dc * 8], D4);

                mma_m16n8k16(o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3],
                             p_frag[pc][0], p_frag[pc][1], p_frag[pc][2], p_frag[pc][3],
                             b0, b1,
                             o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3]);
            }
        }

        __syncthreads();

        compute_stage = (compute_stage + 1) % STAGES_M10_4;
        load_stage    = (load_stage    + 1) % STAGES_M10_4;
    }

    // ---- Normalize and write O ----
    float inv_l_top = (l_state[0] > 0.0f) ? (1.0f / l_state[0]) : 0.0f;
    float inv_l_bot = (l_state[1] > 0.0f) ? (1.0f / l_state[1]) : 0.0f;

    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        int col0 = dc * 8 + 2 * t_lane;
        int col1 = col0 + 1;
        if (row_top < N) {
            O[row_top * D4 + col0] = o_frag[dc][0] * inv_l_top;
            O[row_top * D4 + col1] = o_frag[dc][1] * inv_l_top;
        }
        if (row_bot < N) {
            O[row_bot * D4 + col0] = o_frag[dc][2] * inv_l_bot;
            O[row_bot * D4 + col1] = o_frag[dc][3] * inv_l_bot;
        }
    }
}

inline void launch_flash_mma_ldmatrix(const __half* Q, const __half* K, const __half* V,
                                      float* O, int N) {
    dim3 block(M10_4_THREADS);
    dim3 grid((N + BR4_BLOCK - 1) / BR4_BLOCK);
    flash_attention_mma_ldmatrix<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// M10.6 — flash_attention_mma_swizzled
// ============================================================================
// Same tile shape, mma.sync, ldmatrix.x4 for Q, ldmatrix.x2.trans for V, and
// cp.async double-buffer as M10.5. The one change: K/V/Q shared tiles use an
// XOR-permuted column layout so that the ldmatrix lane→pointer pattern hits
// 8 different bank-groups instead of all stacking on the same column.
//
// Layout (per row r, intra-row half-element offset c ∈ [0, D)):
//   chunk        = c >> 3                       // 16-byte chunk index (0..7)
//   chunk_swizz  = chunk ^ (r & 7)
//   c_swizz      = (c & 7) | (chunk_swizz << 3)
//   stored at:   smem[r * D + c_swizz]
//
// Applied symmetrically to writes (cp.async) and reads (ldmatrix / __half2),
// so values are consistent regardless of permutation. Within any group of
// 8 consecutive rows r..r+7, the swizzle is a bijection on chunks, so 8 lanes
// reading the same column index across those rows touch 8 distinct chunks.
// ============================================================================
__device__ __forceinline__ int m10_6_swizzle(int row, int col_halves) {
    // 16-byte chunk = 8 halves. chunk_idx = col_halves >> 3 (0..7 for D=64).
    int chunk   = col_halves >> 3;
    int chunk_s = chunk ^ (row & 7);
    return (col_halves & 7) | (chunk_s << 3);
}

__device__ __forceinline__ void m10_6_cp_async_16(void* smem_dst, const void* gmem_src) {
    __pipeline_memcpy_async(smem_dst, gmem_src, sizeof(int4));
}

__global__ void flash_attention_mma_swizzled(const __half* __restrict__ Q,
                                             const __half* __restrict__ K,
                                             const __half* __restrict__ V,
                                             float*        __restrict__ O,
                                             int N) {
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int q_lane  = lane >> 2;    // 0..7
    int t_lane  = lane & 3;     // 0..3

    int row_base = blockIdx.x * BR4_BLOCK + warp_id * BR4;

    int row_top = row_base + q_lane;
    int row_bot = row_base + q_lane + 8;

    __shared__ __half Ks[STAGES_M10_4][BC4 * D4];
    __shared__ __half Vs[STAGES_M10_4][BC4 * D4];

    __shared__ __half Qs_blk[WARPS_M10_4][BR4 * D4];
    __half* Qs = Qs_blk[warp_id];

    // ---- Stage Q for this warp (swizzled write) ----
    // BR4 * D4 = 16 * 64 = 1024 halves. 32 lanes → 32 halves/lane. To keep
    // the swizzle aligned on 8-half chunks, each lane writes 8 contiguous
    // halves per pass: 1024 halves / 8 halves/pass / 32 lanes = 4 passes.
    {
        constexpr int VEC_HALVES   = 8;
        constexpr int VECS_PER_ROW = D4 / VEC_HALVES;          // 8
        constexpr int N_VECS       = (BR4 * D4) / VEC_HALVES;  // 128
        #pragma unroll
        for (int p = 0; p < N_VECS / 32; ++p) {
            int t = p * 32 + lane;
            int r = t / VECS_PER_ROW;
            int c = (t % VECS_PER_ROW) * VEC_HALVES;
            int row_global = row_base + r;
            int c_sw = m10_6_swizzle(r, c);
            // Vector half8 load/store via int4.
            int4 v;
            if (row_global < N) {
                v = *reinterpret_cast<const int4*>(&Q[row_global * D4 + c]);
            } else {
                v = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&Qs[r * D4 + c_sw]) = v;
        }
    }
    __syncwarp();

    // ---- Pre-load Q A-fragments via ldmatrix.x4 on swizzled layout ----
    // Each ldmatrix.x4 reads a 16×16 A fragment. Lane L gives a row pointer:
    //   tile      = L >> 3   (0..3)
    //   row_in_t  = L & 7    (0..7)
    //   row_block = tile & 1 (0 or 1: rows 0-7 vs 8-15)
    //   col_block = tile >> 1 (0 or 1: cols 0-7 vs 8-15)
    //   row = row_block * 8 + row_in_t
    //   col_base (logical) = kk * 16 + col_block * 8
    // The hardware then reads 8 contiguous halves from that pointer (= 1 chunk).
    // With swizzle: pointer = &Qs[row * D + swizzle(row, col_base)]. Since
    // col_base is 8-aligned, swizzle(row, col_base) is also 8-aligned, so the
    // 8 read halves are the original logical [col_base..col_base+7] of row.
    uint32_t q_frag[K_CHUNKS_S][4];
    {
        int tile      = lane >> 3;
        int row_in_t  = lane & 7;
        int row_block = tile & 1;
        int col_block = tile >> 1;
        int row = row_block * 8 + row_in_t;
        #pragma unroll
        for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
            int col_base   = kk * 16 + col_block * 8;
            int col_sw     = m10_6_swizzle(row, col_base);
            const __half* row_addr = &Qs[row * D4 + col_sw];
            uint32_t smem_int = __cvta_generic_to_shared(row_addr);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(q_frag[kk][0]), "=r"(q_frag[kk][1]),
                  "=r"(q_frag[kk][2]), "=r"(q_frag[kk][3])
                :  "r"(smem_int));
        }
    }

    // ---- O accumulator ----
    float o_frag[D_CHUNKS_O][4];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        o_frag[dc][0] = 0.0f; o_frag[dc][1] = 0.0f;
        o_frag[dc][2] = 0.0f; o_frag[dc][3] = 0.0f;
    }

    float m_state[2] = {-INFINITY, -INFINITY};
    float l_state[2] = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf((float)D4);
    int Tc = (N + BC4 - 1) / BC4;

    // Each cp.async transfers exactly 16 B = 8 halves = one chunk. The
    // destination column offset within a row is `col_in_row`, swizzled via
    // m10_6_swizzle(row_in_tile, col_in_row). Source (gmem) is unchanged.
    auto issue_tile = [&](int stage, int j, bool may_be_oob) {
        constexpr int VEC_HALVES   = 8;
        constexpr int VECS_PER_ROW = D4 / VEC_HALVES;
        constexpr int N_TRANSFERS  = (BC4 * D4) / VEC_HALVES;
        constexpr int PASSES       = N_TRANSFERS / M10_4_THREADS;

        #pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            int t = p * M10_4_THREADS + threadIdx.x;
            int row_in_tile = t / VECS_PER_ROW;
            int col_in_row  = (t % VECS_PER_ROW) * VEC_HALVES;
            int kcol = j * BC4 + row_in_tile;
            int src_row = may_be_oob ? ((kcol < N) ? kcol : (N - 1)) : kcol;

            int col_sw = m10_6_swizzle(row_in_tile, col_in_row);

            __half*       k_smem = &Ks[stage][row_in_tile * D4 + col_sw];
            __half*       v_smem = &Vs[stage][row_in_tile * D4 + col_sw];
            const __half* k_gmem = &K[src_row * D4 + col_in_row];
            const __half* v_gmem = &V[src_row * D4 + col_in_row];

            m10_6_cp_async_16(k_smem, k_gmem);
            m10_6_cp_async_16(v_smem, v_gmem);
        }
    };

    if (Tc > 0) issue_tile(0, 0, /*may_be_oob=*/(Tc == 1));
    __pipeline_commit();

    int compute_stage = 0;
    int load_stage    = 1 % STAGES_M10_4;

    for (int j = 0; j < Tc; ++j) {
        if (j + 1 < Tc) {
            issue_tile(load_stage, j + 1, /*may_be_oob=*/(j + 2 == Tc));
        }
        __pipeline_commit();

        __pipeline_wait_prior(STAGES_M10_4 - 1);
        __syncthreads();

        __half* Ks_cur = Ks[compute_stage];
        __half* Vs_cur = Vs[compute_stage];

        // ---------------- S = Q · K^T (K read via swizzled __half2) ----------------
        // K is the B operand. Lane (q,t) needs:
        //   b0 = K rows (n*8+q), cols (kk*16 + 2t .. 2t+1)
        //   b1 = K rows (n*8+q), cols (kk*16 + 2t+8 .. 2t+9)
        // i.e., 2 contiguous halves at two column offsets in the same row.
        float s_frag[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            s_frag[n][0] = 0.0f; s_frag[n][1] = 0.0f;
            s_frag[n][2] = 0.0f; s_frag[n][3] = 0.0f;
        }

        #pragma unroll
        for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
            #pragma unroll
            for (int n = 0; n < N_CHUNKS_S; ++n) {
                int k_row = n * 8 + q_lane;
                int c_lo  = kk * 16 + 2 * t_lane;
                int c_hi  = kk * 16 + 2 * t_lane + 8;
                int s_lo  = m10_6_swizzle(k_row, c_lo);
                int s_hi  = m10_6_swizzle(k_row, c_hi);
                __half2 b0v = *reinterpret_cast<__half2*>(&Ks_cur[k_row * D4 + s_lo]);
                __half2 b1v = *reinterpret_cast<__half2*>(&Ks_cur[k_row * D4 + s_hi]);
                uint32_t b0 = pack_half2(b0v);
                uint32_t b1 = pack_half2(b1v);

                mma_m16n8k16(s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3],
                             q_frag[kk][0], q_frag[kk][1], q_frag[kk][2], q_frag[kk][3],
                             b0, b1,
                             s_frag[n][0], s_frag[n][1], s_frag[n][2], s_frag[n][3]);
            }
        }

        // ---- Scale + OOB mask ----
        bool last_tile = (j == Tc - 1);
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            int col0 = n * 8 + 2 * t_lane;
            int col1 = col0 + 1;
            int kcol0 = j * BC4 + col0;
            int kcol1 = j * BC4 + col1;
            bool m0 = last_tile && (kcol0 >= N);
            bool m1 = last_tile && (kcol1 >= N);
            s_frag[n][0] = m0 ? -INFINITY : (s_frag[n][0] * scale);
            s_frag[n][1] = m1 ? -INFINITY : (s_frag[n][1] * scale);
            s_frag[n][2] = m0 ? -INFINITY : (s_frag[n][2] * scale);
            s_frag[n][3] = m1 ? -INFINITY : (s_frag[n][3] * scale);
        }

        // ---- Row max ----
        float row_max_top = -INFINITY;
        float row_max_bot = -INFINITY;
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            row_max_top = fmaxf(row_max_top, fmaxf(s_frag[n][0], s_frag[n][1]));
            row_max_bot = fmaxf(row_max_bot, fmaxf(s_frag[n][2], s_frag[n][3]));
        }
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 1));
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 2));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 1));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 2));

        float m_new_top = fmaxf(m_state[0], row_max_top);
        float m_new_bot = fmaxf(m_state[1], row_max_bot);
        float alpha_top = (m_state[0] == -INFINITY) ? 0.0f : __expf(m_state[0] - m_new_top);
        float alpha_bot = (m_state[1] == -INFINITY) ? 0.0f : __expf(m_state[1] - m_new_bot);

        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
            o_frag[dc][0] *= alpha_top;
            o_frag[dc][1] *= alpha_top;
            o_frag[dc][2] *= alpha_bot;
            o_frag[dc][3] *= alpha_bot;
        }

        float l_ij_top = 0.0f;
        float l_ij_bot = 0.0f;
        float p_vals[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            float p0 = (s_frag[n][0] == -INFINITY) ? 0.0f : __expf(s_frag[n][0] - m_new_top);
            float p1 = (s_frag[n][1] == -INFINITY) ? 0.0f : __expf(s_frag[n][1] - m_new_top);
            float p2 = (s_frag[n][2] == -INFINITY) ? 0.0f : __expf(s_frag[n][2] - m_new_bot);
            float p3 = (s_frag[n][3] == -INFINITY) ? 0.0f : __expf(s_frag[n][3] - m_new_bot);
            p_vals[n][0] = p0; p_vals[n][1] = p1;
            p_vals[n][2] = p2; p_vals[n][3] = p3;
            l_ij_top += p0 + p1;
            l_ij_bot += p2 + p3;
        }
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 1);
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 2);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 1);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 2);

        l_state[0] = alpha_top * l_state[0] + l_ij_top;
        l_state[1] = alpha_bot * l_state[1] + l_ij_bot;
        m_state[0] = m_new_top;
        m_state[1] = m_new_bot;

        // ---- Repack P into A-fragment layout (same as M10.4/10.5) ----
        uint32_t p_frag[P_CHUNKS][4];
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            int n0 = 2 * pc;
            int n1 = 2 * pc + 1;
            p_frag[pc][0] = pack_halves(__float2half(p_vals[n0][0]), __float2half(p_vals[n0][1]));
            p_frag[pc][1] = pack_halves(__float2half(p_vals[n0][2]), __float2half(p_vals[n0][3]));
            p_frag[pc][2] = pack_halves(__float2half(p_vals[n1][0]), __float2half(p_vals[n1][1]));
            p_frag[pc][3] = pack_halves(__float2half(p_vals[n1][2]), __float2half(p_vals[n1][3]));
        }

        // ---------------- O += P · V via ldmatrix.x2.trans on swizzled V ----
        // ldmatrix.x2.trans, lane L (0..15) provides a row pointer to V row
        // (pc*16 + L) at logical column origin (dc*8). With swizzle, lane L's
        // pointer is &Vs[row * D + swizzle(row, dc*8)]. The hardware reads 8
        // halves from each lane (= one chunk = logical cols [dc*8 .. dc*8+7]),
        // then transposes the two 8×8 tiles into b0/b1 in the m16n8k16 B layout.
        //
        // Bank-conflict picture: lanes 0..7 hit 8 *distinct* chunks (because the
        // swizzle XORs by row&7 which cycles 0..7 across those 8 lanes); lanes
        // 8..15 hit the same set of 8 chunks (rows 8..15 mod 8 = 0..7). Net:
        // 2-way conflict instead of 16-way.
        int v_lane = lane & 15;
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            int v_row = pc * 16 + v_lane;
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
                int col_base = dc * 8;
                int col_sw   = m10_6_swizzle(v_row, col_base);
                const __half* row_addr = &Vs_cur[v_row * D4 + col_sw];
                uint32_t smem_int = __cvta_generic_to_shared(row_addr);
                uint32_t b0, b1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b0), "=r"(b1)
                    :  "r"(smem_int));

                mma_m16n8k16(o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3],
                             p_frag[pc][0], p_frag[pc][1], p_frag[pc][2], p_frag[pc][3],
                             b0, b1,
                             o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3]);
            }
        }

        __syncthreads();

        compute_stage = (compute_stage + 1) % STAGES_M10_4;
        load_stage    = (load_stage    + 1) % STAGES_M10_4;
    }

    // ---- Normalize and write O ----
    float inv_l_top = (l_state[0] > 0.0f) ? (1.0f / l_state[0]) : 0.0f;
    float inv_l_bot = (l_state[1] > 0.0f) ? (1.0f / l_state[1]) : 0.0f;

    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        int col0 = dc * 8 + 2 * t_lane;
        int col1 = col0 + 1;
        if (row_top < N) {
            O[row_top * D4 + col0] = o_frag[dc][0] * inv_l_top;
            O[row_top * D4 + col1] = o_frag[dc][1] * inv_l_top;
        }
        if (row_bot < N) {
            O[row_bot * D4 + col0] = o_frag[dc][2] * inv_l_bot;
            O[row_bot * D4 + col1] = o_frag[dc][3] * inv_l_bot;
        }
    }
}

inline void launch_flash_mma_swizzled(const __half* Q, const __half* K, const __half* V,
                                      float* O, int N) {
    dim3 block(M10_4_THREADS);
    dim3 grid((N + BR4_BLOCK - 1) / BR4_BLOCK);
    flash_attention_mma_swizzled<<<grid, block>>>(Q, K, V, O, N);
}

// ============================================================================
// M10.7 — flash_attention_mma_fa2
// ============================================================================
// Stacks three FA-2-shape perf levers on top of M10.6:
//
//   Lever 1 (Q register-resident across K-tile iters):
//     M10.6 already hoists the Q ldmatrix loads out of the main K-tile loop —
//     q_frag[K_CHUNKS_S][4] is computed once before the loop and reused. So
//     this lever was already harvested at 10.6. No code change at 10.7.
//
//   Lever 2 (STAGES=3 cp.async pipeline):
//     M10.6 double-buffers K/V tiles (STAGES=2). M10.7 goes to STAGES=3, which
//     gives the cp.async pipeline more depth to overlap loads with the
//     compute-heavy softmax + P·V section. Shared-memory budget:
//       Q: WARPS_M10_4 * BR4 * D4 * 2B          = 4*16*64*2  =  8 KB
//       K: STAGES * BC4 * D4 * 2B               = 3*32*64*2  = 12 KB
//       V: STAGES * BC4 * D4 * 2B               = 3*32*64*2  = 12 KB
//       Total                                                = 32 KB
//     Comfortably within sm_89's per-block limit, but exceeds the default 48 KB
//     ceiling on dynamic shared, so the launcher calls cudaFuncSetAttribute
//     with cudaFuncAttributeMaxDynamicSharedMemorySize (also gates static
//     shared on sm_89).
//
//   Lever 3 (ldmatrix for K — manual __half2 → ldmatrix.x4):
//     M10.6 loads K's B-fragment via 2 manual LDS.32 per lane per (kk, n) mma
//     — 16 mmas/tile × 2 loads = 32 LDS.32 per lane per tile.
//
//     The trick: K-as-B is naturally col-major (d-axis fast in shared = K_mma
//     axis fast in mma's B operand). So K does NOT need `.trans` (unlike V,
//     where v_row=K_mma is slow but d=N is fast). Instead, an `ldmatrix.x4`
//     (no .trans) call gives an A-fragment of a 16×16 matrix; the resulting
//     4 b32/lane EXACTLY match the B-fragment layout for TWO consecutive
//     (kk, n) and (kk, n+1) mmas. Specifically, with M = K_tile[n*8..n*8+15,
//     kk*16..kk*16+15]:
//       a0 = M[g,   2t..2t+1]  = K_tile[n*8+g,   kk*16+2t..2t+1] = b0 for (kk, n)
//       a1 = M[g+8, 2t..2t+1]  = K_tile[n*8+g+8, kk*16+2t..2t+1] = b0 for (kk, n+1)
//       a2 = M[g,   2t+8..2t+9]= K_tile[n*8+g,   kk*16+2t+8..2t+9]= b1 for (kk, n)
//       a3 = M[g+8, 2t+8..2t+9]= K_tile[n*8+g+8, kk*16+2t+8..2t+9]= b1 for (kk, n+1)
//     So 4 LDSM-x4 per kk (one per n-pair) covers all 4 n values — net 16
//     LDSM total per K-tile, replacing 32 LDS.32 (and the manual swizzle math
//     becomes a single base-pointer swizzle per call instead of per-half2).
// ============================================================================
constexpr int STAGES_M10_7 = 3;

__global__ void flash_attention_mma_fa2(const __half* __restrict__ Q,
                                        const __half* __restrict__ K,
                                        const __half* __restrict__ V,
                                        float*        __restrict__ O,
                                        int N) {
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int q_lane  = lane >> 2;    // 0..7
    int t_lane  = lane & 3;     // 0..3

    int row_base = blockIdx.x * BR4_BLOCK + warp_id * BR4;

    int row_top = row_base + q_lane;
    int row_bot = row_base + q_lane + 8;

    // ---- Shared layout: union Q's buffer with K's stage 0 + V's stage 0. ----
    // Q is read once at kernel entry, copied to registers via ldmatrix, and
    // never touched again. After the block-wide barrier we issue cp.async
    // into Ks[0]/Vs[0], so reusing the same storage for Q and for stage 0 of
    // K/V costs nothing — and saves 8 KB of static shared. That brings the
    // per-block footprint from 32 KB → 24 KB, which lets sm_89 schedule 4
    // blocks/SM instead of 3 (100 KB / 24 KB ≈ 4.17), regaining the
    // occupancy we'd otherwise lose vs M10.6.
    //
    // Layout invariant: Qs_blk[WARPS_M10_4][BR4 * D4] = 4*16*64 = 4096 halves
    // = exactly 2 * (BC4 * D4) = 2 * 2048 halves = enough room for ONE Ks
    // stage AND ONE Vs stage of width D4. We place Ks[0] at offset 0, Vs[0]
    // at offset BC4*D4 within the Q region.
    static_assert(WARPS_M10_4 * BR4 * D4 == 2 * BC4 * D4,
                  "Q/K/V overlay sizes must match");

    __shared__ __half KsVs_overlay[2 * BC4 * D4];               // Q | Ks[0] | Vs[0]
    __shared__ __half Ks_rest[STAGES_M10_7 - 1][BC4 * D4];      // Ks[1..STAGES-1]
    __shared__ __half Vs_rest[STAGES_M10_7 - 1][BC4 * D4];      // Vs[1..STAGES-1]

    auto Ks = [&](int s) -> __half* {
        return (s == 0) ? &KsVs_overlay[0] : Ks_rest[s - 1];
    };
    auto Vs = [&](int s) -> __half* {
        return (s == 0) ? &KsVs_overlay[BC4 * D4] : Vs_rest[s - 1];
    };

    __half* Qs = &KsVs_overlay[warp_id * (BR4 * D4)];

    // ---- Stage Q for this warp (swizzled write) ----
    {
        constexpr int VEC_HALVES   = 8;
        constexpr int VECS_PER_ROW = D4 / VEC_HALVES;          // 8
        constexpr int N_VECS       = (BR4 * D4) / VEC_HALVES;  // 128
        #pragma unroll
        for (int p = 0; p < N_VECS / 32; ++p) {
            int t = p * 32 + lane;
            int r = t / VECS_PER_ROW;
            int c = (t % VECS_PER_ROW) * VEC_HALVES;
            int row_global = row_base + r;
            int c_sw = m10_6_swizzle(r, c);
            int4 v;
            if (row_global < N) {
                v = *reinterpret_cast<const int4*>(&Q[row_global * D4 + c]);
            } else {
                v = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&Qs[r * D4 + c_sw]) = v;
        }
    }
    __syncwarp();

    // ---- Lever 1: Q register-resident. ldmatrix.x4 once, reused every K-tile. ----
    uint32_t q_frag[K_CHUNKS_S][4];
    {
        int tile      = lane >> 3;
        int row_in_t  = lane & 7;
        int row_block = tile & 1;
        int col_block = tile >> 1;
        int row = row_block * 8 + row_in_t;
        #pragma unroll
        for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
            int col_base   = kk * 16 + col_block * 8;
            int col_sw     = m10_6_swizzle(row, col_base);
            const __half* row_addr = &Qs[row * D4 + col_sw];
            uint32_t smem_int = __cvta_generic_to_shared(row_addr);
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(q_frag[kk][0]), "=r"(q_frag[kk][1]),
                  "=r"(q_frag[kk][2]), "=r"(q_frag[kk][3])
                :  "r"(smem_int));
        }
    }

    // ---- O accumulator ----
    float o_frag[D_CHUNKS_O][4];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        o_frag[dc][0] = 0.0f; o_frag[dc][1] = 0.0f;
        o_frag[dc][2] = 0.0f; o_frag[dc][3] = 0.0f;
    }

    float m_state[2] = {-INFINITY, -INFINITY};
    float l_state[2] = {0.0f, 0.0f};

    const float scale = 1.0f / sqrtf((float)D4);
    int Tc = (N + BC4 - 1) / BC4;

    auto issue_tile = [&](int stage, int j, bool may_be_oob) {
        constexpr int VEC_HALVES   = 8;
        constexpr int VECS_PER_ROW = D4 / VEC_HALVES;
        constexpr int N_TRANSFERS  = (BC4 * D4) / VEC_HALVES;
        constexpr int PASSES       = N_TRANSFERS / M10_4_THREADS;

        #pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            int t = p * M10_4_THREADS + threadIdx.x;
            int row_in_tile = t / VECS_PER_ROW;
            int col_in_row  = (t % VECS_PER_ROW) * VEC_HALVES;
            int kcol = j * BC4 + row_in_tile;
            int src_row = may_be_oob ? ((kcol < N) ? kcol : (N - 1)) : kcol;

            int col_sw = m10_6_swizzle(row_in_tile, col_in_row);

            __half*       k_smem = &Ks(stage)[row_in_tile * D4 + col_sw];
            __half*       v_smem = &Vs(stage)[row_in_tile * D4 + col_sw];
            const __half* k_gmem = &K[src_row * D4 + col_in_row];
            const __half* v_gmem = &V[src_row * D4 + col_in_row];

            m10_6_cp_async_16(k_smem, k_gmem);
            m10_6_cp_async_16(v_smem, v_gmem);
        }
    };

    // ---- Lever 2: STAGES=3 prologue — issue STAGES-1=2 commits ahead. ----
    // CRITICAL: Q's ldmatrix reads above MUST be globally complete before any
    // thread issues cp.async into Ks[0]/Vs[0] (they alias the Q buffer).
    // ldmatrix.sync is a warp-level sync, but we need block-wide sync because
    // a different warp's cp.async writes to stage 0 will clobber Q.
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < STAGES_M10_7 - 1; ++s) {
        if (s < Tc) issue_tile(s, s, /*may_be_oob=*/(s + 1 == Tc));
        __pipeline_commit();
    }

    int compute_stage = 0;
    int load_stage    = (STAGES_M10_7 - 1) % STAGES_M10_7;   // = 2

    for (int j = 0; j < Tc; ++j) {
        int next_j = j + (STAGES_M10_7 - 1);
        if (next_j < Tc) {
            issue_tile(load_stage, next_j, /*may_be_oob=*/(next_j + 1 == Tc));
        }
        __pipeline_commit();

        __pipeline_wait_prior(STAGES_M10_7 - 1);
        __syncthreads();

        __half* Ks_cur = Ks(compute_stage);
        __half* Vs_cur = Vs(compute_stage);

        // ---------------- S = Q · K^T ----------------
        // Lever 3: K's B-fragment via ldmatrix.x4 (no .trans).
        // K-as-B is naturally col-major (d-axis fast in shared = K_mma fast in
        // B's `.col` layout). One ldmatrix.x4 reads a 16×16 K sub-tile starting
        // at (row=n2*16, col=kk*16); the resulting A-frag layout maps directly
        // to the B-fragments of two consecutive mmas: (kk, n=2*n2) and
        // (kk, n=2*n2+1). N_CHUNKS_S=4 → n2 ∈ {0, 1}.
        float s_frag[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            s_frag[n][0] = 0.0f; s_frag[n][1] = 0.0f;
            s_frag[n][2] = 0.0f; s_frag[n][3] = 0.0f;
        }

        // Per-lane sub-tile coordinates for ldmatrix.x4 on K (no .trans).
        // Lane L (0..31), tile = L>>3, row_in_tile = L&7.
        //   row_block = tile & 1   (0 or 1: rows 0-7 vs 8-15 within the 16×16)
        //   col_block = tile >> 1  (0 or 1: cols 0-7 vs 8-15 within the 16×16)
        //   in-tile row = row_block * 8 + row_in_tile  (0..15)
        //   in-tile col_base = col_block * 8           (0 or 8)
        int kld_tile     = lane >> 3;
        int kld_row_in_t = lane & 7;
        int kld_rb       = kld_tile & 1;
        int kld_cb       = kld_tile >> 1;
        int kld_row_off  = kld_rb * 8 + kld_row_in_t;   // 0..15 within the 16×16
        int kld_col_off  = kld_cb * 8;                  // 0 or 8

        constexpr int N2_CHUNKS = N_CHUNKS_S / 2;       // 2 (each ldmatrix.x4 covers 2 n's)

        #pragma unroll
        for (int kk = 0; kk < K_CHUNKS_S; ++kk) {
            #pragma unroll
            for (int n2 = 0; n2 < N2_CHUNKS; ++n2) {
                int k_row    = n2 * 16 + kld_row_off;
                int col_base = kk * 16 + kld_col_off;
                int col_sw   = m10_6_swizzle(k_row, col_base);
                const __half* row_addr = &Ks_cur[k_row * D4 + col_sw];
                uint32_t smem_int = __cvta_generic_to_shared(row_addr);
                uint32_t k0, k1, k2, k3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(k0), "=r"(k1), "=r"(k2), "=r"(k3)
                    :  "r"(smem_int));
                // k0 = K_tile[n2*16 + q_lane,   kk*16 + 2t..2t+1]   = b0 for (kk, 2n2)
                // k1 = K_tile[n2*16 + q_lane+8, kk*16 + 2t..2t+1]   = b0 for (kk, 2n2+1)
                // k2 = K_tile[n2*16 + q_lane,   kk*16 + 2t+8..2t+9] = b1 for (kk, 2n2)
                // k3 = K_tile[n2*16 + q_lane+8, kk*16 + 2t+8..2t+9] = b1 for (kk, 2n2+1)
                int n_a = 2 * n2;
                int n_b = 2 * n2 + 1;
                mma_m16n8k16(s_frag[n_a][0], s_frag[n_a][1], s_frag[n_a][2], s_frag[n_a][3],
                             q_frag[kk][0], q_frag[kk][1], q_frag[kk][2], q_frag[kk][3],
                             k0, k2,
                             s_frag[n_a][0], s_frag[n_a][1], s_frag[n_a][2], s_frag[n_a][3]);
                mma_m16n8k16(s_frag[n_b][0], s_frag[n_b][1], s_frag[n_b][2], s_frag[n_b][3],
                             q_frag[kk][0], q_frag[kk][1], q_frag[kk][2], q_frag[kk][3],
                             k1, k3,
                             s_frag[n_b][0], s_frag[n_b][1], s_frag[n_b][2], s_frag[n_b][3]);
            }
        }

        // ---- Scale + OOB mask ----
        // Hot path: tile is fully in-bounds → no OOB check, just scale.
        // The OOB branch only fires on the final tile when N isn't a BC4
        // multiple (rare in benchmark; possible in production).
        bool last_tile_oob = ((j + 1) * BC4 > N);
        if (!last_tile_oob) {
            #pragma unroll
            for (int n = 0; n < N_CHUNKS_S; ++n) {
                s_frag[n][0] *= scale;
                s_frag[n][1] *= scale;
                s_frag[n][2] *= scale;
                s_frag[n][3] *= scale;
            }
        } else {
            #pragma unroll
            for (int n = 0; n < N_CHUNKS_S; ++n) {
                int col0 = n * 8 + 2 * t_lane;
                int col1 = col0 + 1;
                int kcol0 = j * BC4 + col0;
                int kcol1 = j * BC4 + col1;
                bool m0 = (kcol0 >= N);
                bool m1 = (kcol1 >= N);
                s_frag[n][0] = m0 ? -INFINITY : (s_frag[n][0] * scale);
                s_frag[n][1] = m1 ? -INFINITY : (s_frag[n][1] * scale);
                s_frag[n][2] = m0 ? -INFINITY : (s_frag[n][2] * scale);
                s_frag[n][3] = m1 ? -INFINITY : (s_frag[n][3] * scale);
            }
        }

        // ---- Row max ----
        float row_max_top = -INFINITY;
        float row_max_bot = -INFINITY;
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            row_max_top = fmaxf(row_max_top, fmaxf(s_frag[n][0], s_frag[n][1]));
            row_max_bot = fmaxf(row_max_bot, fmaxf(s_frag[n][2], s_frag[n][3]));
        }
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 1));
        row_max_top = fmaxf(row_max_top, __shfl_xor_sync(0xffffffff, row_max_top, 2));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 1));
        row_max_bot = fmaxf(row_max_bot, __shfl_xor_sync(0xffffffff, row_max_bot, 2));

        float m_new_top = fmaxf(m_state[0], row_max_top);
        float m_new_bot = fmaxf(m_state[1], row_max_bot);
        float alpha_top = (m_state[0] == -INFINITY) ? 0.0f : __expf(m_state[0] - m_new_top);
        float alpha_bot = (m_state[1] == -INFINITY) ? 0.0f : __expf(m_state[1] - m_new_bot);

        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
            o_frag[dc][0] *= alpha_top;
            o_frag[dc][1] *= alpha_top;
            o_frag[dc][2] *= alpha_bot;
            o_frag[dc][3] *= alpha_bot;
        }

        float l_ij_top = 0.0f;
        float l_ij_bot = 0.0f;
        float p_vals[N_CHUNKS_S][4];
        #pragma unroll
        for (int n = 0; n < N_CHUNKS_S; ++n) {
            float p0 = (s_frag[n][0] == -INFINITY) ? 0.0f : __expf(s_frag[n][0] - m_new_top);
            float p1 = (s_frag[n][1] == -INFINITY) ? 0.0f : __expf(s_frag[n][1] - m_new_top);
            float p2 = (s_frag[n][2] == -INFINITY) ? 0.0f : __expf(s_frag[n][2] - m_new_bot);
            float p3 = (s_frag[n][3] == -INFINITY) ? 0.0f : __expf(s_frag[n][3] - m_new_bot);
            p_vals[n][0] = p0; p_vals[n][1] = p1;
            p_vals[n][2] = p2; p_vals[n][3] = p3;
            l_ij_top += p0 + p1;
            l_ij_bot += p2 + p3;
        }
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 1);
        l_ij_top += __shfl_xor_sync(0xffffffff, l_ij_top, 2);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 1);
        l_ij_bot += __shfl_xor_sync(0xffffffff, l_ij_bot, 2);

        l_state[0] = alpha_top * l_state[0] + l_ij_top;
        l_state[1] = alpha_bot * l_state[1] + l_ij_bot;
        m_state[0] = m_new_top;
        m_state[1] = m_new_bot;

        // ---- Repack P into A-fragment layout ----
        uint32_t p_frag[P_CHUNKS][4];
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            int n0 = 2 * pc;
            int n1 = 2 * pc + 1;
            p_frag[pc][0] = pack_halves(__float2half(p_vals[n0][0]), __float2half(p_vals[n0][1]));
            p_frag[pc][1] = pack_halves(__float2half(p_vals[n0][2]), __float2half(p_vals[n0][3]));
            p_frag[pc][2] = pack_halves(__float2half(p_vals[n1][0]), __float2half(p_vals[n1][1]));
            p_frag[pc][3] = pack_halves(__float2half(p_vals[n1][2]), __float2half(p_vals[n1][3]));
        }

        // ---------------- O += P · V via ldmatrix.x2.trans on swizzled V ----
        int v_lane = lane & 15;
        #pragma unroll
        for (int pc = 0; pc < P_CHUNKS; ++pc) {
            int v_row = pc * 16 + v_lane;
            #pragma unroll
            for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
                int col_base = dc * 8;
                int col_sw   = m10_6_swizzle(v_row, col_base);
                const __half* row_addr = &Vs_cur[v_row * D4 + col_sw];
                uint32_t smem_int = __cvta_generic_to_shared(row_addr);
                uint32_t b0, b1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b0), "=r"(b1)
                    :  "r"(smem_int));

                mma_m16n8k16(o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3],
                             p_frag[pc][0], p_frag[pc][1], p_frag[pc][2], p_frag[pc][3],
                             b0, b1,
                             o_frag[dc][0], o_frag[dc][1], o_frag[dc][2], o_frag[dc][3]);
            }
        }

        __syncthreads();

        compute_stage = (compute_stage + 1) % STAGES_M10_7;
        load_stage    = (load_stage    + 1) % STAGES_M10_7;
    }

    // ---- Normalize and write O ----
    float inv_l_top = (l_state[0] > 0.0f) ? (1.0f / l_state[0]) : 0.0f;
    float inv_l_bot = (l_state[1] > 0.0f) ? (1.0f / l_state[1]) : 0.0f;

    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS_O; ++dc) {
        int col0 = dc * 8 + 2 * t_lane;
        int col1 = col0 + 1;
        if (row_top < N) {
            O[row_top * D4 + col0] = o_frag[dc][0] * inv_l_top;
            O[row_top * D4 + col1] = o_frag[dc][1] * inv_l_top;
        }
        if (row_bot < N) {
            O[row_bot * D4 + col0] = o_frag[dc][2] * inv_l_bot;
            O[row_bot * D4 + col1] = o_frag[dc][3] * inv_l_bot;
        }
    }
}

inline void launch_flash_mma_fa2(const __half* Q, const __half* K, const __half* V,
                                 float* O, int N) {
    // sm_89: increasing MaxDynamicSharedMemorySize also gates the static shared
    // budget above the default 48 KB. M10.7 uses ~32 KB of static shared (Q + 3
    // stages of K + 3 stages of V), well under the limit, but we raise the cap
    // defensively in case the compiler chooses to lay things out differently.
    // Raise the per-block max-dynamic-shared cap. sm_89 default is 48 KB; on
    // sm_89 this attribute *also* gates the static-shared cap when static
    // shared exceeds 48 KB. M10.7's static footprint at STAGES=3 is ~32 KB
    // (below 48 KB so technically not strictly needed here), but we set it
    // anyway because (a) it documents the FA-2 pattern (any kernel pushing
    // STAGES higher / BC4 wider will need it), and (b) it's a one-time
    // initialization cost. sharedMemPerBlockOptin on sm_89 is 99 KB.
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(flash_attention_mma_fa2,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             64 * 1024);
        attr_set = true;
    }
    dim3 block(M10_4_THREADS);
    dim3 grid((N + BR4_BLOCK - 1) / BR4_BLOCK);
    flash_attention_mma_fa2<<<grid, block>>>(Q, K, V, O, N);
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
