#pragma once

#include <cuda_runtime.h>
#include <cmath>

constexpr int BR = 32;
constexpr int BC = 32;
constexpr int D  = 64;

// ============================================================================
// Multi-head causal FlashAttention forward, FP32.
//   Q, K, V, O have shape [B, H, N, D] in row-major layout.
//   When `causal` is true, position i can only attend to positions ≤ i.
//
// Grid: (N/BR, H, B). Block: BR threads. Each block computes BR rows of O for
// one (batch, head) pair.
// ============================================================================
__global__ void flash_attention_mhc(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    int N, int H, int B, bool causal) {
    int row = blockIdx.x * BR + threadIdx.x;
    int h   = blockIdx.y;
    int b   = blockIdx.z;

    long long head_offset = ((long long)b * H + h) * N * D;
    Q += head_offset;
    K += head_offset;
    V += head_offset;
    O += head_offset;

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
        // Causal pruning: if the entire BR×BC tile is above the diagonal
        // (every k_row > every q_row), skip the whole iteration.
        if (causal) {
            int tile_min_kcol = j * BC;
            int tile_min_qrow = blockIdx.x * BR;
            if (tile_min_kcol > tile_min_qrow + BR - 1) {
                // Not strictly necessary to break — the inner loop would just
                // mask everything to -INF — but skipping the load is a big win.
                __syncthreads();
                continue;
            }
        }

        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int c = idx / D, d = idx % D;
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
                bool oob   = (kcol >= N);
                bool masked = causal && (kcol > row);
                if (oob || masked) {
                    s[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[d] * Ks[c][d];
                    s[c] = dot * scale;
                }
                if (s[c] > m_ij) m_ij = s[c];
            }

            // If every column was masked, m_ij stays -INF and the iteration
            // contributes nothing. Skip the rescaling work.
            if (m_ij == -INFINITY) {
                __syncthreads();
                continue;
            }

            float m_new = fmaxf(m_state, m_ij);
            float alpha = expf(m_state - m_new);

            float l_ij = 0.0f;
            float p[BC];
            for (int c = 0; c < BC; ++c) {
                p[c] = expf(s[c] - m_new);
                l_ij += p[c];
            }

            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                for (int c = 0; c < BC; ++c) pv += p[c] * Vs[c][d];
                o[d] = alpha * o[d] + pv;
            }

            l_state = alpha * l_state + l_ij;
            m_state = m_new;
        }

        __syncthreads();
    }

    if (row < N) {
        if (l_state > 0.0f) {
            float inv = 1.0f / l_state;
            for (int d = 0; d < D; ++d) O[row * D + d] = o[d] * inv;
        } else {
            // Fully masked row (only happens for q_row = 0 with causal=true if
            // we somehow skipped everything — guard against NaN regardless).
            for (int d = 0; d < D; ++d) O[row * D + d] = 0.0f;
        }
    }
}

inline void launch_flash_mhc(const float* Q, const float* K, const float* V, float* O,
                             int B, int H, int N, bool causal) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR, H, B);
    flash_attention_mhc<<<grid, block>>>(Q, K, V, O, N, H, B, causal);
}
