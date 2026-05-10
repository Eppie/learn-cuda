// Module 12 — starter scaffold.
//
// Take Module 10's single-head FlashAttention and extend it with:
//   1. Batch + head dimensions (offset all pointers by (b*H + h)*N*D)
//   2. Causal masking (set s[c] = -INFINITY whenever k_col > q_row)
//   3. (Stretch) tile-skip when the whole BC×BR tile is above the diagonal

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.h"

constexpr int BR = 32;
constexpr int BC = 32;
constexpr int D  = 64;

__global__ void flash_attention_mhc(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ O,
                                    int N, int H, int B, bool causal) {
    // ========================================================================
    // TODO 1: derive (row, h, b) from blockIdx and threadIdx, and offset Q, K,
    // V, O by (b*H + h)*N*D so the rest of the kernel sees only this head's
    // data.
    // ========================================================================
    int row = 0;        // blockIdx.x * BR + threadIdx.x
    // your code here
    (void)H; (void)B;

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
                // ============================================================
                // TODO 2: in addition to the existing OOB check, set
                // s[c] = -INFINITY when (causal && kcol > row).
                // ============================================================
                if (kcol >= N) {
                    s[c] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[d] * Ks[c][d];
                    s[c] = dot * scale;
                }
                if (s[c] > m_ij) m_ij = s[c];
            }

            // If everything in this iteration is masked (m_ij stayed -INF),
            // skip the rescaling work — it would all multiply by 0.
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

// Verification on a small problem.
constexpr int B_VAL = 2, H_VAL = 2, N_VAL = 256;

static void host_attention_mhc(const float* Q, const float* K, const float* V,
                               float* O, int B, int H, int N, bool causal) {
    float scale = 1.0f / std::sqrt((float)D);
    for (int b = 0; b < B; ++b) for (int h = 0; h < H; ++h) {
        const float* q = Q + ((long long)b * H + h) * N * D;
        const float* k = K + ((long long)b * H + h) * N * D;
        const float* v = V + ((long long)b * H + h) * N * D;
        float* o       = O + ((long long)b * H + h) * N * D;
        std::vector<float> S(N * N);
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += q[i*D+d] * k[j*D+d];
            S[i*N + j] = (causal && j > i) ? -INFINITY : dot * scale;
        }
        for (int i = 0; i < N; ++i) {
            float m = -INFINITY;
            for (int j = 0; j < N; ++j) m = std::fmax(m, S[i*N+j]);
            float s = 0;
            for (int j = 0; j < N; ++j) { S[i*N+j] = std::exp(S[i*N+j]-m); s += S[i*N+j]; }
            float inv = 1.0f / s;
            for (int j = 0; j < N; ++j) S[i*N+j] *= inv;
        }
        for (int i = 0; i < N; ++i) for (int d = 0; d < D; ++d) {
            float dot = 0.0f;
            for (int j = 0; j < N; ++j) dot += S[i*N+j] * v[j*D + d];
            o[i*D + d] = dot;
        }
    }
}

int main() {
    int B = B_VAL, H = H_VAL, N = N_VAL;
    long long total = (long long)B * H * N * D;
    std::printf("FA verification: B=%d, H=%d, N=%d, D=%d\n", B, H, N, D);

    std::srand(123);
    std::vector<float> h_Q(total), h_K(total), h_V(total);
    for (auto& v : h_Q) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_K) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_V) v = (std::rand() % 2001 - 1000) * 1e-3f;

    std::vector<float> h_ref(total);
    host_attention_mhc(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), B, H, N, true);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    launch_flash_mhc(d_Q, d_K, d_V, d_O, B, H, N, true);
    CUDA_CHECK_LAST();
    std::vector<float> h_got(total);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    int bad = -1;
    for (long long i = 0; i < total; ++i) {
        float d = std::fabs(h_got[i] - h_ref[i]);
        if (d > 1e-3f && bad < 0) bad = (int)i;
        if (d > max_abs) max_abs = d;
    }
    std::printf("flash_mhc (causal=true)  max_abs=%.3e %s\n",
                max_abs, bad < 0 ? "(PASS)" : "(FAIL)");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
