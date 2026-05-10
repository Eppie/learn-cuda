// Module 10 — starter scaffold. Solve the TODOs.
//
// Block sizes BR=32, BC=32, D=64 are fixed. One block per Q row tile, one thread
// per Q row.

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.h"

constexpr int BR = 32;
constexpr int BC = 32;
constexpr int D  = 64;

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
        // Cooperative tile load.
        for (int idx = threadIdx.x; idx < BC * D; idx += BR) {
            int c = idx / D, d = idx % D;
            int kcol = j * BC + c;
            Ks[c][d] = (kcol < N) ? K[kcol * D + d] : 0.0f;
            Vs[c][d] = (kcol < N) ? V[kcol * D + d] : 0.0f;
        }
        __syncthreads();

        if (row < N) {
            // ================================================================
            // TODO 1: compute s[c] = scale * Σ_d q[d] * Ks[c][d] for c ∈ [0, BC).
            //         Mask kcol >= N with -INFINITY. Track m_ij = max(s).
            // ================================================================
            float s[BC];
            float m_ij = -INFINITY;
            // your code here

            // ================================================================
            // TODO 2: online softmax update.
            //   m_new = max(m_state, m_ij)
            //   alpha = exp(m_state - m_new)
            //   p[c]  = exp(s[c] - m_new)
            //   l_ij  = Σ p[c]
            // ================================================================
            float m_new = m_ij; (void)m_new;
            float alpha = 1.0f; (void)alpha;
            float p[BC]; (void)p;
            float l_ij = 0.0f; (void)l_ij;
            // your code here

            // ================================================================
            // TODO 3: rescale and accumulate.
            //   o[d] = alpha * o[d] + Σ_c p[c] * Vs[c][d]
            //   l_state = alpha * l_state + l_ij
            //   m_state = m_new
            // ================================================================
            // your code here
        }

        __syncthreads();
    }

    if (row < N) {
        float inv = 1.0f / l_state;
        for (int d = 0; d < D; ++d) O[row * D + d] = o[d] * inv;
    }
}

inline void launch_flash(const float* Q, const float* K, const float* V, float* O,
                         int N) {
    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR);
    flash_attention<<<grid, block>>>(Q, K, V, O, N);
}

// Verification harness (uses a host reference at moderate N).
constexpr int N_VAL = 1024;

static void host_attention(const float* Q, const float* K, const float* V,
                           float* O, int N) {
    std::vector<float> S(N * N);
    float scale = 1.0f / std::sqrt((float)D);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i*D+d] * K[j*D+d];
            S[i*N+j] = dot * scale;
        }
    for (int i = 0; i < N; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::fmax(m, S[i*N+j]);
        float s = 0;
        for (int j = 0; j < N; ++j) { S[i*N+j] = std::exp(S[i*N+j] - m); s += S[i*N+j]; }
        float inv = 1.0f / s;
        for (int j = 0; j < N; ++j) S[i*N+j] *= inv;
    }
    for (int i = 0; i < N; ++i)
        for (int d = 0; d < D; ++d) {
            float dot = 0.0f;
            for (int j = 0; j < N; ++j) dot += S[i*N+j] * V[j*D+d];
            O[i*D+d] = dot;
        }
}

int main() {
    int N = N_VAL;
    std::printf("Attention: N=%d, D=%d (FP32, single head)\n", N, D);
    std::srand(123);
    std::vector<float> h_Q(N * D), h_K(N * D), h_V(N * D);
    for (auto& v : h_Q) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_K) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_V) v = (std::rand() % 2001 - 1000) * 1e-3f;

    std::vector<float> h_ref(N * D);
    host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N*D*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N*D*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N*D*sizeof(float), cudaMemcpyHostToDevice));

    launch_flash(d_Q, d_K, d_V, d_O, N);
    CUDA_CHECK_LAST();

    std::vector<float> h_got(N * D);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N*D*sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    int bad = -1;
    for (int i = 0; i < N * D; ++i) {
        float d = std::fabs(h_got[i] - h_ref[i]);
        if (d > 1e-3f && bad < 0) bad = i;
        if (d > max_abs) max_abs = d;
    }
    std::printf("flash attention   max_abs=%.3e %s\n", max_abs, bad < 0 ? "(PASS)" : "(FAIL)");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
