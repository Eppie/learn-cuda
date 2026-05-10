// Module 12 — verifies the multi-head causal FlashAttention against a host
// reference at small B, H, N (host triple-loop is O(B·H·N²·D)).

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.h"
#include "kernels.cuh"

constexpr int B_VAL = 2;
constexpr int H_VAL = 2;
constexpr int N_VAL = 256;

static void host_attention_mhc(const float* Q, const float* K, const float* V,
                               float* O, int B, int H, int N, bool causal) {
    float scale = 1.0f / std::sqrt((float)D);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const float* q = Q + ((long long)b * H + h) * N * D;
            const float* k = K + ((long long)b * H + h) * N * D;
            const float* v = V + ((long long)b * H + h) * N * D;
            float* o       = O + ((long long)b * H + h) * N * D;
            std::vector<float> S(N * N), P(N * N);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += q[i*D+d] * k[j*D+d];
                    S[i*N + j] = dot * scale;
                    if (causal && j > i) S[i*N + j] = -INFINITY;
                }
            }
            for (int i = 0; i < N; ++i) {
                float m = -INFINITY;
                for (int j = 0; j < N; ++j) m = std::fmax(m, S[i*N + j]);
                float s = 0.0f;
                for (int j = 0; j < N; ++j) {
                    P[i*N + j] = std::exp(S[i*N + j] - m);
                    s += P[i*N + j];
                }
                float inv = 1.0f / s;
                for (int j = 0; j < N; ++j) P[i*N + j] *= inv;
            }
            for (int i = 0; i < N; ++i) {
                for (int d = 0; d < D; ++d) {
                    float dot = 0.0f;
                    for (int j = 0; j < N; ++j) dot += P[i*N + j] * v[j*D + d];
                    o[i*D + d] = dot;
                }
            }
        }
    }
}

static bool check(const std::vector<float>& got, const std::vector<float>& expected,
                  const char* name, float rel = 1e-3f, float abs_t = 1e-4f) {
    float max_abs = 0.0f;
    int   bad = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        float d = std::fabs(got[i] - expected[i]);
        float allowed = std::fmax(rel * std::fabs(expected[i]), abs_t);
        if (d > allowed && bad < 0) bad = (int)i;
        if (d > max_abs) max_abs = d;
    }
    bool ok = (bad < 0);
    std::printf("%-30s max_abs=%.3e %s\n", name, max_abs, ok ? "(PASS)" : "(FAIL)");
    return ok;
}

int main() {
    int B = B_VAL, H = H_VAL, N = N_VAL;
    long long total = (long long)B * H * N * D;
    std::printf("Multi-head FA verification: B=%d, H=%d, N=%d, D=%d\n", B, H, N, D);

    std::srand(123);
    std::vector<float> h_Q(total), h_K(total), h_V(total);
    for (auto& v : h_Q) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_K) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_V) v = (std::rand() % 2001 - 1000) * 1e-3f;

    std::vector<float> h_ref_nc(total), h_ref_c(total);
    host_attention_mhc(h_Q.data(), h_K.data(), h_V.data(), h_ref_nc.data(), B, H, N, false);
    host_attention_mhc(h_Q.data(), h_K.data(), h_V.data(), h_ref_c.data(),  B, H, N, true);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_got(total);

    launch_flash_mhc(d_Q, d_K, d_V, d_O, B, H, N, false);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref_nc, "flash_mhc (causal=false)");

    launch_flash_mhc(d_Q, d_K, d_V, d_O, B, H, N, true);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, total * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref_c, "flash_mhc (causal=true)");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
