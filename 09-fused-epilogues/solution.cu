// Module 9 — verifies fused softmax / layernorm / residual+LN / Welford LN /
// FP16 LN / GEMM+bias+GELU against host references.

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>

#include "cuda_utils.h"
#include "kernels.cuh"

constexpr int ROWS = 4096;
constexpr int COLS = 4096;

static void host_softmax(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float m = -INFINITY;
        for (int i = 0; i < cols; ++i) m = std::fmax(m, xr[i]);
        float s = 0.0f;
        for (int i = 0; i < cols; ++i) s += std::exp(xr[i] - m);
        float inv = 1.0f / s;
        for (int i = 0; i < cols; ++i) yr[i] = std::exp(xr[i] - m) * inv;
    }
}

static void host_layernorm(const float* x, float* y, const float* g, const float* b,
                           int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        double s = 0.0, sq = 0.0;
        for (int i = 0; i < cols; ++i) { s += xr[i]; sq += xr[i] * xr[i]; }
        double m = s / cols;
        double v = sq / cols - m * m;
        double rstd = 1.0 / std::sqrt(v + eps);
        for (int i = 0; i < cols; ++i) yr[i] = g[i] * (xr[i] - m) * rstd + b[i];
    }
}

static void host_layernorm_residual_add(const float* x, const float* res,
                                        float* sum_out, float* y_out,
                                        const float* g, const float* b,
                                        int rows, int cols, float eps) {
    std::vector<float> tmp(cols);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x   + r * cols;
        const float* rr = res + r * cols;
        float* sr = sum_out   + r * cols;
        float* yr = y_out     + r * cols;
        double s = 0.0, sq = 0.0;
        for (int i = 0; i < cols; ++i) {
            float v = xr[i] + rr[i];
            sr[i] = v;
            s += v; sq += v * v;
        }
        double m = s / cols;
        double v = sq / cols - m * m;
        double rstd = 1.0 / std::sqrt(v + eps);
        for (int i = 0; i < cols; ++i) yr[i] = g[i] * (sr[i] - m) * rstd + b[i];
    }
}

static bool check(const std::vector<float>& got, const std::vector<float>& expected,
                  const char* name, float tol = 1e-4f) {
    float max_abs = 0.0f;
    int   bad = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        float d = std::fabs(got[i] - expected[i]);
        float allowed = std::fmax(tol * std::fabs(expected[i]), tol);
        if (d > allowed && bad < 0) bad = (int)i;
        if (d > max_abs) max_abs = d;
    }
    bool ok = bad < 0;
    std::printf("%-30s max_abs=%.3e %s\n", name, max_abs, ok ? "(PASS)" : "(FAIL)");
    return ok;
}

int main() {
    std::srand(123);
    std::vector<float> h_x(ROWS * COLS), h_r(ROWS * COLS), h_g(COLS), h_b(COLS);
    for (auto& v : h_x) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_r) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_g) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_b) v = (std::rand() % 2001 - 1000) * 1e-3f;

    std::vector<float> h_y_softmax(ROWS * COLS), h_y_layernorm(ROWS * COLS);
    std::vector<float> h_sum_resln(ROWS * COLS), h_y_resln(ROWS * COLS);
    host_softmax  (h_x.data(), h_y_softmax.data(),   ROWS, COLS);
    host_layernorm(h_x.data(), h_y_layernorm.data(), h_g.data(), h_b.data(),
                   ROWS, COLS, 1e-5f);
    host_layernorm_residual_add(h_x.data(), h_r.data(),
                                h_sum_resln.data(), h_y_resln.data(),
                                h_g.data(), h_b.data(), ROWS, COLS, 1e-5f);

    float *d_x, *d_y, *d_g, *d_b, *d_r, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_x,   ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,   ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r,   ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, COLS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, h_r.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_got(ROWS * COLS);

    softmax_fused<<<ROWS, BLK>>>(d_x, d_y, ROWS, COLS);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_y_softmax, "softmax_fused", 1e-4f);

    softmax_online<<<ROWS, BLK>>>(d_x, d_y, ROWS, COLS);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_y_softmax, "softmax_online", 1e-4f);

    layernorm_fused<<<ROWS, BLK>>>(d_x, d_y, d_g, d_b, ROWS, COLS, 1e-5f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_y_layernorm, "layernorm_fused", 1e-3f);

    layernorm_welford<<<ROWS, BLK>>>(d_x, d_y, d_g, d_b, ROWS, COLS, 1e-5f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_y_layernorm, "layernorm_welford", 1e-3f);

    layernorm_residual_add_v0<<<ROWS, BLK>>>(d_x, d_r, d_sum, d_y, d_g, d_b,
                                             ROWS, COLS, 1e-5f);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_sum, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_sum_resln, "ln_resadd sum stream", 1e-4f);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_y, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_y_resln, "ln_resadd norm output", 1e-3f);

    // FP16 LayerNorm.
    {
        std::vector<__half> h_x16(ROWS * COLS), h_g16(COLS), h_b16(COLS);
        for (size_t i = 0; i < h_x.size(); ++i) h_x16[i] = __float2half(h_x[i]);
        for (int i = 0; i < COLS;            ++i) h_g16[i] = __float2half(h_g[i]);
        for (int i = 0; i < COLS;            ++i) h_b16[i] = __float2half(h_b[i]);

        __half *d_x16, *d_y16, *d_g16, *d_b16;
        CUDA_CHECK(cudaMalloc(&d_x16, ROWS * COLS * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y16, ROWS * COLS * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_g16, COLS        * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_b16, COLS        * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_x16, h_x16.data(), ROWS * COLS * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g16, h_g16.data(), COLS        * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b16, h_b16.data(), COLS        * sizeof(__half), cudaMemcpyHostToDevice));

        layernorm_fused_fp16<<<ROWS, BLK>>>(d_x16, d_y16, d_g16, d_b16, ROWS, COLS, 1e-5f);
        CUDA_CHECK_LAST();

        std::vector<__half> h_y16(ROWS * COLS);
        CUDA_CHECK(cudaMemcpy(h_y16.data(), d_y16, ROWS * COLS * sizeof(__half), cudaMemcpyDeviceToHost));
        std::vector<float> h_y16f(ROWS * COLS);
        for (size_t i = 0; i < h_y16.size(); ++i) h_y16f[i] = __half2float(h_y16[i]);
        // FP16 has ~3 decimal digits; widen tolerance accordingly.
        check(h_y16f, h_y_layernorm, "layernorm_fused_fp16", 5e-2f);

        CUDA_CHECK(cudaFree(d_x16));
        CUDA_CHECK(cudaFree(d_y16));
        CUDA_CHECK(cudaFree(d_g16));
        CUDA_CHECK(cudaFree(d_b16));
    }

    // GEMM + bias + GELU.
    {
        constexpr int M = 256, N = 256, K = 256;
        std::vector<float> hA(M * K), hB(K * N), hBias(N);
        for (auto& v : hA)    v = (std::rand() % 2001 - 1000) * 1e-3f;
        for (auto& v : hB)    v = (std::rand() % 2001 - 1000) * 1e-3f;
        for (auto& v : hBias) v = (std::rand() % 2001 - 1000) * 1e-3f;

        std::vector<float> hCref(M * N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int kk = 0; kk < K; ++kk) acc += hA[i * K + kk] * hB[kk * N + j];
                hCref[i * N + j] = acc;
            }
        std::vector<float> hRef_v0_pre(hCref);
        // post-pass bias+gelu
        constexpr float kk0 = 0.7978845608028654f, kk1 = 0.044715f;
        std::vector<float> hRef(M * N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float v = hCref[i * N + j] + hBias[j];
                hRef[i * N + j] = 0.5f * v * (1.0f + std::tanh(kk0 * (v + kk1 * v * v * v)));
            }

        float *dA, *dB, *dBias, *dC;
        CUDA_CHECK(cudaMalloc(&dA,    M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,    K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dBias, N     * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC,    M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(),       M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(),       K * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), N     * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<float> hOut(M * N);

        launch_gemm_bias_gelu_v0(dA, dB, dBias, dC, M, N, K);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(hOut.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        check(hOut, hRef, "gemm_bias_gelu_v0", 1e-3f);

        CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));
        launch_gemm_bias_gelu_v1(dA, dB, dBias, dC, M, N, K);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(hOut.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        check(hOut, hRef, "gemm_bias_gelu_v1 (fused)", 1e-3f);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dBias));
        CUDA_CHECK(cudaFree(dC));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
