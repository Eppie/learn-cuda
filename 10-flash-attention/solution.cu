// Module 10 — verifies single-head + multi-head + causal + KV-cache + GQA
// flash attention against host references.
//
// Verification sizes are deliberately small (B=2, H=4, N=128, D=64) so the
// host references run in a few seconds. The kernels themselves are correct
// at any (multiple-of-tile) size.

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_fp16.h>

#include "cuda_utils.h"
#include "kernels.cuh"

constexpr int N_VAL = 4096;        // headline single-head verify size

// Single-head host reference (used for the headline test).
static void host_attention(const float* Q, const float* K, const float* V,
                           float* O, int N) {
    std::vector<float> S(N * N), P(N * N);
    float scale = 1.0f / std::sqrt((float)D);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i * D + d] * K[j * D + d];
            S[i * N + j] = dot * scale;
        }
    }
    for (int i = 0; i < N; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::fmax(m, S[i * N + j]);
        float s = 0.0f;
        for (int j = 0; j < N; ++j) { P[i * N + j] = std::exp(S[i * N + j] - m); s += P[i * N + j]; }
        float inv = 1.0f / s;
        for (int j = 0; j < N; ++j) P[i * N + j] *= inv;
    }
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            float dot = 0.0f;
            for (int j = 0; j < N; ++j) dot += P[i * N + j] * V[j * D + d];
            O[i * D + d] = dot;
        }
    }
}

// Generic single-head reference with optional causal mask + variable T (for
// KV-cache). N_q rows of Q attend to T columns of K/V.
static void host_attention_generic(const float* Q, const float* K, const float* V,
                                   float* O, int N_q, int T, int D_, bool causal,
                                   int q_row_offset = 0) {
    std::vector<float> S(N_q * T), P(N_q * T);
    float scale = 1.0f / std::sqrt((float)D_);
    for (int i = 0; i < N_q; ++i) {
        for (int j = 0; j < T; ++j) {
            if (causal && j > (q_row_offset + i)) {
                S[i * T + j] = -INFINITY;
            } else {
                float dot = 0.0f;
                for (int d = 0; d < D_; ++d) dot += Q[i * D_ + d] * K[j * D_ + d];
                S[i * T + j] = dot * scale;
            }
        }
    }
    for (int i = 0; i < N_q; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < T; ++j) m = std::fmax(m, S[i * T + j]);
        float s = 0.0f;
        for (int j = 0; j < T; ++j) {
            float v = std::exp(S[i * T + j] - m);
            P[i * T + j] = v;
            s += v;
        }
        float inv = (s > 0) ? 1.0f / s : 0.0f;
        for (int j = 0; j < T; ++j) P[i * T + j] *= inv;
    }
    for (int i = 0; i < N_q; ++i) {
        for (int d = 0; d < D_; ++d) {
            float dot = 0.0f;
            for (int j = 0; j < T; ++j) dot += P[i * T + j] * V[j * D_ + d];
            O[i * D_ + d] = dot;
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
    std::printf("%-32s max_abs=%.3e %s\n", name, max_abs, ok ? "(PASS)" : "(FAIL)");
    return ok;
}

static void fill_random(std::vector<float>& v) {
    for (auto& x : v) x = (std::rand() % 2001 - 1000) * 1e-3f;
}

// ----------------------------------------------------------------------------
// Test 1: single-head, naive vs flash, large N.
// ----------------------------------------------------------------------------
static void test_single_head() {
    int N = N_VAL;
    std::printf("[1] Single-head: N=%d, D=%d (FP32)\n", N, D);

    std::srand(123);
    std::vector<float> h_Q(N * D), h_K(N * D), h_V(N * D);
    fill_random(h_Q); fill_random(h_K); fill_random(h_V);

    std::printf("    computing host reference...\n");
    std::vector<float> h_ref(N * D);
    host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

    float *d_Q, *d_K, *d_V, *d_O, *d_S;
    CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_got(N * D);

    CUDA_CHECK(cudaMemset(d_O, 0, N * D * sizeof(float)));
    launch_naive(d_Q, d_K, d_V, d_O, d_S, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "naive attention");

    CUDA_CHECK(cudaMemset(d_O, 0, N * D * sizeof(float)));
    launch_flash(d_Q, d_K, d_V, d_O, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "flash attention (10.0)");

    // M10.1: warp-cooperative FP32 flash. Same math, much better SM use.
    CUDA_CHECK(cudaMemset(d_O, 0, N * D * sizeof(float)));
    launch_flash_warp(d_Q, d_K, d_V, d_O, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "flash warp-coop (10.1)");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_S));
}

// ----------------------------------------------------------------------------
// Test 1b: M10.1 warp-cooperative kernel — verify at multiple sizes against
// host reference. We pick small sizes here so the host reference is fast.
// ----------------------------------------------------------------------------
static void test_warp_coop() {
    std::printf("\n[1b] M10.1 warp-coop, multi-N verify (FP32)\n");
    for (int N : {256, 1024, 2048}) {
        std::srand(42 + N);
        std::vector<float> h_Q(N * D), h_K(N * D), h_V(N * D);
        fill_random(h_Q); fill_random(h_K); fill_random(h_V);

        std::vector<float> h_ref(N * D);
        host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

        float *d_Q, *d_K, *d_V, *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

        launch_flash_warp(d_Q, d_K, d_V, d_O, N);
        CUDA_CHECK_LAST();

        std::vector<float> h_got(N * D);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        char name[64];
        std::snprintf(name, sizeof(name), "    N=%-5d 10.1 warp-coop", N);
        check(h_got, h_ref, name, /*rel*/1e-3f, /*abs*/1e-3f);

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

// ----------------------------------------------------------------------------
// Test 1c: M10.2 WMMA kernel — verify at multiple sizes. FP16 inputs.
// Tolerance is looser than FP32 because of the FP16 round-off in inputs.
// ----------------------------------------------------------------------------
static void test_wmma_flash() {
    std::printf("\n[1c] M10.2 WMMA flash, multi-N verify (FP16 in / FP32 out)\n");
    for (int N : {128, 1024, 2048}) {
        std::srand(7 + N);
        std::vector<float>  h_Q(N * D), h_K(N * D), h_V(N * D);
        std::vector<__half> h_Qh(N * D), h_Kh(N * D), h_Vh(N * D);
        fill_random(h_Q); fill_random(h_K); fill_random(h_V);
        // Round to FP16 for the WMMA inputs *and* for the host reference, so
        // we measure kernel error, not FP16 rounding error.
        for (int i = 0; i < N * D; ++i) {
            h_Qh[i] = __float2half(h_Q[i]); h_Q[i] = __half2float(h_Qh[i]);
            h_Kh[i] = __float2half(h_K[i]); h_K[i] = __half2float(h_Kh[i]);
            h_Vh[i] = __float2half(h_V[i]); h_V[i] = __half2float(h_Vh[i]);
        }

        std::vector<float> h_ref(N * D);
        host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

        __half *d_Q, *d_K, *d_V;
        float  *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Qh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_Kh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_Vh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

        launch_flash_wmma(d_Q, d_K, d_V, d_O, N);
        CUDA_CHECK_LAST();

        std::vector<float> h_got(N * D);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        char name[64];
        std::snprintf(name, sizeof(name), "    N=%-5d 10.2 WMMA flash", N);
        // FP16 inputs: rel=2e-2 (matches gemm_tc.h::verify), abs=5e-3 to swallow
        // small-magnitude noise from the softmax → V matmul.
        check(h_got, h_ref, name, /*rel*/2e-2f, /*abs*/5e-3f);

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

// ----------------------------------------------------------------------------
// Test 1d: M10.3 cp.async + WMMA kernel — verify at multiple sizes.
// Same setup as M10.2; same tolerance (the only change is HOW K/V are loaded
// into shared memory, not WHAT lands in shared memory).
// ----------------------------------------------------------------------------
static void test_async_wmma_flash() {
    std::printf("\n[1d] M10.3 cp.async + WMMA flash, multi-N verify (FP16 in / FP32 out)\n");
    for (int N : {128, 1024, 2048}) {
        std::srand(7 + N);
        std::vector<float>  h_Q(N * D), h_K(N * D), h_V(N * D);
        std::vector<__half> h_Qh(N * D), h_Kh(N * D), h_Vh(N * D);
        fill_random(h_Q); fill_random(h_K); fill_random(h_V);
        for (int i = 0; i < N * D; ++i) {
            h_Qh[i] = __float2half(h_Q[i]); h_Q[i] = __half2float(h_Qh[i]);
            h_Kh[i] = __float2half(h_K[i]); h_K[i] = __half2float(h_Kh[i]);
            h_Vh[i] = __float2half(h_V[i]); h_V[i] = __half2float(h_Vh[i]);
        }

        std::vector<float> h_ref(N * D);
        host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

        __half *d_Q, *d_K, *d_V;
        float  *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Qh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_Kh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_Vh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

        launch_flash_async_wmma(d_Q, d_K, d_V, d_O, N);
        CUDA_CHECK_LAST();

        std::vector<float> h_got(N * D);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        char name[64];
        std::snprintf(name, sizeof(name), "    N=%-5d 10.3 cp.async+WMMA", N);
        check(h_got, h_ref, name, /*rel*/2e-2f, /*abs*/5e-3f);

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

// ----------------------------------------------------------------------------
// Test 1e: M10.4 raw mma.sync flash kernel — verify at multiple sizes.
// Same setup as M10.2 / M10.3; same FP16 tolerance.
// ----------------------------------------------------------------------------
static void test_mma_flash() {
    std::printf("\n[1e] M10.4 raw mma.sync flash, multi-N verify (FP16 in / FP32 out)\n");
    for (int N : {128, 1024, 2048}) {
        std::srand(7 + N);
        std::vector<float>  h_Q(N * D), h_K(N * D), h_V(N * D);
        std::vector<__half> h_Qh(N * D), h_Kh(N * D), h_Vh(N * D);
        fill_random(h_Q); fill_random(h_K); fill_random(h_V);
        for (int i = 0; i < N * D; ++i) {
            h_Qh[i] = __float2half(h_Q[i]); h_Q[i] = __half2float(h_Qh[i]);
            h_Kh[i] = __float2half(h_K[i]); h_K[i] = __half2float(h_Kh[i]);
            h_Vh[i] = __float2half(h_V[i]); h_V[i] = __half2float(h_Vh[i]);
        }

        std::vector<float> h_ref(N * D);
        host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

        __half *d_Q, *d_K, *d_V;
        float  *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Qh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_Kh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_Vh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

        launch_flash_mma(d_Q, d_K, d_V, d_O, N);
        CUDA_CHECK_LAST();

        std::vector<float> h_got(N * D);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        char name[64];
        std::snprintf(name, sizeof(name), "    N=%-5d 10.4 mma.sync flash", N);
        check(h_got, h_ref, name, /*rel*/2e-2f, /*abs*/5e-3f);

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

// ----------------------------------------------------------------------------
// Test 1f: M10.5 mma.sync + ldmatrix flash kernel — verify at multiple sizes.
// Same setup as M10.4; same FP16 tolerance. Only the shared→register fragment
// loads change (Q via ldmatrix.x4, V via ldmatrix.x2.trans).
// ----------------------------------------------------------------------------
static void test_mma_ldmatrix_flash() {
    std::printf("\n[1f] M10.5 mma.sync + ldmatrix flash, multi-N verify (FP16 in / FP32 out)\n");
    for (int N : {128, 1024, 2048}) {
        std::srand(7 + N);
        std::vector<float>  h_Q(N * D), h_K(N * D), h_V(N * D);
        std::vector<__half> h_Qh(N * D), h_Kh(N * D), h_Vh(N * D);
        fill_random(h_Q); fill_random(h_K); fill_random(h_V);
        for (int i = 0; i < N * D; ++i) {
            h_Qh[i] = __float2half(h_Q[i]); h_Q[i] = __half2float(h_Qh[i]);
            h_Kh[i] = __float2half(h_K[i]); h_K[i] = __half2float(h_Kh[i]);
            h_Vh[i] = __float2half(h_V[i]); h_V[i] = __half2float(h_Vh[i]);
        }

        std::vector<float> h_ref(N * D);
        host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

        __half *d_Q, *d_K, *d_V;
        float  *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Qh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_Kh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_Vh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

        launch_flash_mma_ldmatrix(d_Q, d_K, d_V, d_O, N);
        CUDA_CHECK_LAST();

        std::vector<float> h_got(N * D);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        char name[64];
        std::snprintf(name, sizeof(name), "    N=%-5d 10.5 mma+ldmatrix flash", N);
        check(h_got, h_ref, name, /*rel*/2e-2f, /*abs*/5e-3f);

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

static void test_mma_swizzled_flash() {
    std::printf("\n[1g] M10.6 mma.sync + ldmatrix + swizzled smem flash, multi-N verify (FP16 in / FP32 out)\n");
    for (int N : {128, 1024, 2048}) {
        std::srand(7 + N);
        std::vector<float>  h_Q(N * D), h_K(N * D), h_V(N * D);
        std::vector<__half> h_Qh(N * D), h_Kh(N * D), h_Vh(N * D);
        fill_random(h_Q); fill_random(h_K); fill_random(h_V);
        for (int i = 0; i < N * D; ++i) {
            h_Qh[i] = __float2half(h_Q[i]); h_Q[i] = __half2float(h_Qh[i]);
            h_Kh[i] = __float2half(h_K[i]); h_K[i] = __half2float(h_Kh[i]);
            h_Vh[i] = __float2half(h_V[i]); h_V[i] = __half2float(h_Vh[i]);
        }

        std::vector<float> h_ref(N * D);
        host_attention(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(), N);

        __half *d_Q, *d_K, *d_V;
        float  *d_O;
        CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Q, h_Qh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_Kh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_Vh.data(), N * D * sizeof(__half), cudaMemcpyHostToDevice));

        launch_flash_mma_swizzled(d_Q, d_K, d_V, d_O, N);
        CUDA_CHECK_LAST();

        std::vector<float> h_got(N * D);
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        char name[64];
        std::snprintf(name, sizeof(name), "    N=%-5d 10.6 mma+ldmatrix+swizzle flash", N);
        check(h_got, h_ref, name, /*rel*/2e-2f, /*abs*/5e-3f);

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_O));
    }
}

// ----------------------------------------------------------------------------
// Test 2: multi-head FA. B=2, H=4, N=128, D=64.
// ----------------------------------------------------------------------------
static void test_mha() {
    constexpr int B = 2, H = 4, N = 128;
    std::printf("\n[2] Multi-head: B=%d, H=%d, N=%d, D=%d\n", B, H, N, D);

    size_t total = (size_t)B * H * N * D;
    std::vector<float> h_Q(total), h_K(total), h_V(total);
    fill_random(h_Q); fill_random(h_K); fill_random(h_V);

    std::vector<float> h_ref(total);
    for (int bh = 0; bh < B * H; ++bh) {
        host_attention_generic(h_Q.data() + bh * N * D,
                               h_K.data() + bh * N * D,
                               h_V.data() + bh * N * D,
                               h_ref.data() + bh * N * D,
                               N, N, D, /*causal=*/false);
    }

    float *d_Q, *d_K, *d_V, *d_O;
    size_t bytes = total * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    launch_flash_mha(d_Q, d_K, d_V, d_O, B, H, N);
    CUDA_CHECK_LAST();

    std::vector<float> h_got(total);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, bytes, cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "flash MHA");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

// ----------------------------------------------------------------------------
// Test 3: causal multi-head FA.
// ----------------------------------------------------------------------------
static void test_causal() {
    constexpr int B = 2, H = 4, N = 128;
    std::printf("\n[3] Causal MHA: B=%d, H=%d, N=%d, D=%d\n", B, H, N, D);

    size_t total = (size_t)B * H * N * D;
    std::vector<float> h_Q(total), h_K(total), h_V(total);
    fill_random(h_Q); fill_random(h_K); fill_random(h_V);

    std::vector<float> h_ref(total);
    for (int bh = 0; bh < B * H; ++bh) {
        host_attention_generic(h_Q.data() + bh * N * D,
                               h_K.data() + bh * N * D,
                               h_V.data() + bh * N * D,
                               h_ref.data() + bh * N * D,
                               N, N, D, /*causal=*/true);
    }

    float *d_Q, *d_K, *d_V, *d_O;
    size_t bytes = total * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    launch_flash_causal(d_Q, d_K, d_V, d_O, B, H, N);
    CUDA_CHECK_LAST();

    std::vector<float> h_got(total);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, bytes, cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "flash causal MHA");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

// ----------------------------------------------------------------------------
// Test 4: KV-cache. T_max=128, T=97 (so we exercise both the in-bounds and
// out-of-bounds tile branches, including a partially-filled final tile).
// ----------------------------------------------------------------------------
static void test_kvcache() {
    constexpr int B = 2, H = 4, T_max = 128, T = 97;
    std::printf("\n[4] KV-cache: B=%d, H=%d, T_max=%d, T=%d, D=%d\n",
                B, H, T_max, T, D);

    size_t q_total  = (size_t)B * H * 1     * D;
    size_t kv_total = (size_t)B * H * T_max * D;
    std::vector<float> h_Q(q_total), h_K(kv_total), h_V(kv_total);
    fill_random(h_Q); fill_random(h_K); fill_random(h_V);

    std::vector<float> h_ref(q_total);
    for (int bh = 0; bh < B * H; ++bh) {
        host_attention_generic(h_Q.data() + bh * 1     * D,
                               h_K.data() + bh * T_max * D,
                               h_V.data() + bh * T_max * D,
                               h_ref.data() + bh * 1   * D,
                               1, T, D, /*causal=*/false);
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, q_total  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, kv_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, kv_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, q_total  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_total  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), kv_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), kv_total * sizeof(float), cudaMemcpyHostToDevice));

    launch_flash_kvcache(d_Q, d_K, d_V, d_O, B, H, T_max, T);
    CUDA_CHECK_LAST();

    std::vector<float> h_got(q_total);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, q_total * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "flash kv-cache");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

// ----------------------------------------------------------------------------
// Test 5: GQA. H_q=8, H_kv=2 → group size 4.
// ----------------------------------------------------------------------------
static void test_gqa() {
    constexpr int B = 2, H_q = 8, H_kv = 2, N = 128;
    static_assert(H_q % H_kv == 0, "GQA requires H_q divisible by H_kv");
    std::printf("\n[5] GQA: B=%d, H_q=%d, H_kv=%d, N=%d, D=%d\n",
                B, H_q, H_kv, N, D);

    size_t q_total  = (size_t)B * H_q  * N * D;
    size_t kv_total = (size_t)B * H_kv * N * D;
    std::vector<float> h_Q(q_total), h_K(kv_total), h_V(kv_total);
    fill_random(h_Q); fill_random(h_K); fill_random(h_V);

    int g = H_q / H_kv;
    std::vector<float> h_ref(q_total);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H_q; ++h) {
            int h_kv = h / g;
            int bh_q  = b * H_q  + h;
            int bh_kv = b * H_kv + h_kv;
            host_attention_generic(h_Q.data() + bh_q  * N * D,
                                   h_K.data() + bh_kv * N * D,
                                   h_V.data() + bh_kv * N * D,
                                   h_ref.data() + bh_q * N * D,
                                   N, N, D, /*causal=*/false);
        }
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, q_total  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, kv_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, kv_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, q_total  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), q_total  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), kv_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), kv_total * sizeof(float), cudaMemcpyHostToDevice));

    launch_flash_gqa(d_Q, d_K, d_V, d_O, B, H_q, H_kv, N);
    CUDA_CHECK_LAST();

    std::vector<float> h_got(q_total);
    CUDA_CHECK(cudaMemcpy(h_got.data(), d_O, q_total * sizeof(float), cudaMemcpyDeviceToHost));
    check(h_got, h_ref, "flash GQA");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

int main() {
    test_single_head();
    test_warp_coop();
    test_wmma_flash();
    test_async_wmma_flash();
    test_mma_flash();
    test_mma_ldmatrix_flash();
    test_mma_swizzled_flash();
    test_mha();
    test_causal();
    test_kvcache();
    test_gqa();
    return 0;
}
