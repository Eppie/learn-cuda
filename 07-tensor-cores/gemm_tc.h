#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

#include "cuda_utils.h"

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS error %s:%d code %d\n",            \
                         __FILE__, __LINE__, (int)_s);                      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// FP16 GEMM via cuBLAS: A (m × k) FP16 row-major, B (k × n) FP16 row-major,
// C (m × n) FP32 row-major. Internally cuBLAS expects column-major; we swap
// args to compute C = A·B with row-major inputs.
inline void cublas_hgemm_fp32acc(cublasHandle_t handle,
                                 const __half* A, const __half* B, float* C,
                                 int m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,
                              &alpha,
                              B, CUDA_R_16F, n,
                              A, CUDA_R_16F, k,
                              &beta,
                              C, CUDA_R_32F, n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// FP16 input + FP16 accumulator + FP16 output. ~2× the throughput of the
// FP32-acc path on Ada at the cost of accumulator precision.
inline void cublas_hgemm_fp16acc(cublasHandle_t handle,
                                 const __half* A, const __half* B, __half* C,
                                 int m, int n, int k) {
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, m, k,
                              &alpha,
                              B, CUDA_R_16F, n,
                              A, CUDA_R_16F, k,
                              &beta,
                              C, CUDA_R_16F, n,
                              CUBLAS_COMPUTE_16F,
                              CUBLAS_GEMM_DEFAULT));
}

inline void fill_random_fp16(std::vector<__half>& v, unsigned seed) {
    std::srand(seed);
    for (auto& x : v) {
        float f = static_cast<float>((std::rand() % 2001) - 1000) * 1e-3f;
        x = __float2half(f);
    }
}

// Default tolerance rel=2e-2, abs=1e-3 — matches measured worst-case max_rel
// of ~3e-5 on FP16-input/FP32-acc GEMM at 4096³ with 600× safety margin. The
// previous default (5e-2) was loose enough to mask real bugs.
inline bool verify(const float* d_actual, const float* d_expected, int n,
                   float rel_tol = 2e-2f, float abs_tol = 1e-3f) {
    std::vector<float> a(n), e(n);
    CUDA_CHECK(cudaMemcpy(a.data(), d_actual,   n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(e.data(), d_expected, n * sizeof(float), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    int   bad_idx = -1;
    for (int i = 0; i < n; ++i) {
        float d       = std::fabs(a[i] - e[i]);
        float allowed = std::fmax(rel_tol * std::fabs(e[i]), abs_tol);
        if (d > allowed && bad_idx < 0) bad_idx = i;
        if (d > max_abs) max_abs = d;
    }
    bool ok = (bad_idx < 0);
    std::printf("    max_abs=%.3e", max_abs);
    if (!ok) {
        std::printf("  (worst at %d: got %.6f, expected %.6f)",
                    bad_idx, a[bad_idx], e[bad_idx]);
    }
    std::printf("  %s\n", ok ? "(PASS)" : "(FAIL)");
    return ok;
}

inline bool verify_half(const __half* d_actual, const __half* d_expected, int n,
                        float rel_tol = 5e-2f, float abs_tol = 1e-2f) {
    std::vector<__half> a(n), e(n);
    CUDA_CHECK(cudaMemcpy(a.data(), d_actual,   n * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(e.data(), d_expected, n * sizeof(__half), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f;
    int   bad_idx = -1;
    for (int i = 0; i < n; ++i) {
        float af = __half2float(a[i]);
        float ef = __half2float(e[i]);
        float d  = std::fabs(af - ef);
        float allowed = std::fmax(rel_tol * std::fabs(ef), abs_tol);
        if (d > allowed && bad_idx < 0) bad_idx = i;
        if (d > max_abs) max_abs = d;
    }
    bool ok = (bad_idx < 0);
    std::printf("    max_abs=%.3e", max_abs);
    if (!ok) {
        std::printf("  (worst at %d: got %.4f, expected %.4f)", bad_idx,
                    __half2float(a[bad_idx]), __half2float(e[bad_idx]));
    }
    std::printf("  %s\n", ok ? "(PASS)" : "(FAIL)");
    return ok;
}
