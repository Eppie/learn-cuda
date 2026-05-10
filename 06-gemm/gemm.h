#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
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

// Compute C = A * B with cuBLAS, treating row-major M×K, K×N, M×N inputs by
// swapping arguments and dimensions (cuBLAS is column-major).
inline void cublas_sgemm_rowmajor(cublasHandle_t handle,
                                  const float* A, const float* B, float* C,
                                  int m, int n, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             B, n,    // B is "n × k" col-major (= row-major k × n)
                             A, k,    // A is "k × m" col-major (= row-major m × k)
                             &beta,
                             C, n));  // C is "n × m" col-major (= row-major m × n)
}

inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::srand(seed);
    for (auto& x : v) x = static_cast<float>((std::rand() % 2001) - 1000) * 1e-3f;
}

// Compare two device buffers element-wise. Pass if every element satisfies
//   |actual - expected| <= max(rel_tol * |expected|, abs_tol).
// FP32 dot products of 4096 elements typically accumulate 1e-3 absolute error,
// so a pure relative test fails for near-zero outputs even when the kernel is
// correct. The combined tolerance handles both regimes.
inline bool verify(const float* d_actual, const float* d_expected, int n,
                   float rel_tol = 1e-2f, float abs_tol = 1e-3f) {
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
