// Module 1 — reference solution.
// Vector add: c[i] = a[i] + b[i].

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

int main() {
    constexpr int N   = 1 << 24;
    constexpr int BLK = 256;
    const     int GRD = (N + BLK - 1) / BLK;

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 1e-3f;
        h_b[i] = static_cast<float>(N - i) * 1e-3f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    vector_add<<<GRD, BLK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::abs(h_c[i] - expected) > 1e-4f) {
            if (errors < 5) {
                std::printf("mismatch at %d: got %f, expected %f\n",
                            i, h_c[i], expected);
            }
            ++errors;
        }
    }
    std::printf("vector_add: N=%d, errors=%d\n", N, errors);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors == 0 ? 0 : 1;
}
