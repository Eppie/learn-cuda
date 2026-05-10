// Module 1 — starter scaffold. Solve the TODOs.
// Build:  make starter
// Run:    ./starter

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

// TODO 1: write the vector add kernel.
// Each thread should compute c[gid] = a[gid] + b[gid] for one element.
// Remember the bounds check: gid might be past the end of the array.
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    // your code here
}

int main() {
    constexpr int N   = 1 << 24;
    constexpr int BLK = 256;

    // TODO 2: compute the grid size given N and BLK.
    // Hint: enough blocks to cover N, rounding up.
    const int GRD = 0; // your code here

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

    // TODO 3: launch the kernel with <GRD, BLK>>>.
    // your code here

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
    std::printf("vector_add: N=%d, errors=%d %s\n",
                N, errors, errors == 0 ? "(PASS)" : "(FAIL)");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors == 0 ? 0 : 1;
}
