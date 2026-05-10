// Module 13 — the simple kernel from README §2. Useful for `make ptx` /
// `make sass` to see the layers of compilation.

#include <cstdio>
#include <vector>

#include "cuda_utils.h"

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) c[gid] = a[gid] + b[gid];
}

int main() {
    constexpr int N   = 1 << 20;
    constexpr int BLK = 256;
    const     int GRD = (N + BLK - 1) / BLK;

    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N, 0.0f);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    vector_add<<<GRD, BLK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    int errs = 0;
    for (int i = 0; i < N; ++i) if (h_c[i] != 3.0f) ++errs;
    std::printf("vector_add N=%d errors=%d\n", N, errs);
    std::printf("Inspect with:  make ptx  &&  cat vector_add.ptx\n");
    std::printf("                make sass &&  cat vector_add.sass\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
