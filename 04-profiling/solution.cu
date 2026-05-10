// Module 4 — reference fix.
//
// The "permutation" was scattering each warp's lanes across 32 distinct cache lines.
// Sector efficiency in `ncu` would have shown ~3 % (4 useful bytes per 128-byte
// fetch). The fix: drop the permutation. Once gid increments by 1 per lane, reads
// are coalesced and we're back to ~95 % of DRAM peak.

#include <cstdio>
#include <vector>

#include "cuda_utils.h"

__global__ void saxpy_fixed(const float* __restrict__ x,
                            float* __restrict__ y,
                            float a, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        y[gid] = a * x[gid] + y[gid];
    }
}

int main() {
    constexpr int N   = 1 << 25;
    constexpr int BLK = 256;
    const     int GRD = (N + BLK - 1) / BLK;

    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) { h_x[i] = static_cast<float>(i); h_y[i] = 0.0f; }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    saxpy_fixed<<<GRD, BLK>>>(d_x, d_y, 2.0f, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("ran solution saxpy.\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
