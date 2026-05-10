// Module 2 — reference solution.
//
// Three flavours of element-wise copy that move the same bytes but differ in access
// pattern. Compare their bandwidth in bench.cu.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

// Coalesced: lane i of each warp reads element i of a contiguous run.
__global__ void copy_scalar(const float* __restrict__ a,
                            float* __restrict__ b,
                            int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        b[gid] = a[gid];
    }
}

// Strided reads: each warp's 32 lanes read 32 distinct cache lines — and crucially,
// each warp reads a *unique* range so other warps can't bring those lines into L2
// for it. Useful payload per fetched line is one float (4 B / 128 B fetched).
__global__ void copy_strided(const float* __restrict__ a,
                             float* __restrict__ b,
                             int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int lane    = threadIdx.x & 31;
    int warp_id = gid >> 5;
    long long idx = ((long long)warp_id * 32 + lane) * 32;
    idx = idx % n;
    b[gid] = a[idx];
}

// Vectorized coalesced: each thread moves 16 bytes per instruction via float4.
// n must be a multiple of 4 and pointers must be 16-byte aligned (cudaMalloc gives
// 256-byte alignment, so we're fine).
__global__ void copy_vec4(const float4* __restrict__ a,
                          float4* __restrict__ b,
                          int n4) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n4) {
        b[gid] = a[gid];
    }
}

int main() {
    constexpr int N   = 1 << 25;        // 32M floats = 128 MB > L2 (72 MB)
    constexpr int BLK = 256;
    const     int GRD = (N + BLK - 1) / BLK;

    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; ++i) h_a[i] = static_cast<float>(i);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Coalesced.
    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
    copy_scalar<<<GRD, BLK>>>(d_a, d_b, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    int errs = 0;
    for (int i = 0; i < N; ++i) {
        if (h_b[i] != h_a[i]) ++errs;
    }
    std::printf("copy_scalar:  errors=%d\n", errs);

    // Strided.
    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
    copy_strided<<<GRD, BLK>>>(d_a, d_b, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    errs = 0;
    for (int gid = 0; gid < N; ++gid) {
        int lane    = gid & 31;
        int warp_id = gid >> 5;
        long long idx = ((long long)warp_id * 32 + lane) * 32;
        idx = idx % N;
        if (h_b[gid] != h_a[idx]) ++errs;
    }
    std::printf("copy_strided: errors=%d\n", errs);

    // Vec4.
    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
    int N4 = N / 4;
    int GRD4 = (N4 + BLK - 1) / BLK;
    copy_vec4<<<GRD4, BLK>>>(reinterpret_cast<float4*>(d_a),
                             reinterpret_cast<float4*>(d_b),
                             N4);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
    errs = 0;
    for (int i = 0; i < N; ++i) {
        if (h_b[i] != h_a[i]) ++errs;
    }
    std::printf("copy_vec4:    errors=%d\n", errs);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
