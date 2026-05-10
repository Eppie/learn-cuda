// Module 2 — starter scaffold. Solve the TODOs.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

// TODO 1: implement a coalesced element-wise copy: b[i] = a[i].
__global__ void copy_scalar(const float* __restrict__ a,
                            float* __restrict__ b,
                            int n) {
    // your code here
}

// TODO 2: implement a *strided-read* copy.
// The 32 threads of each warp should read 32 distinct cache lines, and warps should
// not share lines (so L2 can't rescue you).
//
// Recipe: let warp_id = gid / 32 and lane = gid % 32. Read element
//   a[((warp_id * 32) + lane) * 32  mod  n]
// and write to b[gid]. Writes stay coalesced; only reads are strided.
__global__ void copy_strided(const float* __restrict__ a,
                             float* __restrict__ b,
                             int n) {
    // your code here
}

// TODO 3: implement a vectorized copy using float4.
// Each thread moves one float4 (16 bytes) per instruction.
__global__ void copy_vec4(const float4* __restrict__ a,
                          float4* __restrict__ b,
                          int n4) {
    // your code here
}

int main() {
    constexpr int N   = 1 << 25;        // 128 MB
    constexpr int BLK = 256;
    const     int GRD = (N + BLK - 1) / BLK;

    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; ++i) h_a[i] = static_cast<float>(i);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto check = [&](const char* name, auto verify) {
        CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, N * sizeof(float), cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < N; ++i) if (!verify(i, h_b[i])) ++errs;
        std::printf("%-15s errors=%d %s\n", name, errs, errs == 0 ? "(PASS)" : "(FAIL)");
    };

    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
    copy_scalar<<<GRD, BLK>>>(d_a, d_b, N);
    CUDA_CHECK_LAST();
    check("copy_scalar:", [&](int i, float v) { return v == h_a[i]; });

    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
    copy_strided<<<GRD, BLK>>>(d_a, d_b, N);
    CUDA_CHECK_LAST();
    check("copy_strided:", [&](int i, float v) {
        int lane    = i & 31;
        int warp_id = i >> 5;
        long long idx = ((long long)warp_id * 32 + lane) * 32;
        idx = idx % N;
        return v == h_a[idx];
    });

    CUDA_CHECK(cudaMemset(d_b, 0, N * sizeof(float)));
    int N4 = N / 4;
    int GRD4 = (N4 + BLK - 1) / BLK;
    copy_vec4<<<GRD4, BLK>>>(reinterpret_cast<float4*>(d_a),
                             reinterpret_cast<float4*>(d_b),
                             N4);
    CUDA_CHECK_LAST();
    check("copy_vec4:", [&](int i, float v) { return v == h_a[i]; });

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
