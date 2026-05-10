// Module 5 — starter scaffold. Solve the TODOs.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "cuda_utils.h"

constexpr int BLK     = 256;
constexpr int N_WARPS = BLK / 32;
constexpr int V2_GRID = 256;

// TODO 2a: implement a warp-wide sum reduction using __shfl_down_sync.
// After this returns, lane 0 of the warp should hold the sum of all 32 lanes' values.
// Five shuffle+add steps (offsets 16, 8, 4, 2, 1).
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // your code here
    return v;
}

// TODO 2b: implement a block-wide sum reduction.
//   1. Reduce within each warp via warp_reduce_sum.
//   2. Lane 0 of each warp writes to a __shared__ float warp_sums[N_WARPS].
//   3. __syncthreads.
//   4. The first warp loads warp_sums (one per lane, padded with 0) and warp-reduces
//      them. The result is in lane 0 of warp 0.
__device__ __forceinline__ float block_reduce_sum(float v) {
    // your code here
    return v;
}

// TODO 1: classic shared-memory tree reduction.
// Each block loads BLK elements (with bounds-check, identity = 0.0f for OOB),
// __syncthreads, then halves the active range each step.
__global__ void reduce_v0(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    // your code here
}

// TODO 3: one element per thread, then block_reduce_sum.
__global__ void reduce_v1(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    // your code here
}

// TODO 4: grid-stride loop + block_reduce_sum. Launch with V2_GRID blocks.
__global__ void reduce_v2(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    // your code here
}

// TODO 5: rewrite v0 using __reduce_add_sync (sm_80+).
// __reduce_add_sync returns the warp-wide sum in a single REDUX.SUM instruction
// (much faster than five shuffles), but is unsigned-int only. Sum unsigned input.
__global__ void reduce_v0_intrinsic(const unsigned int* __restrict__ in,
                                    unsigned int*       __restrict__ out,
                                    int n) {
    // your code here
}

// TODO 6: per-row sum of an MxN row-major matrix. Launch with one block per row.
__global__ void row_sum_kernel(const float* __restrict__ X,
                               float* __restrict__ out,
                               int rows, int cols) {
    // your code here
}

// TODO 9: online softmax warp reduction. Returns (max, sum-of-exps) valid in lane 0.
// Recurrence:
//     m_new = max(m1, m2);
//     s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new);
// A singleton stream (one element x) is (m=x, s=1).
struct ms_pair { float m; float s; };
__device__ __forceinline__ ms_pair online_softmax_warp(float x) {
    ms_pair p = {x, 1.0f};
    // your code here — five __shfl_down_sync passes, combining two ms_pairs each step
    return p;
}

static float host_sum(const float* parts, int k) {
    float s = 0.0f;
    for (int i = 0; i < k; ++i) s += parts[i];
    return s;
}

int main() {
    constexpr int N = 1 << 24;
    std::vector<float> h_in(N, 1.0f);
    const float expected = static_cast<float>(N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    int max_blocks = (N + BLK - 1) / BLK;
    CUDA_CHECK(cudaMalloc(&d_out, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&](const char* name, int grid, auto&& launch) {
        std::vector<float> h_out(grid);
        launch(grid);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float got = host_sum(h_out.data(), grid);
        std::printf("%-22s got=%.0f expected=%.0f %s\n",
                    name, got, expected,
                    got == expected ? "(PASS)" : "(FAIL)");
    };

    int g0 = (N + BLK - 1) / BLK;
    run("reduce_v0", g0,      [&](int g) { reduce_v0<<<g, BLK>>>(d_in, d_out, N); });
    run("reduce_v1", g0,      [&](int g) { reduce_v1<<<g, BLK>>>(d_in, d_out, N); });
    run("reduce_v2", V2_GRID, [&](int g) { reduce_v2<<<g, BLK>>>(d_in, d_out, N); });

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
