// Module 4 — Exercise B reference fix.
//
// The dominant limiter is (2): register spill from the C-sized local_buf array,
// which causes massive Long Scoreboard stalls (waiting on local-memory loads).
// Even with perfect coalescing and full occupancy, the second pass would still
// be re-reading those values from local memory (= L1 → L2 → DRAM in the worst
// case).
//
// The fix: don't cache the row in registers/local. Re-read from `x` in the second
// pass; reads of `x` are at L1/L2 hot anyway. Trades a re-read of cols floats per
// row (cheap) for a giant local-memory footprint (expensive).
//
// We also widen the block to 256 threads (4× more warps, same shared-memory cost)
// to lift achieved occupancy.

#include <cstdio>
#include <cmath>
#include <vector>

#include "cuda_utils.h"

constexpr int R = 4096;
constexpr int C = 512;

constexpr int BLK = 256;

__global__ void row_l2_normalize_fixed(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // First pass: just sum-of-squares; no buffering.
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        s += v * v;
    }

    __shared__ float ssum[BLK];
    ssum[threadIdx.x] = s;
    __syncthreads();
    for (int off = BLK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    float inv = rsqrtf(ssum[0] + 1e-6f);

    // Second pass: re-read from x, scale, store.
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[row * cols + c] = x[row * cols + c] * inv;
    }
}

int main() {
    std::vector<float> h_x(R * C), h_y(R * C);
    for (int i = 0; i < R * C; ++i) {
        unsigned u = static_cast<unsigned>(i) * 1664525u + 1013904223u;
        h_x[i] = static_cast<float>(u & 0xffff) * 1e-4f;
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, R * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, R * C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), R * C * sizeof(float), cudaMemcpyHostToDevice));

    row_l2_normalize_fixed<<<R, BLK>>>(d_x, d_y, R, C);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, R * C * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int r = 0; r < R; ++r) {
        double s = 0.0;
        for (int c = 0; c < C; ++c) s += double(h_x[r*C+c]) * double(h_x[r*C+c]);
        float inv = 1.0f / std::sqrt(float(s) + 1e-6f);
        for (int c = 0; c < C; ++c) {
            float exp = h_x[r*C+c] * inv;
            if (std::abs(h_y[r*C+c] - exp) > 1e-3f) ++errors;
        }
    }

    std::printf("row_l2_normalize_fixed: errors=%d %s\n",
                errors, errors == 0 ? "(PASS)" : "(FAIL)");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return errors == 0 ? 0 : 1;
}
