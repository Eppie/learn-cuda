// Module 4 — Exercise A: diagnose-this kernel.
//
// This kernel claims to compute y[i] = a * x[i] + y[i]  ("saxpy"),
// but instead it computes y[i] = a * x[perm(i)] + y[i] for a scattering perm().
//
// It is therefore *wrong* (the host-side verify below will say FAIL) AND much slower
// than a coalesced saxpy. Your job: profile it with `ncu --set full ./starter`,
// identify the issue, and fix it here.
//
// The expected fix is small. Once it's right, ./starter should print PASS and run
// ~5–10x faster.
//
// Hint: think about what the 32 lanes of a warp see when threadIdx.x runs 0..31.

#include <cstdio>
#include <vector>

#include "cuda_utils.h"

__global__ void saxpy_with_a_twist(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   float a, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // The "permutation": each warp's 32 lanes are scattered across 32 cache lines.
    int lane    = threadIdx.x & 31;
    int warp_id = gid >> 5;
    long long idx = ((long long)warp_id * 32 + lane) * 32;
    idx = idx % n;

    y[gid] = a * x[idx] + y[gid];
}

int main() {
    constexpr int N   = 1 << 25;        // 32M floats = 128 MB
    constexpr int BLK = 256;
    const     int GRD = (N + BLK - 1) / BLK;
    constexpr float A = 2.0f;

    std::vector<float> h_x(N), h_y(N), h_y_init(N);
    for (int i = 0; i < N; ++i) {
        // x[i] = deterministic non-zero so perm-vs-coalesced gives different
        // answers (otherwise both produce the right output and verify wouldn't
        // catch the bug).
        unsigned ux = static_cast<unsigned>(i) * 1664525u + 1013904223u;
        unsigned uy = static_cast<unsigned>(i) * 22695477u + 1u;
        h_x[i]      = static_cast<float>(ux & 0xffff) * 1e-3f;
        h_y[i]      = static_cast<float>(uy & 0xffff) * 1e-3f;
        h_y_init[i] = h_y[i];
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    saxpy_with_a_twist<<<GRD, BLK>>>(d_x, d_y, A, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Host-side verify: a correct saxpy should produce y[i] = A * x[i] + y_init[i].
    int errors = 0;
    int first_err = -1;
    for (int i = 0; i < N; ++i) {
        float expected = A * h_x[i] + h_y_init[i];
        if (std::abs(h_y[i] - expected) > 1e-3f) {
            if (first_err < 0) first_err = i;
            ++errors;
        }
    }

    if (errors == 0) {
        std::printf("saxpy_with_a_twist: errors=0  (PASS)\n");
        std::printf("Now check the throughput with: ncu --set full ./starter\n");
    } else {
        std::printf("saxpy_with_a_twist: errors=%d  (FAIL)\n", errors);
        std::printf("  first mismatch at i=%d: got %f, expected %f\n",
                    first_err, h_y[first_err], A * h_x[first_err] + h_y_init[first_err]);
        std::printf("Profile and fix: ncu --set full ./starter\n");
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return errors == 0 ? 0 : 1;
}
