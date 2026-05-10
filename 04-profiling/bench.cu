// Module 4 — runs the buggy and fixed versions of both exercises side-by-side, so
// you can see both before-and-after numbers and the relative cost of each issue.
//
// Each kernel is verified once before timing; PASS/FAIL is printed alongside the
// throughput so silently-wrong versions can't post fake speedups.

#include <cstdio>
#include <cmath>
#include <vector>

#include "bench.h"
#include "cuda_utils.h"

// ---- Exercise A kernels --------------------------------------------------

__global__ void saxpy_buggy(const float* __restrict__ x,
                            float* __restrict__ y,
                            float a, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int lane    = threadIdx.x & 31;
    int warp_id = gid >> 5;
    long long idx = ((long long)warp_id * 32 + lane) * 32;
    idx = idx % n;
    y[gid] = a * x[idx] + y[gid];
}

__global__ void saxpy_fixed(const float* __restrict__ x,
                            float* __restrict__ y,
                            float a, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) y[gid] = a * x[gid] + y[gid];
}

// ---- Exercise B kernels --------------------------------------------------

constexpr int R = 4096;
constexpr int C = 512;

__global__ void row_l2_slow(const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    constexpr int BLK = 64;
    int row = blockIdx.x;
    if (row >= rows) return;

    float local_buf[512];
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        local_buf[c] = v;
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
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[row * cols + c] = local_buf[c] * inv;
    }
}

__global__ void row_l2_fast(const float* __restrict__ x,
                            float* __restrict__ y,
                            int rows, int cols) {
    constexpr int BLK = 256;
    int row = blockIdx.x;
    if (row >= rows) return;

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
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[row * cols + c] = x[row * cols + c] * inv;
    }
}

// ---- Verification helpers ------------------------------------------------

static int verify_saxpy(const float* y, const float* x, const float* y0,
                        float a, int n) {
    int errs = 0;
    for (int i = 0; i < n; ++i) {
        float exp = a * x[i] + y0[i];
        if (std::abs(y[i] - exp) > 1e-3f) ++errs;
    }
    return errs;
}

static int verify_row_l2(const float* y, const float* x, int rows, int cols) {
    int errs = 0;
    for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += double(x[r*cols+c]) * double(x[r*cols+c]);
        float inv = 1.0f / std::sqrt(float(s) + 1e-6f);
        for (int c = 0; c < cols; ++c) {
            float exp = x[r*cols+c] * inv;
            if (std::abs(y[r*cols+c] - exp) > 1e-3f) ++errs;
        }
    }
    return errs;
}

// --------------------------------------------------------------------------

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // ---- Exercise A: saxpy buggy vs fixed -------------------------------
    {
        constexpr int N     = 1 << 25;
        constexpr int BLK   = 256;
        constexpr int ITERS = 50;
        const     int GRD   = (N + BLK - 1) / BLK;
        const     long bytes = static_cast<long>(N) * 3 * sizeof(float);
        constexpr float A = 2.0f;

        std::vector<float> h_x(N), h_y(N), h_y0(N);
        for (int i = 0; i < N; ++i) {
            unsigned ux = static_cast<unsigned>(i) * 1664525u + 1013904223u;
            unsigned uy = static_cast<unsigned>(i) * 22695477u + 1u;
            h_x[i]  = static_cast<float>(ux & 0xffff) * 1e-3f;
            h_y0[i] = static_cast<float>(uy & 0xffff) * 1e-3f;
        }

        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        std::printf("=== Exercise A: saxpy (N=%d) ===\n", N);

        // buggy
        CUDA_CHECK(cudaMemcpy(d_y, h_y0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        saxpy_buggy<<<GRD, BLK>>>(d_x, d_y, A, N);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
        int errs_b = verify_saxpy(h_y.data(), h_x.data(), h_y0.data(), A, N);
        // For timing, freshen y each iteration is fine — the wrong answer is the
        // wrong answer either way; we just want the timing.
        CUDA_CHECK(cudaMemcpy(d_y, h_y0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        float ms_b = bench_min_ms(ITERS, [=] { saxpy_buggy<<<GRD, BLK>>>(d_x, d_y, A, N); });

        // fixed
        CUDA_CHECK(cudaMemcpy(d_y, h_y0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        saxpy_fixed<<<GRD, BLK>>>(d_x, d_y, A, N);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
        int errs_f = verify_saxpy(h_y.data(), h_x.data(), h_y0.data(), A, N);
        CUDA_CHECK(cudaMemcpy(d_y, h_y0.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        float ms_f = bench_min_ms(ITERS, [=] { saxpy_fixed<<<GRD, BLK>>>(d_x, d_y, A, N); });

        auto report = [&](const char* name, int errs, float ms) {
            const char* tag = (errs == 0) ? "PASS" : "FAIL";
            float gbs = bytes / (ms * 1.0e6);
            std::printf("  %-15s [%s] %.3f ms   %.1f GB/s   (%.0f%% of peak)\n",
                        name, tag, ms, gbs, gbs / 1008.0f * 100.0f);
        };
        report("saxpy_buggy", errs_b, ms_b);
        report("saxpy_fixed", errs_f, ms_f);

        std::printf("\n  Profile each with:\n");
        std::printf("    ncu --kernel-name regex:saxpy_buggy --set full ./bench\n");
        std::printf("    ncu --kernel-name regex:saxpy_fixed --set full ./bench\n");

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    // ---- Exercise B: row L2 normalize -----------------------------------
    {
        constexpr int ITERS = 50;
        const long bytes = static_cast<long>(R) * C * 2 * sizeof(float); // r + w

        std::vector<float> h_x(R * C), h_y(R * C);
        for (int i = 0; i < R * C; ++i) {
            unsigned u = static_cast<unsigned>(i) * 1664525u + 1013904223u;
            h_x[i] = static_cast<float>(u & 0xffff) * 1e-4f;
        }

        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, R * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, R * C * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), R * C * sizeof(float), cudaMemcpyHostToDevice));

        std::printf("\n=== Exercise B: row_l2_normalize (R=%d, C=%d) ===\n", R, C);

        row_l2_slow<<<R, 64>>>(d_x, d_y, R, C);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, R * C * sizeof(float), cudaMemcpyDeviceToHost));
        int errs_s = verify_row_l2(h_y.data(), h_x.data(), R, C);
        float ms_s = bench_min_ms(ITERS, [=] { row_l2_slow<<<R, 64>>>(d_x, d_y, R, C); });

        row_l2_fast<<<R, 256>>>(d_x, d_y, R, C);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, R * C * sizeof(float), cudaMemcpyDeviceToHost));
        int errs_f = verify_row_l2(h_y.data(), h_x.data(), R, C);
        float ms_f = bench_min_ms(ITERS, [=] { row_l2_fast<<<R, 256>>>(d_x, d_y, R, C); });

        auto report = [&](const char* name, int errs, float ms) {
            const char* tag = (errs == 0) ? "PASS" : "FAIL";
            float gbs = bytes / (ms * 1.0e6);
            std::printf("  %-15s [%s] %.3f ms   %.1f GB/s\n",
                        name, tag, ms, gbs);
        };
        report("row_l2_slow",   errs_s, ms_s);
        report("row_l2_fast",   errs_f, ms_f);

        std::printf("\n  Profile each with:\n");
        std::printf("    ncu --kernel-name regex:row_l2_slow --set full ./bench\n");
        std::printf("    ncu --kernel-name regex:row_l2_fast --set full ./bench\n");

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    return 0;
}
