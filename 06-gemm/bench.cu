// Module 6 — perf comparison of all GEMM kernels vs cuBLAS.
//
// Runs a verify pass per kernel at startup (PASS/FAIL) before any timing
// numbers are printed. A silently-wrong kernel still posting "50 TFLOPs" is
// the most insidious failure mode; verify-first prevents it.

#include <cstdio>

#include "bench.h"
#include "gemm.h"
#include "kernels.cuh"

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("GEMM %d x %d x %d FP32\n", M, N, K);
    std::printf("Total FLOPs: %.1f G\n\n", 2.0 * M * N * K / 1.0e9);

    std::vector<float> h_A(M * K), h_B(K * N);
    fill_random(h_A, 1234);
    fill_random(h_B, 5678);

    float *d_A, *d_B, *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A,   M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,   K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,   M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Reference once.
    cublas_sgemm_rowmajor(handle, d_A, d_B, d_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Verify pass: every kernel must pass before we trust its timings. ----
    std::printf("Verify (vs cuBLAS):\n");
    auto verify_one = [&](const char* name, auto&& launch) -> bool {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        launch(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("  %-22s", name);
        return verify(d_C, d_ref, M * N);
    };
    bool all_ok = true;
    all_ok &= verify_one("6.0 naive",                launch_v0);
    all_ok &= verify_one("6.1 coalesced",            launch_v1);
    all_ok &= verify_one("6.2 shared",               launch_v2);
    all_ok &= verify_one("6.3 1d_tiling",            launch_v3);
    all_ok &= verify_one("6.4 2d_tiling",            launch_v4);
    all_ok &= verify_one("6.5a vectorized",          launch_v5a);
    all_ok &= verify_one("6.5b vec+transposed_As",   launch_v5b);
    all_ok &= verify_one("6.6 warptiling",           launch_v6);
    if (!all_ok) {
        std::fprintf(stderr, "\nat least one kernel failed verify; aborting bench.\n");
        return 1;
    }
    std::printf("\n");

    // ---- Timing ----
    constexpr int ITERS = 10;
    constexpr int WARMUP = 2;

    auto bench_kernel = [&](auto&& launch) {
        for (int i = 0; i < WARMUP; ++i) launch(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            launch(d_A, d_B, d_C, M, N, K);
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    auto bench_cublas = [&]() {
        for (int i = 0; i < WARMUP; ++i) cublas_sgemm_rowmajor(handle, d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            cublas_sgemm_rowmajor(handle, d_A, d_B, d_C, M, N, K);
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    struct Row {
        const char* name;
        float       ms;
    };

    Row rows[] = {
        {"6.0 naive",              bench_kernel(launch_v0)},
        {"6.1 coalesced",          bench_kernel(launch_v1)},
        {"6.2 shared",             bench_kernel(launch_v2)},
        {"6.3 1d_tiling",          bench_kernel(launch_v3)},
        {"6.4 2d_tiling",          bench_kernel(launch_v4)},
        {"6.5a vectorized",        bench_kernel(launch_v5a)},
        {"6.5b vec+transposed_As", bench_kernel(launch_v5b)},
        {"6.6 warptiling",         bench_kernel(launch_v6)},
        {"cuBLAS",                 bench_cublas()},
    };
    constexpr int NROWS = sizeof(rows) / sizeof(rows[0]);

    float cublas_ms = rows[NROWS - 1].ms;
    float total_gflops = 2.0f * M * N * K / 1.0e9f;

    std::printf("  %-26s %-10s %-10s %-10s\n", "kernel", "ms", "TFLOPs", "% of cuBLAS");
    std::printf("  %-26s %-10s %-10s %-10s\n", "------", "--", "------", "-----------");
    for (const Row& r : rows) {
        float tflops = total_gflops / r.ms;
        float pct    = (cublas_ms / r.ms) * 100.0f;
        std::printf("  %-26s %-10.3f %-10.2f %-10.1f\n", r.name, r.ms, tflops, pct);
    }

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
