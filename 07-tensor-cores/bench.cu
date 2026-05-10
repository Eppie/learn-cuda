// Module 7 — perf comparison: WMMA, swizzled WMMA, raw mma.sync, vs cuBLAS.

#include <cstdio>

#include "bench.h"
#include "gemm_tc.h"
#include "kernels.cuh"

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("GEMM %d x %d x %d FP16 -> FP32\n", M, N, K);
    std::printf("Total FLOPs: %.1f G\n\n", 2.0 * M * N * K / 1.0e9);

    std::vector<__half> h_A(M * K), h_B(K * N);
    fill_random_fp16(h_A, 1234);
    fill_random_fp16(h_B, 5678);

    __half *d_A, *d_B, *d_C16;
    float  *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A,   M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B,   K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C,   M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C16, M * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ref, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // -- pre-bench correctness pass: fail loud before printing TFLOPs --
    cublas_hgemm_fp32acc(handle, d_A, d_B, d_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto verify_kernel = [&](const char* name, auto&& launch) {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        launch(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("verify %-26s", name);
        if (!verify(d_C, d_ref, M * N)) std::exit(1);
    };
    verify_kernel("7.0 wmma fp32-acc",      launch_v0);
    verify_kernel("7.1 wmma swizzled",      launch_v1);
    verify_kernel("7.2 mma.sync (raw PTX)", launch_v2);
    std::printf("\n");

    constexpr int ITERS = 10, WARMUP = 2;

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

    auto bench_kernel_h = [&](auto&& launch) {
        for (int i = 0; i < WARMUP; ++i) launch(d_A, d_B, d_C16, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            launch(d_A, d_B, d_C16, M, N, K);
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    auto bench_cublas_fp32 = [&]() {
        for (int i = 0; i < WARMUP; ++i) cublas_hgemm_fp32acc(handle, d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            cublas_hgemm_fp32acc(handle, d_A, d_B, d_C, M, N, K);
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    auto bench_cublas_fp16 = [&]() {
        for (int i = 0; i < WARMUP; ++i) cublas_hgemm_fp16acc(handle, d_A, d_B, d_C16, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            cublas_hgemm_fp16acc(handle, d_A, d_B, d_C16, M, N, K);
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    struct Row { const char* name; float ms; };
    Row rows[] = {
        {"7.0 wmma fp32-acc",        bench_kernel(launch_v0)},
        {"7.1 wmma swizzled",        bench_kernel(launch_v1)},
        {"7.2 mma.sync (raw PTX)",   bench_kernel(launch_v2)},
        {"7.X wmma fp16-acc",        bench_kernel_h(launch_v0_fp16acc)},
        {"cuBLAS fp32-acc",          bench_cublas_fp32()},
        {"cuBLAS fp16-acc",          bench_cublas_fp16()},
    };

    float cublas_ms = rows[4].ms;
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
    CUDA_CHECK(cudaFree(d_C16));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
