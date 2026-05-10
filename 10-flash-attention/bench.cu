// Module 10 — bench: naive vs flash attention at increasing sequence lengths.

#include <cstdio>
#include <vector>
#include <cuda_fp16.h>

#include "bench.h"
#include "cuda_utils.h"
#include "kernels.cuh"

static void run_for_size(int N) {
    std::printf("\nN=%d (D=%d)  S matrix would be %.0f MB\n", N, D,
                (double)N * N * sizeof(float) / 1.0e6);

    float *d_Q, *d_K, *d_V, *d_O, *d_S;
    CUDA_CHECK(cudaMalloc(&d_Q, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_Q, 0, N * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K, 0, N * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V, 0, N * D * sizeof(float)));

    constexpr int ITERS = 10, WARMUP = 2;

    auto bench = [&](auto&& fn) {
        for (int i = 0; i < WARMUP; ++i) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            fn();
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    auto naive      = [&] { launch_naive    (d_Q, d_K, d_V, d_O, d_S, N); };
    auto flash      = [&] { launch_flash    (d_Q, d_K, d_V, d_O,      N); };
    auto flash_warp = [&] { launch_flash_warp(d_Q, d_K, d_V, d_O,     N); };

    double flops = 4.0 * N * N * D;          // 2 matmuls of size N×N×D
    auto report = [&](const char* name, float ms) {
        std::printf("  %-32s %.3f ms   %.2f TFLOPs/s\n",
                    name, ms, flops / (ms * 1.0e9));
    };

    report("10.0 flash (one-thread/row)", bench(flash));
    report("10.1 flash warp-cooperative", bench(flash_warp));

    // M10.2 / 10.3 WMMA: FP16 inputs. Uses a separate set of buffers.
    {
        __half *d_Qh, *d_Kh, *d_Vh;
        CUDA_CHECK(cudaMalloc(&d_Qh, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_Kh, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_Vh, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_Qh, 0, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_Kh, 0, N * D * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_Vh, 0, N * D * sizeof(__half)));
        auto flash_wmma       = [&] { launch_flash_wmma      (d_Qh, d_Kh, d_Vh, d_O, N); };
        auto flash_async_wmma = [&] { launch_flash_async_wmma(d_Qh, d_Kh, d_Vh, d_O, N); };
        report("10.2 flash WMMA (fp16 in)",        bench(flash_wmma));
        report("10.3 flash cp.async + WMMA",       bench(flash_async_wmma));
        CUDA_CHECK(cudaFree(d_Qh));
        CUDA_CHECK(cudaFree(d_Kh));
        CUDA_CHECK(cudaFree(d_Vh));
    }

    report("naive (3 kernels, S/P)",      bench(naive));

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_S));
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);

    run_for_size(2048);
    run_for_size(4096);
    run_for_size(8192);
    return 0;
}
