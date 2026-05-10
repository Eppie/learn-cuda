// Module 8 — perf comparison: sync, async (legacy + modern), vs cuBLAS.
//
// Two benches run:
//   1. Single-size sweep at M=N=K=4096: legacy 2/3/4-stage, modern 2/3-stage,
//      cuBLAS. Each kernel is verified vs cuBLAS before its TFLOPs are
//      reported (no silently-wrong "fast" results).
//   2. Scaling study at M=N=K ∈ {512, 1024, 2048, 4096, 8192}: shows how
//      the cuBLAS-vs-our-kernel gap shifts with problem size — at small K
//      cuBLAS's pipeline-startup overhead is visible; at large K it
//      asymptotically dominates because of better software pipelining.

#include <cstdio>

#include "bench.h"
#include "../07-tensor-cores/gemm_tc.h"
#include "kernels.cuh"

template <typename Launch>
static float bench_one(Launch&& launch, const __half* A, const __half* B, float* C,
                       int m, int n, int k) {
    constexpr int ITERS = 10, WARMUP = 2;
    for (int i = 0; i < WARMUP; ++i) launch(A, B, C, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < ITERS; ++i) {
        GpuTimer t;
        t.start();
        launch(A, B, C, m, n, k);
        best = std::min(best, t.stop_ms());
    }
    return best;
}

template <typename Launch>
static float bench_cublas(Launch&& launch) {
    constexpr int ITERS = 10, WARMUP = 2;
    for (int i = 0; i < WARMUP; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < ITERS; ++i) {
        GpuTimer t;
        t.start();
        launch();
        best = std::min(best, t.stop_ms());
    }
    return best;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // -- buffers sized for the largest problem in the scaling study --
    constexpr int MAX_NK = 8192;
    std::vector<__half> h_A(MAX_NK * MAX_NK), h_B(MAX_NK * MAX_NK);
    fill_random_fp16(h_A, 1234);
    fill_random_fp16(h_B, 5678);

    __half *d_A, *d_B;
    float  *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A,   (size_t)MAX_NK * MAX_NK * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B,   (size_t)MAX_NK * MAX_NK * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C,   (size_t)MAX_NK * MAX_NK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, (size_t)MAX_NK * MAX_NK * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)MAX_NK * MAX_NK * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), (size_t)MAX_NK * MAX_NK * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ============================================================
    // Bench 1: kernel comparison at M=N=K=4096
    // ============================================================
    {
        constexpr int M_ = 4096, N_ = 4096, K_ = 4096;
        std::printf("=== Kernel comparison at %dx%dx%d FP16 -> FP32 ===\n", M_, N_, K_);
        std::printf("Total FLOPs: %.1f G\n\n", 2.0 * M_ * N_ * K_ / 1.0e9);

        // verify all kernels first
        cublas_hgemm_fp32acc(handle, d_A, d_B, d_ref, M_, N_, K_);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto verify_kernel = [&](const char* name, auto&& launch) {
            CUDA_CHECK(cudaMemset(d_C, 0, (size_t)M_ * N_ * sizeof(float)));
            launch(d_A, d_B, d_C, M_, N_, K_);
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaDeviceSynchronize());
            std::printf("verify %-26s", name);
            if (!verify(d_C, d_ref, M_ * N_)) std::exit(1);
        };
        verify_kernel("8.0 sync wmma",            launch_v0);
        verify_kernel("8.1 legacy 2-stage",       launch_v1);
        verify_kernel("8.2 legacy 3-stage",       launch_v2);
        verify_kernel("8.3 legacy 4-stage",       launch_v3);
        verify_kernel("8.4 modern (pipe) 2-stg",  launch_modern_2);
        verify_kernel("8.5 modern (pipe) 3-stg",  launch_modern_3);
        std::printf("\n");

        struct Row { const char* name; float ms; };
        Row rows[] = {
            {"8.0 sync wmma",            bench_one(launch_v0, d_A, d_B, d_C, M_, N_, K_)},
            {"8.1 legacy 2-stage",       bench_one(launch_v1, d_A, d_B, d_C, M_, N_, K_)},
            {"8.2 legacy 3-stage",       bench_one(launch_v2, d_A, d_B, d_C, M_, N_, K_)},
            {"8.3 legacy 4-stage",       bench_one(launch_v3, d_A, d_B, d_C, M_, N_, K_)},
            {"8.4 modern (pipe) 2-stg",  bench_one(launch_modern_2, d_A, d_B, d_C, M_, N_, K_)},
            {"8.5 modern (pipe) 3-stg",  bench_one(launch_modern_3, d_A, d_B, d_C, M_, N_, K_)},
            {"cuBLAS",                   bench_cublas([&]{
                cublas_hgemm_fp32acc(handle, d_A, d_B, d_C, M_, N_, K_);
            })},
        };

        float cublas_ms = rows[6].ms;
        float total_gflops = 2.0f * M_ * N_ * K_ / 1.0e9f;
        std::printf("  %-26s %-10s %-10s %-10s\n", "kernel", "ms", "TFLOPs", "% of cuBLAS");
        std::printf("  %-26s %-10s %-10s %-10s\n", "------", "--", "------", "-----------");
        for (const Row& r : rows) {
            float tflops = total_gflops / r.ms;
            float pct    = (cublas_ms / r.ms) * 100.0f;
            std::printf("  %-26s %-10.3f %-10.2f %-10.1f\n", r.name, r.ms, tflops, pct);
        }
    }

    // ============================================================
    // Bench 2: scaling study (best-perf legacy 4-stage vs cuBLAS)
    //
    // The claim from §"Where it stops helping": at small K, cuBLAS's
    // pipeline-startup cost dominates and the gap to our kernel narrows.
    // At large K the gap re-opens because cuBLAS has more sophisticated
    // pipelining (warp specialization, deeper pipelines). This bench
    // makes that visible.
    // ============================================================
    {
        std::printf("\n=== Scaling study: 4-stage async vs cuBLAS ===\n");
        std::printf("  %-8s %-12s %-12s %-12s %-12s\n",
                    "size", "ours_ms", "ours_TF", "cublas_ms", "cublas_TF");
        std::printf("  %-8s %-12s %-12s %-12s %-12s\n",
                    "----", "-------", "-------", "---------", "---------");

        const int sizes[] = {512, 1024, 2048, 4096, 8192};
        for (int s : sizes) {
            int M_ = s, N_ = s, K_ = s;
            float gflops = 2.0f * (float)M_ * N_ * K_ / 1.0e9f;
            float ours_ms = bench_one(launch_v3, d_A, d_B, d_C, M_, N_, K_);
            float cu_ms   = bench_cublas([&]{
                cublas_hgemm_fp32acc(handle, d_A, d_B, d_C, M_, N_, K_);
            });
            std::printf("  %-8d %-12.3f %-12.2f %-12.3f %-12.2f\n",
                        s, ours_ms, gflops / ours_ms, cu_ms, gflops / cu_ms);
        }
    }

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
