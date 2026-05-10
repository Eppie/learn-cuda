// Module 6 — runs all GEMM kernels and verifies each one against cuBLAS.
// Default size is M = N = K = 4096 (the headline number); pass `--small`
// (or `-s`) on the command line to run at 512 for fast correctness checks.

#include <cstring>

#include "gemm.h"
#include "kernels.cuh"

int main(int argc, char** argv) {
    int m = M, n = N, k = K;
    bool small = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--small") == 0 || std::strcmp(argv[i], "-s") == 0) {
            small = true;
        }
    }
    if (small) { m = n = k = 512; }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("GEMM %d x %d x %d FP32\n\n", m, n, k);

    std::vector<float> h_A(m * k), h_B(k * n);
    fill_random(h_A, 1234);
    fill_random(h_B, 5678);

    float *d_A, *d_B, *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A,   m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,   k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,   m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cublas_sgemm_rowmajor(handle, d_A, d_B, d_ref, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&](const char* name, auto&& launch) {
        CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));
        launch(d_A, d_B, d_C, m, n, k);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("%-26s", name);
        verify(d_C, d_ref, m * n);
    };

    run("6.0 naive",              launch_v0);
    run("6.1 coalesced",          launch_v1);
    run("6.2 shared",             launch_v2);
    run("6.3 1d_tiling",          launch_v3);
    run("6.4 2d_tiling",          launch_v4);
    run("6.5a vectorized",        launch_v5a);
    run("6.5b vec+transposed_As", launch_v5b);
    run("6.6 warptiling",         launch_v6);

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
