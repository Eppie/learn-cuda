// Module 8 — verifies sync, async (legacy), and async (modern cuda::pipeline)
// WMMA kernels against cuBLAS.

#include "gemm_tc.h"
#include "kernels.cuh"

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("GEMM %d x %d x %d FP16 -> FP32\n\n", M, N, K);

    std::vector<__half> h_A(M * K), h_B(K * N);
    fill_random_fp16(h_A, 1234);
    fill_random_fp16(h_B, 5678);

    __half *d_A, *d_B;
    float  *d_C, *d_ref;
    CUDA_CHECK(cudaMalloc(&d_A,   M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B,   K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C,   M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublas_hgemm_fp32acc(handle, d_A, d_B, d_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&](const char* name, auto&& launch) {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        launch(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("%-30s", name);
        verify(d_C, d_ref, M * N);
    };

    run("8.0 sync wmma",                launch_v0);
    run("8.1 legacy 2-stage",           launch_v1);
    run("8.2 legacy 3-stage",           launch_v2);
    run("8.3 legacy 4-stage",           launch_v3);
    run("8.4 modern (pipeline) 2-stg",  launch_modern_2);
    run("8.5 modern (pipeline) 3-stg",  launch_modern_3);

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
