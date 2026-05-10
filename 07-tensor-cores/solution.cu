// Module 7 — verifies the WMMA + mma.sync kernels against cuBLAS FP16 GEMM.

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

    run("7.0 wmma fp32-acc",        launch_v0);
    run("7.1 wmma swizzled",        launch_v1);
    run("7.2 mma.sync (raw PTX)",   launch_v2);

    // FP16-acc stretch: separate FP16 buffers, separate cuBLAS reference.
    __half *d_C16, *d_ref16;
    CUDA_CHECK(cudaMalloc(&d_C16,   M * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ref16, M * N * sizeof(__half)));
    cublas_hgemm_fp16acc(handle, d_A, d_B, d_ref16, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_C16, 0, M * N * sizeof(__half)));
    launch_v0_fp16acc(d_A, d_B, d_C16, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%-30s", "7.X wmma fp16-acc (stretch)");
    verify_half(d_C16, d_ref16, M * N);

    CUDA_CHECK(cudaFree(d_C16));
    CUDA_CHECK(cudaFree(d_ref16));

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
