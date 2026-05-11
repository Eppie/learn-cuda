// Roofline reference for M10's bench shape: single-head, D=64, FP16 in / FP32 out.
//
// Workload (per the M10 bench): N × N × D FlashAttention forward pass.
// FLOPs counted as 4 · N² · D (Q·K^T + P·V).  Bytes from DRAM: ~5 · N · D · 2.
//
// We measure the two underlying cuBLAS hgemm shapes that a "naive split" attention
// would issue:
//   Q·K^T : hgemm(M=N, N=N, K=D)   — skinny K
//   P·V   : hgemm(M=N, N=D, K=N)   — skinny output
// Both at FP16 input, FP32 accumulator (CUBLAS_COMPUTE_32F).

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA %d:%s\n", __LINE__, cudaGetErrorString(e)); std::exit(1); } } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s = (call); if (s) { \
    std::fprintf(stderr, "cuBLAS %d:%d\n", __LINE__, s); std::exit(1); } } while(0)

static float time_hgemm(cublasHandle_t h, int M, int N, int K,
                        const __half* dA, const __half* dB, float* dC,
                        int iters) {
    const float alpha = 1.0f, beta = 0.0f;
    // Warmup.
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                              &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(e0);
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                                  &beta, dC, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
}

int main() {
    int dev = 0;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    std::printf("Device: %s\n\n", p.name);

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));

    const int D = 64;
    const int Ns[] = {2048, 4096, 8192};

    std::printf("%-6s | %-8s %-8s %-9s | %-8s %-8s %-9s | %-12s %-9s\n",
                "N", "QK^T ms", "TF/s", "%peak",
                "PV ms", "TF/s", "%peak",
                "sum TF/s", "sum/peak");
    std::printf("--------------------------------------------------------------------------------------\n");

    const float TC_FP32_ACC_PEAK = 165.2f;  // RTX 4090 FP16-in / FP32-acc tensor peak (TFLOPs)

    for (int N : Ns) {
        // Q·K^T: hgemm(M=N, K=D, N=N) where output is N×N
        __half *dQ, *dKt, *dP, *dV; float *dS, *dO;
        CUDA_CHECK(cudaMalloc(&dQ,  (size_t)N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dKt, (size_t)D * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dS,  (size_t)N * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(dQ, 0, (size_t)N * D * sizeof(__half)));
        CUDA_CHECK(cudaMemset(dKt, 0, (size_t)D * N * sizeof(__half)));

        float ms_qk = time_hgemm(h, N, N, D, dQ, dKt, dS, 20);
        double flops_qk = 2.0 * (double)N * N * D;
        double tf_qk = flops_qk / (ms_qk * 1e-3) / 1e12;

        // P·V: hgemm(M=N, K=N, N=D)
        CUDA_CHECK(cudaMalloc(&dP,  (size_t)N * N * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dV,  (size_t)N * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&dO,  (size_t)N * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(dP, 0, (size_t)N * N * sizeof(__half)));
        CUDA_CHECK(cudaMemset(dV, 0, (size_t)N * D * sizeof(__half)));

        float ms_pv = time_hgemm(h, N, D, N, dP, dV, dO, 20);
        double flops_pv = 2.0 * (double)N * D * N;
        double tf_pv = flops_pv / (ms_pv * 1e-3) / 1e12;

        // Combined: pretend a single flash kernel did both matmuls back-to-back.
        // The "wall clock" floor for a 2-gemm-equivalent attention.
        double tf_sum = (flops_qk + flops_pv) / ((ms_qk + ms_pv) * 1e-3) / 1e12;

        std::printf("%-6d | %-8.3f %-8.2f %-9.1f | %-8.3f %-8.2f %-9.1f | %-12.2f %-9.1f\n",
                    N, ms_qk, tf_qk, 100.0 * tf_qk / TC_FP32_ACC_PEAK,
                    ms_pv, tf_pv, 100.0 * tf_pv / TC_FP32_ACC_PEAK,
                    tf_sum, 100.0 * tf_sum / TC_FP32_ACC_PEAK);

        cudaFree(dQ); cudaFree(dKt); cudaFree(dS);
        cudaFree(dP); cudaFree(dV); cudaFree(dO);
    }

    std::printf("\nReference peak (FP16 in / FP32 acc, tensor cores): %.1f TF/s\n", TC_FP32_ACC_PEAK);
    std::printf("Note: 'sum TF/s' is the cuBLAS-only floor for a hypothetical 'split flash':\n");
    std::printf("  two cuBLAS hgemms back-to-back with NO softmax cost folded in.  Adding\n");
    std::printf("  the per-row softmax (online, fused) costs nothing extra in arithmetic but\n");
    std::printf("  burns real time on shared-memory traffic.  M10's flash kernels include all\n");
    std::printf("  of that cost in their TF/s number.\n");

    cublasDestroy(h);
    return 0;
}
