// Module 12 — bench: realistic transformer-block attention shape.
//   B=32, H=12, N=2048, D=64 — close to GPT-2 small at 2K context.
//
// Performance criteria (RTX 4090; see README §Performance success criteria):
//   passing  ≥  8 TF/s  (FP32, this kernel as shipped)
//   solid    ≥ 14 TF/s  (FP32 + tile-skip on causal=true)
//   strong   ≥ 25 TF/s  (FP16 inputs, no tensor cores)
//   prod     ≥ 50 TF/s  (FP16 + WMMA inner matmul)
//
// TODO-USER: cuBLAS+cuDNN MHA reference comparison.
// The fair apples-to-apples reference for "is this kernel any good?" is
//   - cuDNN frontend `Graph` with a `SDPA` (scaled dot-product attention) op,
//     OR
//   - the descriptor API `cudnnMultiHeadAttnForward` with weights set to
//     identity so it computes plain attention.
// Either is ~150-200 lines of setup code (backends, weight buffers, KV
// layout, dropout descriptor, etc.). It's a worthy exercise on its own and
// out of scope for this bench by default.
//
// As a faster sanity check, you can also call `cublasSgemmStridedBatched`
// twice (Q@K^T then P@V, with a separate softmax kernel between) and compare
// the unfused total against this fused FA. If your fused FA isn't winning
// by ≥ 2× over the unfused two-GEMM version, the fusion isn't pulling its
// weight — go look for a bug.

#include <cstdio>

#include "bench.h"
#include "cuda_utils.h"
#include "kernels.cuh"

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);

    constexpr int B = 32;
    constexpr int H = 12;
    constexpr int N = 2048;

    long long total = (long long)B * H * N * D;
    std::printf("Shape: B=%d, H=%d, N=%d, D=%d  (%.0f MB per tensor)\n\n",
                B, H, N, D, total * sizeof(float) / 1.0e6);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_Q, 0, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K, 0, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V, 0, total * sizeof(float)));

    constexpr int ITERS = 10, WARMUP = 2;

    auto run = [&](bool causal) {
        for (int i = 0; i < WARMUP; ++i) launch_flash_mhc(d_Q, d_K, d_V, d_O, B, H, N, causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer t;
            t.start();
            launch_flash_mhc(d_Q, d_K, d_V, d_O, B, H, N, causal);
            best = std::min(best, t.stop_ms());
        }
        return best;
    };

    double flops_full = 4.0 * B * H * N * N * D;
    auto report = [&](const char* name, float ms, double flops) {
        std::printf("  %-30s %.3f ms   %.1f GFLOPs/s\n",
                    name, ms, flops / (ms * 1.0e6));
    };

    report("flash_mhc (causal=false)", run(false), flops_full);
    report("flash_mhc (causal=true)",  run(true),  flops_full / 2.0);  // ~half work

    std::printf("\n");
    std::printf("Targets (RTX 4090): passing >= 8 TF/s, solid >= 14, strong >= 25.\n");
    std::printf("To compare vs cuBLAS/cuDNN, see TODO-USER at the top of this file.\n");

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    return 0;
}
