// Module 6 — starter scaffold. Solve the eight TODOs in order. Each one is
// expected to produce identical output to cuBLAS within FP32 tolerance.
//
// Convention: blockIdx.x -> column tile, blockIdx.y -> row tile (uniform across
// all kernels in this module — see README §1).

#include <cstring>

#include "gemm.h"
#include "cuda_utils.h"

// ============================================================================
// TODO 6.0 — Naive (intentionally bad coalescing).
// Map threadIdx.x to the *row* of C (varies fastest within a warp), threadIdx.y
// to the column. With block (32,32), 32 lanes within a warp will read 32
// different rows of A — uncoalesced.
// ============================================================================
__global__ void gemm_v0_naive(const float* A, const float* B, float* C,
                              int m, int n, int k) {
    // your code here
}

inline void launch_v0(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);   // x: cols, y: rows
    gemm_v0_naive<<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.1 — Coalesced (1D block, derived row/col).
// Use a 1D block of 32*32 = 1024 threads.
//   row = blockIdx.y * 32 + threadIdx.x / 32   (slow within warp)
//   col = blockIdx.x * 32 + threadIdx.x % 32   (fast within warp)
// ============================================================================
__global__ void gemm_v1_coalesced(const float* A, const float* B, float* C,
                                  int m, int n, int k) {
    // your code here
}

inline void launch_v1(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 block(32 * 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    gemm_v1_coalesced<<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.2 — Shared-memory blocking.
// Each block computes a 32×32 tile of C by walking K in chunks of BS = 32.
// Cooperatively load As (32×32) and Bs (32×32) into __shared__ each iteration.
// ============================================================================
template <int BS>
__global__ void gemm_v2_shared(const float* A, const float* B, float* C,
                               int m, int n, int k) {
    // your code here
}

inline void launch_v2(const float* A, const float* B, float* C, int m, int n, int k) {
    constexpr int BS = 32;
    dim3 block(BS * BS);
    dim3 grid((n + BS - 1) / BS, (m + BS - 1) / BS);
    gemm_v2_shared<BS><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.3 — 1D thread tiling.
// Each thread computes TM elements of C (a column of length TM). Block has
// (BM/TM) × BN = 8 × 64 = 512 threads. Tile sizes BM=64, BN=64, BK=8, TM=8.
//   inner loop:
//     for dotIdx in [0, BK):
//       Btmp = Bs[dotIdx * BN + threadCol]
//       for t in [0, TM):
//         threadResults[t] += As[(threadRow*TM + t)*BK + dotIdx] * Btmp
// ============================================================================
template <int BM, int BN, int BK, int TM>
__global__ void gemm_v3_1d_tiling(const float* A, const float* B, float* C,
                                  int m, int n, int k) {
    // your code here
}

inline void launch_v3(const float* A, const float* B, float* C, int m, int n, int k) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v3_1d_tiling<BM, BN, BK, TM><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.4 — 2D thread tiling.
// Each thread computes a TM×TN tile of C. Tile sizes BM=BN=128, BK=8, TM=TN=8,
// 256 threads per block. Use regM[TM] and regN[TN] as small register caches
// inside the inner-product loop so each shared-mem load feeds TM×TN multiplies.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v4_2d_tiling(const float* A, const float* B, float* C,
                                  int m, int n, int k) {
    // your code here
}

inline void launch_v4(const float* A, const float* B, float* C, int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v4_2d_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.5a — Vectorized loads (As still row-major).
// Same as 6.4 but use float4 (LDG.E.128) for the global loads of A and B
// into shared memory. The inner-loop access of As is still strided (stride BK),
// so it does NOT vectorize — that's the next step.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v5a_vectorized(const float* A, const float* B, float* C,
                                    int m, int n, int k) {
    // your code here
}

inline void launch_v5a(const float* A, const float* B, float* C, int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v5a_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.5b — Vectorized loads + transposed As.
// Store As transposed: As[BK * BM] indexed As[k][m]. Then the inner-loop load
//   regM[i] = As[dotIdx * BM + threadRow * TM + i]
// is contiguous across `i`, which lets the compiler emit LDS.128.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v5b_vectorized_transposed(const float* A, const float* B, float* C,
                                               int m, int n, int k) {
    // your code here
}

inline void launch_v5b(const float* A, const float* B, float* C, int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v5b_vectorized_transposed<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// TODO 6.6 — Full warp tiling (Boehm's kernel 10).
// Block tile -> warp tile -> sub-tile -> thread tile.
//   block tile      BM × BN
//     warp tile     WM × WN              (NUM_WARPS = (BM*BN)/(WM*WN))
//       sub-tile    WSUBM × WSUBN        iterated WMITER × WNITER per warp
//         thread tile TM × TN            (one per lane per sub-tile)
//
// Each warp now iterates over WMITER × WNITER sub-tiles per outer K-step,
// reusing regM/regN across sub-tiles for ILP. Suggested config:
//   BM=128, BN=128, BK=16
//   WM=64,  WN=64                4 warps/block
//   WMITER=2, WNITER=2           4 sub-tiles per warp
//   TM=4,   TN=8                 (WSUBM/TM) * (WSUBN/TN) = 8*4 = 32 lanes/sub-tile
// ============================================================================
template <int BM, int BN, int BK,
          int WM, int WN, int WMITER, int WNITER,
          int TM, int TN>
__global__ void gemm_v6_warptiling(const float* A, const float* B, float* C,
                                   int m, int n, int k) {
    // your code here
}

inline void launch_v6(const float* A, const float* B, float* C, int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    constexpr int WMITER = 2, WNITER = 2;
    constexpr int TM = 4,   TN = 8;
    dim3 block(((BM * BN) / (WM * WN)) * 32);   // 128
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v6_warptiling<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// Verification main — compares each kernel's output to cuBLAS.
// Pass --small / -s for a 512^3 problem (fast correctness checks).
// ============================================================================
int main(int argc, char** argv) {
    int m = M, n = N, k = K;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--small") == 0 || std::strcmp(argv[i], "-s") == 0) {
            m = n = k = 512;
        }
    }

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
