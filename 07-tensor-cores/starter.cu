// Module 7 — starter scaffold. Solve the TODOs.
//
// The skeleton below handles block/warp tiling and cooperative shared-memory
// loads — same shape as Module 6's v6 (BM=BN=128, BK=16, 4 warps/block, each
// warp owns a 64x64 tile). The new work — replacing v6's `dotIdx` FMA inner
// loop with WMMA fragments — is in three TODOs.

#include <cuda_fp16.h>
#include <mma.h>

#include "gemm_tc.h"

using namespace nvcuda::wmma;

template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_v0_wmma(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float* __restrict__ C,
                             int m, int n, int k) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int NUM_WARPS   = (BM * BN) / (WM * WN);
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int WMITER      = WM / WMMA_M;     // 4 with WM=64
    constexpr int WNITER      = WN / WMMA_N;     // 4 with WN=64

    int warpId  = threadIdx.x / 32;
    int warpRow = warpId / (BN / WN);
    int warpCol = warpId % (BN / WN);

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * n + cCol * BN + warpCol * WN;

    constexpr int VEC = 8;
    int innerRowA = threadIdx.x / (BK / VEC);
    int innerColA = (threadIdx.x % (BK / VEC)) * VEC;
    int innerRowB = threadIdx.x / (BN / VEC);
    int innerColB = (threadIdx.x % (BN / VEC)) * VEC;
    constexpr int strideA = NUM_THREADS / (BK / VEC);
    constexpr int strideB = NUM_THREADS / (BN / VEC);

    // ========================================================================
    // TODO 1: declare a 2D array of accumulator fragments, sized [WMITER][WNITER].
    //   fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    // and zero them all using fill_fragment(c_frag[i][j], 0.0f).
    // ========================================================================

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        for (int off = 0; off < BM; off += strideA) {
            *reinterpret_cast<int4*>(&As[(innerRowA + off) * BK + innerColA]) =
                *reinterpret_cast<const int4*>(&A[(innerRowA + off) * k + innerColA]);
        }
        for (int off = 0; off < BK; off += strideB) {
            *reinterpret_cast<int4*>(&Bs[(innerRowB + off) * BN + innerColB]) =
                *reinterpret_cast<const int4*>(&B[(innerRowB + off) * n + innerColB]);
        }
        __syncthreads();

        A += BK;
        B += BK * n;

        // ====================================================================
        // TODO 2: replace v6's `for (int dotIdx = 0; dotIdx < BK; ++dotIdx)`
        // FMA inner loop with WMMA. Walk kk over BK in steps of WMMA_K. For
        // each kk:
        //   for wm in [0, WMITER):
        //     load_matrix_sync(a_frag, &As[(warpRow*WM + wm*WMMA_M)*BK + kk], BK)
        //     for wn in [0, WNITER):
        //       load_matrix_sync(b_frag, &Bs[kk*BN + warpCol*WN + wn*WMMA_N], BN)
        //       mma_sync(c_frag[wm][wn], a_frag, b_frag, c_frag[wm][wn])
        // a_frag is matrix_a row_major __half; b_frag is matrix_b row_major __half.
        // ====================================================================

        __syncthreads();
    }

    // ========================================================================
    // TODO 3: store each c_frag[wm][wn] into C with stride n, mem_row_major.
    //   store_matrix_sync(&C[(wm*WMMA_M)*n + wn*WMMA_N], c_frag[wm][wn], n,
    //                     mem_row_major);
    // ========================================================================
}

inline void launch_v0(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;             // 4 warps/block (matches v6)
    dim3 block(((BM * BN) / (WM * WN)) * 32);    // 128 threads
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v0_wmma<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, m, n, k);
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);

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

    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    launch_v0(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%-22s", "7.0 wmma");
    verify(d_C, d_ref, M * N);

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
