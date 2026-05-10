// Module 8 — starter scaffold. Solve the TODOs.
//
// The skeleton handles fragments, indexing, and the K loop. Your job is the
// pipelining: replace the synchronous load with __pipeline_memcpy_async + commit
// and a STAGES-deep wait_prior pattern.
//
// (After this works, look at `kernels.cuh`'s `gemm_v_modern` for the same
// kernel re-expressed in the modern `cuda::pipeline` API. The two are
// functionally equivalent on Ada; production code prefers the modern form.)

#include "../07-tensor-cores/gemm_tc.h"

#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

using namespace nvcuda::wmma;

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
__global__ void gemm_v1_async(const __half* __restrict__ A,
                              const __half* __restrict__ B,
                              float* __restrict__ C,
                              int m, int n, int k) {
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int NUM_WARPS   = (BM * BN) / (WM * WN);
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int WMITER      = WM / WMMA_M;
    constexpr int WNITER      = WN / WMMA_N;

    int warpId  = threadIdx.x / 32;
    int warpRow = warpId / (BN / WN);
    int warpCol = warpId % (BN / WN);

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ __half As[STAGES][BM * BK];
    __shared__ __half Bs[STAGES][BK * BN];

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
    // TODO 1: complete this lambda. Each call should issue ALL the async copies
    // (one per (off, innerRow*) iteration) for one (stage, kOffset) pair, but
    // NOT call __pipeline_commit. The caller decides commit boundaries.
    //
    // Each individual copy is:
    //   __pipeline_memcpy_async(smem_dst, gmem_src, sizeof(int4));
    // ========================================================================
    auto issue_loads = [&](int stage, int kOffset) {
        const __half* Ag = A + kOffset;
        const __half* Bg = B + kOffset * n;
        // your code here
        (void)Ag; (void)Bg;
    };

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; ++i)
        for (int j = 0; j < WNITER; ++j)
            fill_fragment(c_frag[i][j], 0.0f);

    // ========================================================================
    // TODO 2: prologue. Issue STAGES-1 stages worth of loads, one commit per
    // stage. Guard each with `if (s * BK < k)` for tiny K.
    // ========================================================================

    int compute_stage = 0;
    int load_stage    = STAGES - 1;

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        // ====================================================================
        // TODO 3: at the top of every iteration, issue the next set of loads
        // (target = load_stage, kOffset = kBlock + (STAGES-1)*BK), then commit.
        // After committing, call __pipeline_wait_prior(STAGES-1) so all but the
        // most recent STAGES-1 batches drain — this leaves the *current* stage
        // ready. Don't forget __syncthreads() after the wait.
        // ====================================================================

        for (int kk = 0; kk < BK; kk += WMMA_K) {
            for (int wm = 0; wm < WMITER; ++wm) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                load_matrix_sync(a_frag,
                                 &As[compute_stage][(warpRow * WM + wm * WMMA_M) * BK + kk],
                                 BK);
                for (int wn = 0; wn < WNITER; ++wn) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                    load_matrix_sync(b_frag,
                                     &Bs[compute_stage][kk * BN + warpCol * WN + wn * WMMA_N],
                                     BN);
                    mma_sync(c_frag[wm][wn], a_frag, b_frag, c_frag[wm][wn]);
                }
            }
        }

        // Inter-iteration fence: next iter's issue_loads writes to load_stage,
        // which under STAGES=2 equals this iter's compute_stage — i.e. the
        // buffer we just MMA-read. We must ensure every warp finished consuming
        // before the next cp.async stores begin.
        __syncthreads();

        compute_stage = (compute_stage + 1) % STAGES;
        load_stage    = (load_stage    + 1) % STAGES;
    }

    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            store_matrix_sync(&C[(wm * WMMA_M) * n + wn * WMMA_N],
                              c_frag[wm][wn], n, mem_row_major);
        }
    }
}

inline void launch_v1(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v1_async<BM, BN, BK, WM, WN, /*STAGES=*/2><<<grid, block>>>(A, B, C, m, n, k);
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
    launch_v1(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%-22s", "8.1 async 2-stage");
    verify(d_C, d_ref, M * N);

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));
    return 0;
}
