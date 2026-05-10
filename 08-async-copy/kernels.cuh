#pragma once

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

using namespace nvcuda::wmma;

// ============================================================================
// 8.0 — gemm_v0_sync: Module 7's WMMA kernel, copied for direct comparison.
// Same block/warp tiling as M07 v0 (4 warps, each 64x64 → 4x4 = 16 fragments).
// ============================================================================
template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_v0_sync(const __half* __restrict__ A,
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

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; ++i)
        for (int j = 0; j < WNITER; ++j)
            fill_fragment(c_frag[i][j], 0.0f);

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

        for (int kk = 0; kk < BK; kk += WMMA_K) {
            for (int wm = 0; wm < WMITER; ++wm) {
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                load_matrix_sync(a_frag,
                                 &As[(warpRow * WM + wm * WMMA_M) * BK + kk], BK);

                for (int wn = 0; wn < WNITER; ++wn) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
                    load_matrix_sync(b_frag,
                                     &Bs[kk * BN + warpCol * WN + wn * WMMA_N], BN);
                    mma_sync(c_frag[wm][wn], a_frag, b_frag, c_frag[wm][wn]);
                }
            }
        }
        __syncthreads();
    }

    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            store_matrix_sync(&C[(wm * WMMA_M) * n + wn * WMMA_N],
                              c_frag[wm][wn], n, mem_row_major);
        }
    }
}

inline void launch_v0(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v0_sync<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 8.1 — gemm_v1_async: legacy `__pipeline_memcpy_async` API, STAGES-deep.
// This is the "old" pipeline API (`<cuda_pipeline.h>`). It still works, still
// emits `cp.async`, and is what a lot of pre-2022 code uses. Compare to
// `gemm_v_modern` below for the cuda::pipeline form most production code uses
// today.
// See M13 § cp.async PTX form for the underlying instruction.
// ============================================================================
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

    auto issue_loads = [&](int stage, int kOffset) {
        const __half* Ag = A + kOffset;
        const __half* Bg = B + kOffset * n;
        for (int off = 0; off < BM; off += strideA) {
            __pipeline_memcpy_async(
                &As[stage][(innerRowA + off) * BK + innerColA],
                &Ag[(innerRowA + off) * k + innerColA],
                sizeof(int4));
        }
        for (int off = 0; off < BK; off += strideB) {
            __pipeline_memcpy_async(
                &Bs[stage][(innerRowB + off) * BN + innerColB],
                &Bg[(innerRowB + off) * n + innerColB],
                sizeof(int4));
        }
    };

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; ++i)
        for (int j = 0; j < WNITER; ++j)
            fill_fragment(c_frag[i][j], 0.0f);

    // Prologue: issue STAGES-1 ahead.
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s * BK < k) issue_loads(s, s * BK);
        __pipeline_commit();
    }

    int compute_stage = 0;
    int load_stage    = STAGES - 1;

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        int next_k = kBlock + (STAGES - 1) * BK;
        if (next_k < k) issue_loads(load_stage, next_k);
        __pipeline_commit();

        __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

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

        // Fence at the end of each iteration: next iteration's `issue_loads`
        // writes to `load_stage`, which is a *different* shared buffer from the
        // one we just MMA-read (`compute_stage` here). However, with STAGES=2
        // the cycle wraps fast: iter N+1's load_stage equals iter N's
        // compute_stage, so we need to ensure every warp finished consuming
        // before the cp.async stores can start overwriting. With STAGES >= 3
        // the buffers don't alias across consecutive iterations, but
        // wait_prior(STAGES-1) at the top of next iter still requires a
        // memory-consistent shared state. __syncthreads is the right cheap
        // answer here.
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

inline void launch_v2(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v1_async<BM, BN, BK, WM, WN, /*STAGES=*/3><<<grid, block>>>(A, B, C, m, n, k);
}

inline void launch_v3(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v1_async<BM, BN, BK, WM, WN, /*STAGES=*/4><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 8.2 — gemm_v_modern: cuda::pipeline + cuda::memcpy_async (the modern API).
//
// Same algorithm, different API. `cuda::pipeline` is the typed cousin of the
// `__pipeline_*` C functions:
//   pipeline.producer_acquire() — wait for a stage slot to be free
//   cuda::memcpy_async(...)     — issue async copy into the slot
//   pipeline.producer_commit()  — close the slot (= __pipeline_commit)
//   pipeline.consumer_wait()    — wait for next consumed stage to be ready
//   pipeline.consumer_release() — release the stage slot
//
// Why prefer it: it's the C++-typed entry point, plays nicely with
// `cuda::barrier` (mbarriers) for finer synchronization than block-wide
// __syncthreads, and is the form CUTLASS / FlashAttention idiom around.
// Functionally equivalent on Ada.
// ============================================================================
template <int BM, int BN, int BK, int WM, int WN, int STAGES>
__global__ void gemm_v_modern(const __half* __restrict__ A,
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

    // Per-thread pipeline state. shared_state must be in shared memory and
    // shared across all threads in the block; cuda::make_pipeline gives us
    // that shape via a thread-block scope.
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pss;
    auto block = cooperative_groups::this_thread_block();
    auto pipe  = cuda::make_pipeline(block, &pss);

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

    auto issue_loads = [&](int stage, int kOffset) {
        const __half* Ag = A + kOffset;
        const __half* Bg = B + kOffset * n;
        // The size hint cuda::aligned_size_t<16> tells the implementation to
        // emit a 16-byte cp.async (cp.async.cg.shared.global ... 16, 16) —
        // the same instruction __pipeline_memcpy_async(.., 16) emits.
        for (int off = 0; off < BM; off += strideA) {
            cuda::memcpy_async(
                &As[stage][(innerRowA + off) * BK + innerColA],
                &Ag[(innerRowA + off) * k + innerColA],
                cuda::aligned_size_t<16>(sizeof(int4)),
                pipe);
        }
        for (int off = 0; off < BK; off += strideB) {
            cuda::memcpy_async(
                &Bs[stage][(innerRowB + off) * BN + innerColB],
                &Bg[(innerRowB + off) * n + innerColB],
                cuda::aligned_size_t<16>(sizeof(int4)),
                pipe);
        }
    };

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; ++i)
        for (int j = 0; j < WNITER; ++j)
            fill_fragment(c_frag[i][j], 0.0f);

    // Prologue.
    for (int s = 0; s < STAGES - 1; ++s) {
        pipe.producer_acquire();
        if (s * BK < k) issue_loads(s, s * BK);
        pipe.producer_commit();
    }

    int compute_stage = 0;

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        int next_k = kBlock + (STAGES - 1) * BK;
        pipe.producer_acquire();
        if (next_k < k) issue_loads((compute_stage + STAGES - 1) % STAGES, next_k);
        pipe.producer_commit();

        pipe.consumer_wait();
        __syncthreads();    // pipe.consumer_wait is per-thread; need block barrier
                             // before reading shared written by sibling threads.

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

        pipe.consumer_release();
        compute_stage = (compute_stage + 1) % STAGES;
    }

    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            store_matrix_sync(&C[(wm * WMMA_M) * n + wn * WMMA_N],
                              c_frag[wm][wn], n, mem_row_major);
        }
    }
}

inline void launch_modern_2(const __half* A, const __half* B, float* C,
                            int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v_modern<BM, BN, BK, WM, WN, /*STAGES=*/2><<<grid, block>>>(A, B, C, m, n, k);
}

inline void launch_modern_3(const __half* A, const __half* B, float* C,
                            int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v_modern<BM, BN, BK, WM, WN, /*STAGES=*/3><<<grid, block>>>(A, B, C, m, n, k);
}
