#pragma once

// All eight GEMM kernels and their launchers. C = A · B for row-major
// A (M × K), B (K × N), C (M × N).
//
// Block-index convention used uniformly: blockIdx.x -> column of C,
//                                        blockIdx.y -> row    of C.
// (See README §1.)

// ============================================================================
// 6.0 — Naive (intentionally bad: lanes within a warp walk *rows* of C)
// ============================================================================
__global__ void gemm_v0_naive(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int m, int n, int k) {
    // Deliberately wrong mapping: threadIdx.x walks rows, threadIdx.y walks cols.
    // Within a warp, threadIdx.x varies fastest -> 32 different rows of A and C
    // are touched -> reads are uncoalesced. Module 6.1 fixes this.
    int row = blockIdx.y * 32 + threadIdx.x;   // row, varies within warp
    int col = blockIdx.x * 32 + threadIdx.y;   // col, constant within warp
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) acc += A[row * k + kk] * B[kk * n + col];
        C[row * n + col] = acc;
    }
}

inline void launch_v0(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    dim3 block(32, 32);
    // blockIdx.x -> column tile, blockIdx.y -> row tile.
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    gemm_v0_naive<<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 6.1 — Coalesced (threadIdx.x decomposed so lanes walk *columns* of C)
// ============================================================================
__global__ void gemm_v1_coalesced(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int m, int n, int k) {
    constexpr int BS = 32;
    int row = blockIdx.y * BS + (threadIdx.x / BS);  // row, slow within warp
    int col = blockIdx.x * BS + (threadIdx.x % BS);  // col, fast within warp
    if (row < m && col < n) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) acc += A[row * k + kk] * B[kk * n + col];
        C[row * n + col] = acc;
    }
}

inline void launch_v1(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    dim3 block(32 * 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    gemm_v1_coalesced<<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 6.2 — Shared-memory blocking (each block computes a 32×32 tile of C)
// ============================================================================
template <int BS>
__global__ void gemm_v2_shared(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int m, int n, int k) {
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;

    __shared__ float As[BS * BS];
    __shared__ float Bs[BS * BS];

    int threadCol = threadIdx.x % BS;
    int threadRow = threadIdx.x / BS;

    A += cRow * BS * k;
    B += cCol * BS;
    C += cRow * BS * n + cCol * BS;

    float acc = 0.0f;
    for (int kBlock = 0; kBlock < k; kBlock += BS) {
        As[threadRow * BS + threadCol] = A[threadRow * k + threadCol];
        Bs[threadRow * BS + threadCol] = B[threadRow * n + threadCol];
        __syncthreads();

        A += BS;
        B += BS * n;

        for (int kk = 0; kk < BS; ++kk) {
            acc += As[threadRow * BS + kk] * Bs[kk * BS + threadCol];
        }
        __syncthreads();
    }

    C[threadRow * n + threadCol] = acc;
}

inline void launch_v2(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    constexpr int BS = 32;
    dim3 block(BS * BS);
    dim3 grid((n + BS - 1) / BS, (m + BS - 1) / BS);   // x: cols, y: rows
    gemm_v2_shared<BS><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 6.3 — 1D thread tiling (each thread computes TM elements of C in a column)
// ============================================================================
template <int BM, int BN, int BK, int TM>
__global__ void gemm_v3_1d_tiling(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int m, int n, int k) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int threadCol = threadIdx.x % BN;
    int threadRow = threadIdx.x / BN;   // 0 .. BM/TM - 1

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0f};

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * k + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * n + innerColB];
        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (int t = 0; t < TM; ++t) {
                threadResults[t] += As[(threadRow * TM + t) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }

    for (int t = 0; t < TM; ++t) {
        C[(threadRow * TM + t) * n + threadCol] = threadResults[t];
    }
}

inline void launch_v3(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);  // 512
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v3_1d_tiling<BM, BN, BK, TM><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 6.4 — 2D thread tiling (each thread computes a TM×TN tile)
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v4_2d_tiling(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int m, int n, int k) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    constexpr int numThreads = (BM * BN) / (TM * TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    constexpr int strideA = numThreads / BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;
    constexpr int strideB = numThreads / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        for (int off = 0; off < BM; off += strideA) {
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * k + innerColA];
        }
        for (int off = 0; off < BK; off += strideB) {
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * n + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            for (int i = 0; i < TN; ++i) regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm * TN + rn] += regM[rm] * regN[rn];
        }
        __syncthreads();
    }

    for (int rm = 0; rm < TM; ++rm) {
        for (int rn = 0; rn < TN; ++rn) {
            C[(threadRow * TM + rm) * n + threadCol * TN + rn] =
                threadResults[rm * TN + rn];
        }
    }
}

inline void launch_v4(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));   // 256
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v4_2d_tiling<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 6.5a — Vectorized loads (As still row-major in shared memory)
// One idea per step: introduce float4 LDG/STG instructions but keep the v4
// shared-memory layout for As (As[BM][BK], indexed As[m][k]). The inner-loop
// load `regM[i] = As[(threadRow*TM + i)*BK + dotIdx]` is strided-by-BK; we
// fix that in 6.5b by transposing As.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v5a_vectorized(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int m, int n, int k) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    constexpr int numThreads = (BM * BN) / (TM * TN);

    __shared__ float As[BM * BK];     // row-major: As[m][k]
    __shared__ float Bs[BK * BN];

    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideA = numThreads / (BK / 4);
    constexpr int strideB = numThreads / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        // float4 loads of A and B into shared memory.
        for (int off = 0; off < BM; off += strideA) {
            reinterpret_cast<float4*>(As + (innerRowA + off) * BK)[innerColA] =
                reinterpret_cast<const float4*>(A + (innerRowA + off) * k)[innerColA];
        }
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4*>(Bs + (innerRowB + off) * BN)[innerColB] =
                reinterpret_cast<const float4*>(B + (innerRowB + off) * n)[innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Strided load of As: stride BK -> compiler can't vectorize.
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            for (int i = 0; i < TN; ++i) regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm * TN + rn] += regM[rm] * regN[rn];
        }
        __syncthreads();
    }

    for (int rm = 0; rm < TM; ++rm) {
        for (int rn = 0; rn < TN; rn += 4) {
            float4 t;
            t.x = threadResults[rm * TN + rn + 0];
            t.y = threadResults[rm * TN + rn + 1];
            t.z = threadResults[rm * TN + rn + 2];
            t.w = threadResults[rm * TN + rn + 3];
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + rm) * n + threadCol * TN + rn])[0] = t;
        }
    }
}

inline void launch_v5a(const float* A, const float* B, float* C,
                       int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));   // 256
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v5a_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 6.5b — Vectorized loads + transposed As in shared memory
// Step two: store As transposed (As[BK][BM]). The inner-loop load
//   regM[i] = As[dotIdx * BM + threadRow * TM + i]
// is now contiguous across `i`, so the compiler emits LDS.128.
// ============================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v5b_vectorized_transposed(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int m, int n, int k) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    constexpr int numThreads = (BM * BN) / (TM * TN);

    __shared__ float As[BK * BM];     // transposed: As[k][m]
    __shared__ float Bs[BK * BN];

    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * k;
    B += cCol * BN;
    C += cRow * BM * n + cCol * BN;

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideA = numThreads / (BK / 4);
    constexpr int strideB = numThreads / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        for (int off = 0; off < BM; off += strideA) {
            float4 t = reinterpret_cast<const float4*>(A + (innerRowA + off) * k)[innerColA];
            // Transpose during the store: row-major source -> col-major dest.
            As[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
        }
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4*>(Bs + (innerRowB + off) * BN)[innerColB] =
                reinterpret_cast<const float4*>(B + (innerRowB + off) * n)[innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * n;

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i) regM[i] = As[dotIdx * BM + threadRow * TM + i];
            for (int i = 0; i < TN; ++i) regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm * TN + rn] += regM[rm] * regN[rn];
        }
        __syncthreads();
    }

    for (int rm = 0; rm < TM; ++rm) {
        for (int rn = 0; rn < TN; rn += 4) {
            float4 t;
            t.x = threadResults[rm * TN + rn + 0];
            t.y = threadResults[rm * TN + rn + 1];
            t.z = threadResults[rm * TN + rn + 2];
            t.w = threadResults[rm * TN + rn + 3];
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + rm) * n + threadCol * TN + rn])[0] = t;
        }
    }
}

inline void launch_v5b(const float* A, const float* B, float* C,
                       int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));   // 256
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v5b_vectorized_transposed<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, m, n, k);
}

// Backwards-compat alias: code (and the README) still refers to "v5".
// Maps to v5b (the final form of v5).
inline void launch_v5(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    launch_v5b(A, B, C, m, n, k);
}

// ============================================================================
// 6.6 — Full warp tiling with WSUBM × WSUBN sub-tiles (Boehm's kernel 10).
//
// Hierarchy (block tile -> warp tile -> sub-tile -> thread tile):
//
//   block tile      BM × BN
//     warp tile     WM × WN              (NUM_WARPS = (BM*BN) / (WM*WN))
//       sub-tile    WSUBM × WSUBN        iterated WMITER × WNITER per warp
//         thread tile TM × TN            (each lane owns one TM×TN per sub-tile)
//
// Each warp now iterates over WMITER * WNITER sub-tiles per outer K-step,
// reusing regM/regN registers across sub-tiles. This is what gets us closer
// to cuBLAS — without it, v6 is just a renaming of v5.
//
// Suggested config (and the one used by launch_v6 below):
//   BM=128, BN=128, BK=16
//   WM=64,  WN=64                4 warps/block, NUM_THREADS=128
//   WMITER=2, WNITER=2           WSUBM=32, WSUBN=32 (4 sub-tiles per warp)
//   TM=4, TN=8                   per-thread tile
//   per thread total: TM*TN * WMITER * WNITER = 4*8 * 4 = 128 elements
// ============================================================================
template <int BM, int BN, int BK,
          int WM, int WN, int WMITER, int WNITER,
          int TM, int TN>
__global__ void gemm_v6_warptiling(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int m, int n, int k) {
    constexpr int NUM_WARPS   = (BM * BN) / (WM * WN);
    constexpr int NUM_THREADS = NUM_WARPS * 32;

    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // 32 lanes within a warp tile WSUBM x WSUBN sub-tile each owning TM x TN.
    constexpr int LANE_COLS = WSUBN / TN;     // lanes laid out across N
    constexpr int LANE_ROWS = WSUBM / TM;     // lanes laid out across M
    static_assert(LANE_ROWS * LANE_COLS == 32,
                  "WSUBM/TM * WSUBN/TN must equal 32 (one lane per (m,n) cell)");

    // ---- block / warp / lane indexing ----
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int warpIdx = threadIdx.x / 32;
    int warpRow = warpIdx / (BN / WN);
    int warpCol = warpIdx % (BN / WN);

    int laneIdx         = threadIdx.x % 32;
    int laneColInWarp   = laneIdx % LANE_COLS;     // 0 .. WSUBN/TN - 1
    int laneRowInWarp   = laneIdx / LANE_COLS;     // 0 .. WSUBM/TM - 1

    __shared__ float As[BK * BM];     // transposed: As[k][m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    // C points at the warp's corner. Thread/sub-tile offsets applied at write.
    C += (cRow * BM + warpRow * WM) * n + cCol * BN + warpCol * WN;

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    constexpr int strideA = NUM_THREADS / (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    constexpr int strideB = NUM_THREADS / (BN / 4);

    // Per-thread accumulators: WMITER * TM × WNITER * TN floats, all in registers.
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    // Register caches for the inner FMA: TM rows (across all WMITER sub-tiles)
    // and TN cols (across all WNITER sub-tiles).
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (int kBlock = 0; kBlock < k; kBlock += BK) {
        // -------- load A (transposed) and B into shared memory --------
        for (int off = 0; off < BM; off += strideA) {
            float4 t = reinterpret_cast<const float4*>(A + (innerRowA + off) * k)[innerColA];
            As[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
        }
        for (int off = 0; off < BK; off += strideB) {
            reinterpret_cast<float4*>(Bs + (innerRowB + off) * BN)[innerColB] =
                reinterpret_cast<const float4*>(B + (innerRowB + off) * n)[innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * n;

        // -------- inner FMA, iterating over sub-tiles --------
        // For each k step, load regM/regN slices for *all* WMITER × WNITER
        // sub-tiles, then issue all FMAs. The compiler is now free to schedule
        // across sub-tiles for ILP.
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load regM: WMITER sub-tile rows × TM lane rows.
            for (int wm = 0; wm < WMITER; ++wm) {
                for (int t = 0; t < TM; ++t) {
                    regM[wm * TM + t] = As[dotIdx * BM
                                          + warpRow * WM
                                          + wm * WSUBM
                                          + laneRowInWarp * TM + t];
                }
            }
            // Load regN: WNITER sub-tile cols × TN lane cols.
            for (int wn = 0; wn < WNITER; ++wn) {
                for (int t = 0; t < TN; ++t) {
                    regN[wn * TN + t] = Bs[dotIdx * BN
                                          + warpCol * WN
                                          + wn * WSUBN
                                          + laneColInWarp * TN + t];
                }
            }

            // FMA: outer over (wm, wn), inner over (rm, rn).
            for (int wm = 0; wm < WMITER; ++wm) {
                for (int wn = 0; wn < WNITER; ++wn) {
                    for (int rm = 0; rm < TM; ++rm) {
                        for (int rn = 0; rn < TN; ++rn) {
                            int idx = ((wm * TM + rm) * WNITER + wn) * TN + rn;
                            threadResults[idx] += regM[wm * TM + rm]
                                                * regN[wn * TN + rn];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // -------- write back C (vectorized stores per row) --------
    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            // C-corner of this sub-tile (relative to the warp's C corner).
            float* Csub = C + (wm * WSUBM + laneRowInWarp * TM) * n
                            + (wn * WSUBN + laneColInWarp * TN);
            for (int rm = 0; rm < TM; ++rm) {
                for (int rn = 0; rn < TN; rn += 4) {
                    float4 t;
                    int idx = ((wm * TM + rm) * WNITER + wn) * TN + rn;
                    t.x = threadResults[idx + 0];
                    t.y = threadResults[idx + 1];
                    t.z = threadResults[idx + 2];
                    t.w = threadResults[idx + 3];
                    reinterpret_cast<float4*>(&Csub[rm * n + rn])[0] = t;
                }
            }
        }
    }
}

inline void launch_v6(const float* A, const float* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;             // 4 warps/block (2x2)
    constexpr int WMITER = 2, WNITER = 2;        // 4 sub-tiles per warp (2x2)
    constexpr int TM = 4,   TN = 8;              // (WSUBM/TM)*(WSUBN/TN) = 8*4 = 32
    constexpr int NUM_THREADS = ((BM * BN) / (WM * WN)) * 32;   // 128
    dim3 block(NUM_THREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v6_warptiling<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<grid, block>>>(A, B, C, m, n, k);
}
