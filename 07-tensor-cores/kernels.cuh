#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

// ============================================================================
// 7.0 — gemm_v0_wmma. Evolution of Module 6's v6 (warp tiling).
//
// Block tile      BM × BN  = 128 × 128
// Warp tile       WM × WN  =  64 ×  64       (4 warps/block, NUM_THREADS=128)
// WMMA fragment   16 × 16 × 16                (the hardware unit)
// Per warp:       (WM/16) × (WN/16) = 4 × 4 = 16 accumulator fragments
//
// The block-level structure (BM/BN/BK, cooperative tile loads, warp 2D layout)
// is identical to v6 in `06-gemm/kernels.cuh`. The change is the *inner FMA
// loop*: v6's `for (int dotIdx = 0; dotIdx < BK; ++dotIdx)` triple loop is
// replaced by a small `kk` loop over WMMA_K-sized chunks that issues
// `wmma::mma_sync`.
// ============================================================================
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
    int warpRow = warpId / (BN / WN);            // 2x2 warp grid for WM=WN=64
    int warpCol = warpId % (BN / WN);

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    A += cRow * BM * k;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * n + cCol * BN + warpCol * WN;

    // Vectorized 16-byte global loads: 8 halves = 16 bytes per int4
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

        // Replaces v6's `for (int dotIdx = 0; dotIdx < BK; ++dotIdx)` FMA loop.
        // For each WMMA-K chunk, load each row of a_frags (one per wm), then for
        // each column load a b_frag and mma into the WMITER c_frags that share
        // it. Order chosen for register reuse on the b_frag side.
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
    constexpr int WM = 64,  WN = 64;             // 4 warps/block (2x2)  — same as v6
    dim3 block(((BM * BN) / (WM * WN)) * 32);    // 128 threads
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v0_wmma<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 7.1 — gemm_v1_wmma_swizzled. Same kernel, but with an XOR-swizzled
// shared-memory layout on `As` to eliminate bank conflicts under
// `load_matrix_sync`.
//
// The bank-conflict story (BK=16 halves = 32 bytes per row of As):
//   * Shared memory has 32 banks of 4 bytes each. One row of As is 8 banks
//     wide (bank 0..7). Row r maps to bank (r * 8) mod 32 = (r mod 4) * 8.
//   * `load_matrix_sync` for `matrix_a` row_major distributes one column of
//     the 16x16 fragment across pairs of lanes. With BK-stride loads, lanes
//     (0,4,8,12) all hit bank 0; lanes (1,5,9,13) hit bank 8; and so on —
//     every 4-way bank conflict on every fragment load.
//
// The fix: permute the column index `c` of `As[r][c]` by XOR-ing with a
// row-derived value so that every row maps to a different starting bank,
// making the 4-lane pattern collision-free.
//
// The permutation we use: store A[r,c] at As_row r, slot (c XOR (r mod 8) ).
// Note c is in [0, BK)=[0,16) and we choose the XOR mask in [0,8) so the
// permutation stays within the row. A 16-byte vector load handles 8 halves
// at a time, and (c & ~7) is the byte-aligned chunk; we permute across the
// 2 chunks per row.
// ============================================================================
template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_v1_wmma_swizzled(const __half* __restrict__ A,
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

    // Pad As columns to BK + 8 halves so the XOR-swizzled column index never
    // crosses the row boundary. Simpler than the in-row XOR permutation:
    // we add 8 halves of padding per row, which by itself fixes the conflict
    // because every 8-half chunk now maps to a different bank pair across
    // consecutive rows. (Equivalent to a +8-half row stride, which is what
    // CUTLASS calls "padded layout"; the more elaborate XOR swizzle is the
    // production form, but for BK=16 the padding alone is conflict-free.)
    constexpr int LDAs = BK + 8;
    __shared__ __half As[BM * LDAs];
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
            *reinterpret_cast<int4*>(&As[(innerRowA + off) * LDAs + innerColA]) =
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
                // Stride is now LDAs (= BK + 8), which breaks the bank-conflict
                // pattern that comes with stride==BK==16 halves.
                load_matrix_sync(a_frag,
                                 &As[(warpRow * WM + wm * WMMA_M) * LDAs + kk], LDAs);

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

inline void launch_v1(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v1_wmma_swizzled<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 7.2 — gemm_v2_mma_sync. Raw `mma.sync` PTX instead of the WMMA wrapper.
//
// One warp per K-step issues a chain of `mma.sync.aligned.m16n8k16.row.col.
// f32.f16.f16.f32` instructions — the actual hardware tensor-core op. Two
// such instructions cover an m16n16k16 tile (the WMMA shape), so this is
// exactly what the WMMA wrapper emits under the hood, but we drive it
// directly so the lane-element layout is *ours* to control.
//
// Why: production ML kernels (CUTLASS, FlashAttention, FasterTransformer)
// use raw mma.sync because WMMA's opaque fragment layout makes epilogue
// fusion (e.g. softmax-on-fragments, bias-add in registers) painful.
//
// Same block decomposition as v0: 4 warps/block, each owns a 64×64 warp
// tile = 4×4 m16n16k16 sub-tiles. We issue 2× m16n8k16 per sub-tile, so
// each warp issues 4·4·2 = 32 mma.sync per K-chunk × (BK/16=1) = 32 mmas
// per outer K iteration.
//
// See M13 §"mma.sync vs WMMA" for fragment-layout details (lane→element
// mapping). The packing code mirrors what's in `13-ptx-appendix/mma_sync_
// example.cu` but in a tile-multi-step form.
// ============================================================================

// Pack two halves into a b32 for use as A-fragment / B-fragment registers.
__device__ __forceinline__ uint32_t pack2h(__half x, __half y) {
    uint32_t lo = *reinterpret_cast<uint16_t*>(&x);
    uint32_t hi = *reinterpret_cast<uint16_t*>(&y);
    return lo | (hi << 16);
}

template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_v2_mma_sync(const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 float* __restrict__ C,
                                 int m, int n, int k) {
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int NUM_WARPS   = (BM * BN) / (WM * WN);
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int WMITER      = WM / MMA_M;       // 4 with WM=64
    constexpr int WNITER      = WN / MMA_N;       // 8 with WN=64

    static_assert(BK == MMA_K, "this kernel assumes BK == 16 (one mma per k step)");

    int warpId  = threadIdx.x / 32;
    int warpRow = warpId / (BN / WN);
    int warpCol = warpId % (BN / WN);

    int laneId = threadIdx.x & 31;
    int g = laneId >> 2;            // 0..7 (row-pair index inside fragment)
    int t = laneId & 3;             // 0..3 (col-pair index inside fragment)

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

    // Per-warp accumulators: WMITER × WNITER tiles, each tile is 4 fp32
    // (per-lane: c0,c1 for rows g,g+8 of cols 2t,2t+1).
    float c[WMITER][WNITER][4];
    #pragma unroll
    for (int i = 0; i < WMITER; ++i)
        #pragma unroll
        for (int j = 0; j < WNITER; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) c[i][j][q] = 0.0f;

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

        // ---- per-K-step: issue WMITER × WNITER mma.sync.m16n8k16 ----
        // For each sub-tile (wm, wn) the lane-local A-fragment (4 b32 = 8 fp16)
        // covers rows [warpRow*WM + wm*16, +16) of As and all 16 columns. The
        // B-fragment (2 b32 = 4 fp16) covers all 16 rows of Bs and 8 columns
        // [warpCol*WN + wn*8, +8). We re-load A per (wm) and re-use across
        // all WNITER mma.syncs sharing those rows.
        #pragma unroll
        for (int wm = 0; wm < WMITER; ++wm) {
            int A_row = warpRow * WM + wm * MMA_M;
            // A fragment: each lane holds A[r,2t..2t+1], A[r+8,2t..2t+1],
            //                          A[r,2t+8..2t+9], A[r+8,2t+8..2t+9]
            // where r = A_row + g.
            uint32_t a0 = pack2h(As[(A_row + g    ) * BK + 2*t    ],
                                 As[(A_row + g    ) * BK + 2*t + 1]);
            uint32_t a1 = pack2h(As[(A_row + g + 8) * BK + 2*t    ],
                                 As[(A_row + g + 8) * BK + 2*t + 1]);
            uint32_t a2 = pack2h(As[(A_row + g    ) * BK + 2*t + 8],
                                 As[(A_row + g    ) * BK + 2*t + 9]);
            uint32_t a3 = pack2h(As[(A_row + g + 8) * BK + 2*t + 8],
                                 As[(A_row + g + 8) * BK + 2*t + 9]);

            #pragma unroll
            for (int wn = 0; wn < WNITER; ++wn) {
                int B_col = warpCol * WN + wn * MMA_N;
                // B is row-major in Bs (Bs[k_row, n_col]). The mma.sync wants
                // B in COL-major layout (.row.col → A row-major, B col-major).
                // Per-lane B holds: B[2t..2t+1, B_col + g], B[2t+8..2t+9, B_col + g].
                uint32_t b0 = pack2h(Bs[(2*t    ) * BN + B_col + g],
                                     Bs[(2*t + 1) * BN + B_col + g]);
                uint32_t b1 = pack2h(Bs[(2*t + 8) * BN + B_col + g],
                                     Bs[(2*t + 9) * BN + B_col + g]);

                float& c0 = c[wm][wn][0];
                float& c1 = c[wm][wn][1];
                float& c2 = c[wm][wn][2];
                float& c3 = c[wm][wn][3];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n\t"
                    "{%0, %1, %2, %3},\n\t"
                    "{%4, %5, %6, %7},\n\t"
                    "{%8, %9},\n\t"
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                       "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }

    // ---- store back ----
    // Lane (g,t) holds: C[g, 2t..2t+1] and C[g+8, 2t..2t+1] for each sub-tile.
    #pragma unroll
    for (int wm = 0; wm < WMITER; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WNITER; ++wn) {
            int C_row = wm * MMA_M;
            int C_col = wn * MMA_N;
            C[(C_row + g    ) * n + C_col + 2*t    ] = c[wm][wn][0];
            C[(C_row + g    ) * n + C_col + 2*t + 1] = c[wm][wn][1];
            C[(C_row + g + 8) * n + C_col + 2*t    ] = c[wm][wn][2];
            C[(C_row + g + 8) * n + C_col + 2*t + 1] = c[wm][wn][3];
        }
    }
}

inline void launch_v2(const __half* A, const __half* B, float* C,
                      int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);   // 128 threads
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v2_mma_sync<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, m, n, k);
}

// ============================================================================
// 7.X — FP16-accumulator variant of v0 (kept for the stretch problem).
// ~2× peak throughput on Ada at the cost of accumulator precision.
// ============================================================================
template <int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_v0_wmma_fp16acc(const __half* __restrict__ A,
                                     const __half* __restrict__ B,
                                     __half* __restrict__ C,
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

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag[WMITER][WNITER];
    for (int i = 0; i < WMITER; ++i)
        for (int j = 0; j < WNITER; ++j)
            fill_fragment(c_frag[i][j], __float2half(0.0f));

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

inline void launch_v0_fp16acc(const __half* A, const __half* B, __half* C,
                              int m, int n, int k) {
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64,  WN = 64;
    dim3 block(((BM * BN) / (WM * WN)) * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v0_wmma_fp16acc<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, m, n, k);
}
