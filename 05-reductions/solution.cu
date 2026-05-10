// Module 5 — reference solution.
//
// Sum reductions:
//   v0 : classic shared-memory tree
//   v1 : warp shuffles, one element per thread
//   v2 : warp shuffles + grid-stride loop (production-style)
//   v0i: v0 rewritten using __reduce_add_sync (sm_80+ hardware redux on u32)
//
// Plus building blocks reused by Modules 9 and 10:
//   row_sum_kernel        — per-row sum of an MxN matrix
//   online_softmax_*      — single-pass (max, sum) recurrence at warp/block scope
//                           (lifted to common/online_softmax.cuh so M9 and M10
//                            can share the same primitives — see the header for
//                            full documentation)
//
// Each kernel writes one partial sum per block; the host adds them up.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "cuda_utils.h"
#include "online_softmax.cuh"

constexpr int BLK     = 256;
constexpr int N_WARPS = BLK / 32;
constexpr int V2_GRID = 256;        // a few resident blocks per SM

// ---------------------------------------------------------------------------
// Warp / block reduction helpers (re-used by all kernels and by M09/M10).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Five shuffle+add steps (offsets 16, 8, 4, 2, 1).
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    v = warp_reduce_sum(v);

    __shared__ float warp_sums[N_WARPS];
    if (lane == 0) warp_sums[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        v = (lane < N_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;       // valid in lane 0 of warp 0
}

// (Online-softmax primitives — `ms_pair`, `online_softmax_combine`,
// `online_softmax_warp`, `online_softmax_block`, plus the FA-style
// `online_softmax_combine_with_factors` — now live in
// common/online_softmax.cuh.  M9 and M10 share the same primitives.)

// ---------------------------------------------------------------------------
// Sum-reduction kernels.
// ---------------------------------------------------------------------------
__global__ void reduce_v0(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    __shared__ float smem[BLK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLK + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    for (int s = BLK / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}

__global__ void reduce_v1(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    int gid = blockIdx.x * BLK + threadIdx.x;
    float v = (gid < n) ? in[gid] : 0.0f;
    v = block_reduce_sum(v);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

__global__ void reduce_v2(const float* __restrict__ in,
                          float* __restrict__ out,
                          int n) {
    int gid    = blockIdx.x * BLK + threadIdx.x;
    int stride = BLK * gridDim.x;

    float v = 0.0f;
    for (int i = gid; i < n; i += stride) v += in[i];

    v = block_reduce_sum(v);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

// v0 reimplemented with the __reduce_add_sync hardware intrinsic (Ampere+).
// __reduce_*_sync is u32-only, so this kernel sums *unsigned int* inputs.
__global__ void reduce_v0_intrinsic(const unsigned int* __restrict__ in,
                                    unsigned int*       __restrict__ out,
                                    int n) {
    int gid = blockIdx.x * BLK + threadIdx.x;
    unsigned int v = (gid < n) ? in[gid] : 0u;

    // Warp-level reduction in a single SASS instruction (REDUX.SUM).
    unsigned int warp_v = __reduce_add_sync(0xffffffff, v);

    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ unsigned int warp_sums[N_WARPS];
    if (lane == 0) warp_sums[warp_id] = warp_v;
    __syncthreads();

    if (warp_id == 0) {
        unsigned int x = (lane < N_WARPS) ? warp_sums[lane] : 0u;
        unsigned int s = __reduce_add_sync(0xffffffff, x);
        if (lane == 0) out[blockIdx.x] = s;
    }
}

// ---------------------------------------------------------------------------
// Per-row sum (segmented reduction): one block per row, threads sweep columns.
// Building block for M9 (layernorm) and M10 (softmax).
// ---------------------------------------------------------------------------
__global__ void row_sum_kernel(const float* __restrict__ X,
                               float* __restrict__ out,
                               int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        v += X[row * cols + c];
    }
    v = block_reduce_sum(v);
    if (threadIdx.x == 0) out[row] = v;
}

// Per-row online softmax: returns (max_i, sum_i exp(X[i,j] - max_i)).
// This is what M9's softmax forward and M10's per-tile rescaling consume.
__global__ void row_online_softmax_kernel(const float* __restrict__ X,
                                          float* __restrict__ row_max,
                                          float* __restrict__ row_sum,
                                          int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ ms_pair warp_scratch[N_WARPS];

    // Sweep cols, folding into a per-thread (m, s) pair.
    ms_pair acc = online_softmax_identity();
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        acc = online_softmax_combine(acc, online_softmax_singleton(X[row * cols + c]));
    }

    // Reduce across the block. We piggy-back on the warp/block helper by feeding
    // the per-thread acc as a "singleton" — the warp helper already operates on
    // singletons, so we open-code the warp/block path:
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        ms_pair other;
        other.m = __shfl_down_sync(0xffffffff, acc.m, offset);
        other.s = __shfl_down_sync(0xffffffff, acc.s, offset);
        acc = online_softmax_combine(acc, other);
    }
    if (lane == 0) warp_scratch[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
        ms_pair q = (lane < N_WARPS) ? warp_scratch[lane] : online_softmax_identity();
        for (int offset = 16; offset > 0; offset >>= 1) {
            ms_pair other;
            other.m = __shfl_down_sync(0xffffffff, q.m, offset);
            other.s = __shfl_down_sync(0xffffffff, q.s, offset);
            q = online_softmax_combine(q, other);
        }
        if (lane == 0) {
            row_max[row] = q.m;
            row_sum[row] = q.s;
        }
    }
}

// ---------------------------------------------------------------------------
// Host driver: runs each kernel and verifies.
// ---------------------------------------------------------------------------
static float host_sum(const float* parts, int k) {
    float s = 0.0f;
    for (int i = 0; i < k; ++i) s += parts[i];
    return s;
}

int main() {
    constexpr int N = 1 << 24;          // 16M elements; sum of all-ones is exactly representable

    std::vector<float> h_in(N, 1.0f);
    const float expected = static_cast<float>(N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    int max_blocks = (N + BLK - 1) / BLK;
    CUDA_CHECK(cudaMalloc(&d_out, max_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&](const char* name, int grid, auto&& launch) {
        std::vector<float> h_out(grid);
        launch(grid);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float got = host_sum(h_out.data(), grid);
        std::printf("%-22s blocks=%-6d got=%.0f expected=%.0f %s\n",
                    name, grid, got, expected,
                    got == expected ? "(PASS)" : "(FAIL)");
    };

    int g0 = (N + BLK - 1) / BLK;
    run("reduce_v0",       g0,      [&](int g) { reduce_v0<<<g, BLK>>>(d_in, d_out, N); });
    run("reduce_v1",       g0,      [&](int g) { reduce_v1<<<g, BLK>>>(d_in, d_out, N); });
    run("reduce_v2",       V2_GRID, [&](int g) { reduce_v2<<<g, BLK>>>(d_in, d_out, N); });

    // __reduce_add_sync version: integer input.
    {
        std::vector<unsigned int> h_in_u(N, 1u);
        unsigned int *d_in_u, *d_out_u;
        CUDA_CHECK(cudaMalloc(&d_in_u, N * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_out_u, g0 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_in_u, h_in_u.data(), N * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        std::vector<unsigned int> h_out_u(g0);
        reduce_v0_intrinsic<<<g0, BLK>>>(d_in_u, d_out_u, N);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_out_u.data(), d_out_u, g0 * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        unsigned long long got_u = 0;
        for (int i = 0; i < g0; ++i) got_u += h_out_u[i];
        std::printf("%-22s blocks=%-6d got=%llu expected=%u %s\n",
                    "reduce_v0_intrinsic", g0,
                    got_u, (unsigned int)N,
                    got_u == (unsigned long long)N ? "(PASS)" : "(FAIL)");
        CUDA_CHECK(cudaFree(d_in_u));
        CUDA_CHECK(cudaFree(d_out_u));
    }

    // Per-row sum: 1024 rows of 4096 floats, all 1.0 -> each row sums to 4096.
    {
        constexpr int R = 1024, C = 4096;
        std::vector<float> hX(R * C, 1.0f);
        float *dX, *dOut;
        CUDA_CHECK(cudaMalloc(&dX, R * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, R * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dX, hX.data(), R * C * sizeof(float), cudaMemcpyHostToDevice));
        row_sum_kernel<<<R, BLK>>>(dX, dOut, R, C);
        CUDA_CHECK_LAST();
        std::vector<float> hOut(R);
        CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, R * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < R; ++i) {
            if (hOut[i] != static_cast<float>(C)) { ok = false; break; }
        }
        std::printf("%-22s rows=%-4d cols=%-4d row_sum[0]=%.0f expected=%d %s\n",
                    "row_sum_kernel", R, C, hOut[0], C, ok ? "(PASS)" : "(FAIL)");
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dOut));
    }

    // Per-row online softmax: 256 rows of 1024 floats, X[i,j] = (j == 0 ? 1.0 : 0.0).
    // Then row_max = 1, row_sum = exp(0) + 1023 * exp(-1) ≈ 1 + 376.45.
    {
        constexpr int R = 256, C = 1024;
        std::vector<float> hX(R * C, 0.0f);
        for (int i = 0; i < R; ++i) hX[i * C + 0] = 1.0f;

        float *dX, *dM, *dS;
        CUDA_CHECK(cudaMalloc(&dX, R * C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dM, R * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dS, R * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dX, hX.data(), R * C * sizeof(float), cudaMemcpyHostToDevice));
        row_online_softmax_kernel<<<R, BLK>>>(dX, dM, dS, R, C);
        CUDA_CHECK_LAST();
        std::vector<float> hM(R), hS(R);
        CUDA_CHECK(cudaMemcpy(hM.data(), dM, R * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hS.data(), dS, R * sizeof(float), cudaMemcpyDeviceToHost));
        // Expected: row_max = 1.0; row_sum = 1.0 + 1023 * exp(-1).
        float expected_s = 1.0f + 1023.0f * std::exp(-1.0f);
        bool ok = true;
        for (int i = 0; i < R; ++i) {
            if (std::fabs(hM[i] - 1.0f) > 1e-5f) { ok = false; break; }
            if (std::fabs(hS[i] - expected_s) > 1e-2f) { ok = false; break; }
        }
        std::printf("%-22s rows=%-4d cols=%-4d max[0]=%.3f sum[0]=%.3f expected_sum=%.3f %s\n",
                    "row_online_softmax", R, C, hM[0], hS[0], expected_s,
                    ok ? "(PASS)" : "(FAIL)");
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dM));
        CUDA_CHECK(cudaFree(dS));
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
