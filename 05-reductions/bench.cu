// Module 5 — benchmark v0/v1/v2 reductions. Reads N floats; one partial sum per
// block in the output. Bandwidth metric uses the input bytes only (the partial-sum
// output is tiny by comparison).

#include <cstdio>

#include "bench.h"
#include "cuda_utils.h"

constexpr int BLK     = 256;
constexpr int N_WARPS = BLK / 32;
constexpr int V2_GRID = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
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
    return v;
}

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

int main() {
    constexpr int N     = 1 << 26;          // 64M floats = 256 MB
    constexpr int ITERS = 50;
    const     long bytes = static_cast<long>(N) * sizeof(float); // input read

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("N = %d, input = %.0f MB\n\n", N, bytes / 1.0e6);

    int g0 = (N + BLK - 1) / BLK;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, g0 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    auto report = [&](const char* name, float ms) {
        float gbs = bytes / (ms * 1.0e6);
        std::printf("  %-12s %.3f ms   %.1f GB/s   (%.0f%% of peak)\n",
                    name, ms, gbs, gbs / 1008.0f * 100.0f);
    };

    float ms;
    ms = bench_min_ms(ITERS, [&] { reduce_v0<<<g0, BLK>>>(d_in, d_out, N); });
    report("reduce_v0", ms);
    ms = bench_min_ms(ITERS, [&] { reduce_v1<<<g0, BLK>>>(d_in, d_out, N); });
    report("reduce_v1", ms);
    ms = bench_min_ms(ITERS, [&] { reduce_v2<<<V2_GRID, BLK>>>(d_in, d_out, N); });
    report("reduce_v2", ms);

    std::printf("\nRTX 4090 peak DRAM bandwidth: ~1008 GB/s.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
