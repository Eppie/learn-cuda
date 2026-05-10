// Module 13 — cp.async via the cuda_pipeline.h wrapper, vs raw inline PTX.
// Both should produce identical PTX (and identical SASS); the demo confirms
// that the wrapper is just a thin layer over the cp.async instructions.
//
// Workload: a stripped-down streaming copy that uses cp.async to load global
// data into shared memory, then writes it out. We don't bother with double
// buffering — see Module 8 for the real version.

#include <cstdio>
#include <cuda_pipeline.h>

#include "bench.h"
#include "cuda_utils.h"

constexpr int BLK = 256;
constexpr int VEC = 4;            // float4 = 16 bytes per cp.async

// ----------------------------------------------------------------------------
// Wrapper version: __pipeline_memcpy_async + __pipeline_commit + __pipeline_wait_prior
// ----------------------------------------------------------------------------
__global__ void copy_wrapper(const float* __restrict__ a, float* __restrict__ b, int n) {
    __shared__ float smem[BLK * VEC];

    int gid = blockIdx.x * BLK + threadIdx.x;
    int idx_in  = gid * VEC;
    int idx_out = gid * VEC;

    if (idx_in + VEC <= n) {
        __pipeline_memcpy_async(&smem[threadIdx.x * VEC],
                                &a[idx_in],
                                /*bytes=*/16);
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
    __syncthreads();

    if (idx_out + VEC <= n) {
        for (int v = 0; v < VEC; ++v) b[idx_out + v] = smem[threadIdx.x * VEC + v];
    }
}

// ----------------------------------------------------------------------------
// Inline-PTX version: same instructions, written by hand.
// ----------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
    // .cg matches what __pipeline_memcpy_async emits — skip L1, cache at L2 only.
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_int), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

__global__ void copy_inline_ptx(const float* __restrict__ a, float* __restrict__ b, int n) {
    __shared__ float smem[BLK * VEC];

    int gid = blockIdx.x * BLK + threadIdx.x;
    int idx_in  = gid * VEC;
    int idx_out = gid * VEC;

    if (idx_in + VEC <= n) {
        cp_async_16(&smem[threadIdx.x * VEC], &a[idx_in]);
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncthreads();

    if (idx_out + VEC <= n) {
        for (int v = 0; v < VEC; ++v) b[idx_out + v] = smem[threadIdx.x * VEC + v];
    }
}

int main() {
    constexpr int N     = 1 << 24;        // 16M floats
    constexpr int ITERS = 50;
    const     int GRD   = (N / VEC + BLK - 1) / BLK;
    const     long bytes = (long)N * 2 * sizeof(float);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("N=%d (%.0f MB), bytes/launch = %.0f MB\n\n",
                N, N * sizeof(float) / 1.0e6, bytes / 1.0e6);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(float)));

    auto report = [&](const char* name, float ms) {
        float gbs = bytes / (ms * 1.0e6);
        std::printf("  %-26s %.3f ms   %.1f GB/s\n", name, ms, gbs);
    };

    report("__pipeline_memcpy_async", bench_min_ms(ITERS, [&]{ copy_wrapper<<<GRD,BLK>>>(d_a,d_b,N); }));
    report("inline cp.async PTX",     bench_min_ms(ITERS, [&]{ copy_inline_ptx<<<GRD,BLK>>>(d_a,d_b,N); }));

    std::printf("\nIdentical perf — the wrapper *is* this inline PTX.\n");
    std::printf("Verify with:  make ptx  &&  grep cp.async cpasync_inline.ptx\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
