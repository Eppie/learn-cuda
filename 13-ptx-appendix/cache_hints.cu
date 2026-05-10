// Module 13 — inline PTX cache modifiers.
//
// Three variants of a streaming copy y[i] = x[i] (each element used once):
//   1. Default global load:  ld.global.f32      (cache at L1 + L2; "cache all")
//   2. .cg (skip L1):        ld.global.cg.f32   (cache global only — L2)
//   3. .nc (read-only):      ld.global.nc.f32   (read-only cache path; can dedupe)
//
// CRITICAL pedagogical note: if you mark `a` as `const float* __restrict__`
// and dereference it normally, the compiler will silently emit
// `ld.global.nc.f32` — same as the .nc variant! That makes the comparison
// pointless. So `copy_default` deliberately takes a NON-restricted pointer
// (and uses inline asm to force `ld.global.f32`) to land squarely on the
// "default cached" path instead of the read-only path.
//
// On Ada with a 16 MB L2 and unified L1/shared, the perf delta on a streaming
// workload is small but measurable; on smaller GPUs or in mixed-pattern kernels
// it's bigger. Use this as a template for finer-grained access-pattern control.

#include <cstdio>

#include "bench.h"
#include "cuda_utils.h"

// NOTE: NO __restrict__ on the input pointer. With __restrict__ the compiler
// would lift this to ld.global.nc.f32 and the three kernels below would
// produce only two distinct PTX instructions instead of three.
__device__ __forceinline__ float ld_default(const float* p) {
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float ld_cg(const float* __restrict__ p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float ld_nc(const float* __restrict__ p) {
    float v;
    asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

// Note: `a` is intentionally NOT __restrict__ here so the compiler doesn't
// promote ld.global.f32 to ld.global.nc.f32.
__global__ void copy_default(const float* a, float* __restrict__ b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) b[gid] = ld_default(&a[gid]);
}

__global__ void copy_cg(const float* __restrict__ a, float* __restrict__ b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) b[gid] = ld_cg(&a[gid]);
}

__global__ void copy_nc(const float* __restrict__ a, float* __restrict__ b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) b[gid] = ld_nc(&a[gid]);
}

int main() {
    constexpr int N     = 1 << 26;        // 256 MB — well past L2
    constexpr int BLK   = 256;
    constexpr int ITERS = 50;
    const     int GRD   = (N + BLK - 1) / BLK;
    const     long bytes = (long)N * 2 * sizeof(float);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("N=%d (%.0f MB), bytes/launch (read+write) = %.0f MB\n\n",
                N, N * sizeof(float) / 1.0e6, bytes / 1.0e6);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(float)));

    auto report = [&](const char* name, float ms) {
        float gbs = bytes / (ms * 1.0e6);
        std::printf("  %-22s %.3f ms   %.1f GB/s\n", name, ms, gbs);
    };

    report("ld.global (default)", bench_min_ms(ITERS, [&]{ copy_default<<<GRD,BLK>>>(d_a,d_b,N); }));
    report("ld.global.cg",        bench_min_ms(ITERS, [&]{ copy_cg     <<<GRD,BLK>>>(d_a,d_b,N); }));
    report("ld.global.nc",        bench_min_ms(ITERS, [&]{ copy_nc     <<<GRD,BLK>>>(d_a,d_b,N); }));

    std::printf("\nInspect:  make ptx  &&  grep ld.global cache_hints.ptx\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
