// Module 5 — bench the two single-block scan kernels at BLK = 256.
// (Don't expect production-scan numbers from this — we're not running across
// the whole device. The point is to compare Hillis-Steele to Blelloch with
// identical workloads.)

#include <cstdio>
#include <cmath>
#include <vector>

#include "bench.h"
#include "cuda_utils.h"

constexpr int BLK = 256;

__device__ __forceinline__ float warp_scan_inclusive(float v) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float t = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) v += t;
    }
    return v;
}

__global__ void scan_v0_hillis_steele(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    constexpr int N_WARPS = BLK / 32;

    float v = (gid < n) ? in[gid] : 0.0f;
    v = warp_scan_inclusive(v);

    __shared__ float warp_totals[N_WARPS];
    if (lane == 31) warp_totals[warp_id] = v;
    __syncthreads();

    if (warp_id == 0) {
        float t = (lane < N_WARPS) ? warp_totals[lane] : 0.0f;
        t = warp_scan_inclusive(t);
        if (lane < N_WARPS) warp_totals[lane] = t;
    }
    __syncthreads();

    if (warp_id > 0) v += warp_totals[warp_id - 1];
    if (gid < n) out[gid] = v;
}

__global__ void scan_v1_blelloch(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    __shared__ float s[BLK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float my = (gid < n) ? in[gid] : 0.0f;
    s[tid] = my;
    __syncthreads();

    for (int stride = 1; stride < BLK; stride <<= 1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < BLK) s[idx] += s[idx - stride];
        __syncthreads();
    }
    if (tid == 0) s[BLK - 1] = 0.0f;
    __syncthreads();

    for (int stride = BLK >> 1; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < BLK) {
            float t = s[idx - stride];
            s[idx - stride] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }
    float incl = s[tid] + my;
    if (gid < n) out[gid] = incl;
}

int main() {
    // Many independent 256-element scans launched as a grid: gives the GPU
    // enough work that we can compare per-block algorithm cost with low noise.
    constexpr int BLOCKS = 1 << 16;        // 65536 blocks -> 16M elements
    constexpr int N      = BLOCKS * BLK;
    constexpr int ITERS  = 50;
    const long bytes_in_out = 2L * N * sizeof(float);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("N = %d (%d blocks of %d), each block scans independently\n\n",
                N, BLOCKS, BLK);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));

    // Quick correctness sanity (single block) before timing.
    {
        std::vector<float> hin(BLK, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_in, hin.data(), BLK * sizeof(float), cudaMemcpyHostToDevice));
        scan_v0_hillis_steele<<<1, BLK>>>(d_in, d_out, BLK);
        CUDA_CHECK_LAST();
        std::vector<float> hout(BLK);
        CUDA_CHECK(cudaMemcpy(hout.data(), d_out, BLK * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = std::fabs(hout[BLK - 1] - static_cast<float>(BLK)) < 1e-3f;
        std::printf("verify scan_v0_hillis_steele: last=%.0f %s\n",
                    hout[BLK - 1], ok ? "(PASS)" : "(FAIL)");

        scan_v1_blelloch<<<1, BLK>>>(d_in, d_out, BLK);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(hout.data(), d_out, BLK * sizeof(float), cudaMemcpyDeviceToHost));
        ok = std::fabs(hout[BLK - 1] - static_cast<float>(BLK)) < 1e-3f;
        std::printf("verify scan_v1_blelloch:      last=%.0f %s\n\n",
                    hout[BLK - 1], ok ? "(PASS)" : "(FAIL)");
        CUDA_CHECK(cudaMemset(d_in, 0, N * sizeof(float)));
    }

    auto report = [&](const char* name, float ms) {
        float gbs = bytes_in_out / (ms * 1.0e6);
        std::printf("  %-22s %.3f ms   %.1f GB/s\n", name, ms, gbs);
    };

    float ms;
    ms = bench_min_ms(ITERS, [&] { scan_v0_hillis_steele<<<BLOCKS, BLK>>>(d_in, d_out, N); });
    report("scan_v0_hillis_steele", ms);
    ms = bench_min_ms(ITERS, [&] { scan_v1_blelloch<<<BLOCKS, BLK>>>(d_in, d_out, N); });
    report("scan_v1_blelloch", ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
