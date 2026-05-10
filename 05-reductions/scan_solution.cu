// Module 5 — scan reference solution.
//
// Inclusive scans (prefix sums) via two algorithms:
//   warp_scan_inclusive   — Hillis-Steele, O(N log N) work, depth log N.
//   block_scan_blelloch   — Blelloch up-down, O(N) work, depth 2 log N, BLK = 256.
//
// Both produce the same output for blocks of <= BLK elements; the difference is
// whether work is wasted (Hillis-Steele) or whether some scratch space + extra
// barriers are spent (Blelloch).
//
// Run: prints PASS/FAIL for both kernels against a CPU reference.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "cuda_utils.h"

constexpr int BLK = 256;

// ---------------------------------------------------------------------------
// Hillis-Steele (work-inefficient): in-warp inclusive scan via __shfl_up_sync.
// Five shuffle+add steps with offsets 1, 2, 4, 8, 16. Each lane reads upstream
// neighbor; if upstream is within the warp boundary, add it.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_scan_inclusive(float v) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float t = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) v += t;
    }
    return v;
}

// Block-level Hillis-Steele: warp-scan, then propagate per-warp totals.
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

    // Scan the per-warp totals using one warp.
    if (warp_id == 0) {
        float t = (lane < N_WARPS) ? warp_totals[lane] : 0.0f;
        t = warp_scan_inclusive(t);
        if (lane < N_WARPS) warp_totals[lane] = t;
    }
    __syncthreads();

    // Add the prefix from earlier warps.
    if (warp_id > 0) v += warp_totals[warp_id - 1];

    if (gid < n) out[gid] = v;
}

// ---------------------------------------------------------------------------
// Blelloch (work-efficient) up-down scan over BLK elements in shared memory.
// Up-sweep: standard reduction; node `i + 2s - 1` accumulates `i + s - 1`.
// Down-sweep: starting from the (saved) total then identity-replaced root,
// each node distributes its value to the left child and adds to the right child.
// Final pass converts exclusive scan to inclusive by adding the original input.
// ---------------------------------------------------------------------------
__global__ void scan_v1_blelloch(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    __shared__ float s[BLK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float my = (gid < n) ? in[gid] : 0.0f;
    s[tid] = my;
    __syncthreads();

    // Up-sweep: build the reduction tree in place.
    // After step `stride`, s[tid] holds the sum of the `2*stride` elements ending at tid.
    for (int stride = 1; stride < BLK; stride <<= 1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < BLK) s[idx] += s[idx - stride];
        __syncthreads();
    }

    // Replace root with identity so down-sweep produces exclusive scan.
    if (tid == 0) s[BLK - 1] = 0.0f;
    __syncthreads();

    // Down-sweep.
    for (int stride = BLK >> 1; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < BLK) {
            float t = s[idx - stride];
            s[idx - stride] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }

    // s now holds an exclusive scan; convert to inclusive by adding the original input.
    float incl = s[tid] + my;
    if (gid < n) out[gid] = incl;
}

// ---------------------------------------------------------------------------
// Host driver.
// ---------------------------------------------------------------------------
int main() {
    constexpr int N = 256;   // single-block to verify both algorithms cleanly

    std::vector<float> h_in(N), h_out(N), h_ref(N);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    h_ref[0] = h_in[0];
    for (int i = 1; i < N; ++i) h_ref[i] = h_ref[i - 1] + h_in[i];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto check = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        int  bad = -1;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_out[i] - h_ref[i]) > 1e-3f) { ok = false; bad = i; break; }
        }
        if (ok) std::printf("%-24s last=%.0f expected=%.0f (PASS)\n",
                            name, h_out[N - 1], h_ref[N - 1]);
        else    std::printf("%-24s mismatch at %d: got=%.3f expected=%.3f (FAIL)\n",
                            name, bad, h_out[bad], h_ref[bad]);
    };

    scan_v0_hillis_steele<<<1, BLK>>>(d_in, d_out, N);
    CUDA_CHECK_LAST();
    check("scan_v0_hillis_steele");

    scan_v1_blelloch<<<1, BLK>>>(d_in, d_out, N);
    CUDA_CHECK_LAST();
    check("scan_v1_blelloch");

    // Mixed input pattern.
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 7) - 3.0f;
    h_ref[0] = h_in[0];
    for (int i = 1; i < N; ++i) h_ref[i] = h_ref[i - 1] + h_in[i];
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    scan_v0_hillis_steele<<<1, BLK>>>(d_in, d_out, N);
    CUDA_CHECK_LAST();
    check("scan_v0_hillis_steele (mix)");

    scan_v1_blelloch<<<1, BLK>>>(d_in, d_out, N);
    CUDA_CHECK_LAST();
    check("scan_v1_blelloch (mix)");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
