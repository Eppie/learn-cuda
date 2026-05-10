// Module 5 — scan starter scaffold. Solve the TODOs.
//
// Two scans:
//   warp_scan_inclusive    — Hillis-Steele inside one warp (5 shuffle+adds).
//   scan_v0_hillis_steele  — block scan = warp scan + propagate per-warp totals.
//   scan_v1_blelloch       — up-sweep / down-sweep work-efficient scan over BLK.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "cuda_utils.h"

constexpr int BLK = 256;

// TODO 7: in-warp inclusive scan via __shfl_up_sync.
// Iterate offsets 1, 2, 4, 8, 16; if `lane >= offset`, add the upstream neighbor.
__device__ __forceinline__ float warp_scan_inclusive(float v) {
    // your code here
    return v;
}

// TODO 7 (cont.): block-level Hillis-Steele scan.
// 1. Each warp does warp_scan_inclusive.
// 2. Lane 31 of each warp writes the warp's total to shared mem.
// 3. Warp 0 scans those totals.
// 4. Each thread (in warp w > 0) adds the prefix from warps before it.
__global__ void scan_v0_hillis_steele(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int n) {
    // your code here
}

// TODO 8: Blelloch up-down scan over BLK elements in shared memory.
// Up-sweep: for stride in {1, 2, 4, ..., BLK/2}: s[idx] += s[idx - stride] where
//   idx = (tid + 1) * 2*stride - 1.
// Set s[BLK-1] = 0 (replaces total with identity).
// Down-sweep: for stride in {BLK/2, ..., 1}: swap-and-add (left = parent, right += old left).
// Convert exclusive scan back to inclusive by adding the original input.
__global__ void scan_v1_blelloch(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int n) {
    // your code here
}

int main() {
    constexpr int N = 256;
    std::vector<float> h_in(N, 1.0f), h_out(N), h_ref(N);
    h_ref[0] = h_in[0];
    for (int i = 1; i < N; ++i) h_ref[i] = h_ref[i - 1] + h_in[i];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto check = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(h_out[i] - h_ref[i]) > 1e-3f) { ok = false; break; }
        }
        std::printf("%-24s last=%.0f expected=%.0f %s\n",
                    name, h_out[N - 1], h_ref[N - 1], ok ? "(PASS)" : "(FAIL)");
    };

    scan_v0_hillis_steele<<<1, BLK>>>(d_in, d_out, N);
    CUDA_CHECK_LAST();
    check("scan_v0_hillis_steele");

    scan_v1_blelloch<<<1, BLK>>>(d_in, d_out, N);
    CUDA_CHECK_LAST();
    check("scan_v1_blelloch");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
