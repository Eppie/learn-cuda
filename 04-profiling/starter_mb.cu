// Module 4 — Exercise B: multi-bottleneck diagnosis.
//
// This kernel computes a per-row L2-normalized scaling:
//   for each row r in [0, R):
//     s = sum_{c=0..C-1} x[r * C + c]^2
//     for each c in [0, C):
//       y[r * C + c] = x[r * C + c] / sqrt(s + 1e-6)
//
// It is *correct* (host verify will print PASS), but it is slower than it should
// be for THREE different reasons, all visible in `ncu --set full ./starter_mb`:
//
//   (1) STRIDED ACCESS. The grid maps blockIdx.x → row, threadIdx.x → column. That's
//       fine for the inner loop, but the kernel processes ONE row per block — so
//       the column traversal inside the block is correct, but you can also see this
//       with a different access pattern. Look at sectors-per-request.
//
//   (2) REGISTER PRESSURE. The kernel keeps every column it reads in a stack-array
//       so the second pass can divide by the norm. With C = 512 floats per row, the
//       compiler may spill heavily — check `launch__registers_per_thread` and Long
//       Scoreboard stall percentages.
//
//   (3) BAD BLOCK SIZE. The block is launched as 64 threads (= 2 warps), so each
//       SM can only resident a handful of warps even though there's plenty of
//       shared memory and registers. Check Achieved Occupancy.
//
// Three plausible fixes; ONLY ONE is the actual top limiter on this kernel. Which
// one? Profile, identify, fix that, re-profile.
//
// (After fixing the top limiter, the SECOND limiter often becomes visible — that's
// fine; it's how real optimization goes. Iterate until ncu reports the kernel is
// near a sane efficiency for the workload.)
//
// Reference fix is in solution_mb.cu — don't peek.

#include <cstdio>
#include <cmath>
#include <vector>

#include "cuda_utils.h"

constexpr int R = 4096;        // rows
constexpr int C = 512;         // columns

// Block size that caps occupancy.
constexpr int BLK = 64;        // 2 warps per block

__global__ void row_l2_normalize_slow(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // (2) Local stack array sized to cols → register-spilled.
    // (cols is a kernel argument, so the compiler can't see it's 512 statically;
    //  it allocates max-sized local memory backed by L1.)
    float local_buf[/*expected cols=*/ 512];

    // First pass: load + accumulate sum-of-squares.
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        local_buf[c] = v;
        s += v * v;
    }

    // Block-wide reduction of s. Tiny because BLK is small.
    __shared__ float ssum[BLK];
    ssum[threadIdx.x] = s;
    __syncthreads();
    for (int off = BLK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    float inv = rsqrtf(ssum[0] + 1e-6f);

    // Second pass: emit normalized output, reading from local_buf (register-spilled).
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[row * cols + c] = local_buf[c] * inv;
    }
}

int main() {
    std::vector<float> h_x(R * C), h_y(R * C);
    for (int i = 0; i < R * C; ++i) {
        unsigned u = static_cast<unsigned>(i) * 1664525u + 1013904223u;
        h_x[i] = static_cast<float>(u & 0xffff) * 1e-4f;
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, R * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, R * C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), R * C * sizeof(float), cudaMemcpyHostToDevice));

    row_l2_normalize_slow<<<R, BLK>>>(d_x, d_y, R, C);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, R * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Host-side verify.
    int errors = 0;
    for (int r = 0; r < R; ++r) {
        double s = 0.0;
        for (int c = 0; c < C; ++c) s += double(h_x[r*C+c]) * double(h_x[r*C+c]);
        float inv = 1.0f / std::sqrt(float(s) + 1e-6f);
        for (int c = 0; c < C; ++c) {
            float exp = h_x[r*C+c] * inv;
            if (std::abs(h_y[r*C+c] - exp) > 1e-3f) ++errors;
        }
    }

    std::printf("row_l2_normalize: errors=%d %s\n",
                errors, errors == 0 ? "(PASS)" : "(FAIL)");
    std::printf("\nProfile with: ncu --set full ./starter_mb\n");
    std::printf("Three plausible fixes — only one is the actual limiter.\n");
    std::printf("Hint: which counter is most extreme?\n");
    std::printf("  (a) sectors-per-request   (uncoalesced reads/writes)\n");
    std::printf("  (b) registers-per-thread + Long Scoreboard stalls   (register spill)\n");
    std::printf("  (c) Achieved Occupancy   (block size too small)\n");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return errors == 0 ? 0 : 1;
}
