// Module 9 — starter scaffold. Solve the TODOs.

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda_utils.h"

constexpr int BLK     = 256;
constexpr int N_WARPS = BLK / 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < N_WARPS) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}
__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < N_WARPS) ? smem[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// ============================================================================
// TODO 1: numerically stable softmax. One block per row.
// Pass 1: row max via block_reduce_max.
// Pass 2: row sum of exp(x - row_max) via block_reduce_sum.
// Pass 3: write y[i] = exp(x[i] - row_max) / row_sum.
// ============================================================================
__global__ void softmax_fused(const float* __restrict__ in,
                              float* __restrict__ out,
                              int rows, int cols) {
    // your code here
}

// ============================================================================
// TODO 2: fused layernorm. Single pass collects sum(x) and sum(x²); second pass
// writes y[i] = gamma[i] * (x[i] - mean) * rstd + beta[i].
// ============================================================================
__global__ void layernorm_fused(const float* __restrict__ in,
                                float* __restrict__ out,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                int rows, int cols, float eps) {
    // your code here
}

// ============================================================================
// Verification harness
// ============================================================================
constexpr int ROWS = 4096;
constexpr int COLS = 4096;

static void host_softmax(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        float m = -INFINITY;
        for (int i = 0; i < cols; ++i) m = std::fmax(m, xr[i]);
        float s = 0.0f;
        for (int i = 0; i < cols; ++i) s += std::exp(xr[i] - m);
        float inv = 1.0f / s;
        for (int i = 0; i < cols; ++i) yr[i] = std::exp(xr[i] - m) * inv;
    }
}

static void host_layernorm(const float* x, float* y, const float* g, const float* b,
                           int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr = y + r * cols;
        double s = 0.0, sq = 0.0;
        for (int i = 0; i < cols; ++i) { s += xr[i]; sq += xr[i] * xr[i]; }
        double m = s / cols;
        double v = sq / cols - m * m;
        double rstd = 1.0 / std::sqrt(v + eps);
        for (int i = 0; i < cols; ++i) yr[i] = g[i] * (xr[i] - m) * rstd + b[i];
    }
}

int main() {
    std::srand(123);
    std::vector<float> h_x(ROWS * COLS), h_g(COLS), h_b(COLS);
    for (auto& v : h_x) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_g) v = (std::rand() % 2001 - 1000) * 1e-3f;
    for (auto& v : h_b) v = (std::rand() % 2001 - 1000) * 1e-3f;

    std::vector<float> h_ref_softmax(ROWS * COLS), h_ref_ln(ROWS * COLS);
    host_softmax  (h_x.data(), h_ref_softmax.data(), ROWS, COLS);
    host_layernorm(h_x.data(), h_ref_ln.data(), h_g.data(), h_b.data(),
                   ROWS, COLS, 1e-5f);

    float *d_x, *d_y, *d_g, *d_b;
    CUDA_CHECK(cudaMalloc(&d_x, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, COLS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), COLS * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_got(ROWS * COLS);

    auto check = [&](const std::vector<float>& expected, const char* name, float tol) {
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_y, ROWS * COLS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float max_abs = 0.0f;
        int bad = -1;
        for (size_t i = 0; i < h_got.size(); ++i) {
            float d = std::fabs(h_got[i] - expected[i]);
            float allowed = std::fmax(tol * std::fabs(expected[i]), tol);
            if (d > allowed && bad < 0) bad = (int)i;
            if (d > max_abs) max_abs = d;
        }
        std::printf("%-22s max_abs=%.3e %s\n", name, max_abs,
                    bad < 0 ? "(PASS)" : "(FAIL)");
    };

    softmax_fused<<<ROWS, BLK>>>(d_x, d_y, ROWS, COLS);
    CUDA_CHECK_LAST();
    check(h_ref_softmax, "softmax_fused", 1e-4f);

    layernorm_fused<<<ROWS, BLK>>>(d_x, d_y, d_g, d_b, ROWS, COLS, 1e-5f);
    CUDA_CHECK_LAST();
    check(h_ref_ln, "layernorm_fused", 1e-3f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
