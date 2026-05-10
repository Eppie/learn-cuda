// Module 3 — bandwidth comparison: naive vs shared vs shared+pad transpose,
// plus the production-shape 32×8 / 4-rows-per-thread variant, dynamic shmem,
// and float4 global loads.
//
// Each kernel is verified once (PASS/FAIL printed) before timing is reported, so
// silently-wrong implementations can't post fictitious GB/s.

#include <cstdio>
#include <vector>

#include "bench.h"
#include "cuda_utils.h"

constexpr int TILE = 32;

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int W, int H) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) {
        out[x * H + y] = in[y * W + x];
    }
}

__global__ void transpose_shared(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int W, int H) {
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    __syncthreads();
    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;
    if (x_t < H && y_t < W) out[y_t * H + x_t] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_shared_padded(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int W, int H) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    __syncthreads();
    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;
    if (x_t < H && y_t < W) out[y_t * H + x_t] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_shared_4rows(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int W, int H) {
    constexpr int ROWS = 8;
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < W && (y + j) < H) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * W + x];
    }
    __syncthreads();
    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE; j += ROWS) {
        if (x_t < H && (y_t + j) < W) out[(y_t + j) * H + x_t] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transpose_shared_dynamic(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int W, int H) {
    extern __shared__ float tile[];
    constexpr int LD = TILE + 1;
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) tile[threadIdx.y * LD + threadIdx.x] = in[y * W + x];
    __syncthreads();
    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;
    if (x_t < H && y_t < W) out[y_t * H + x_t] = tile[threadIdx.x * LD + threadIdx.y];
}

__global__ void transpose_shared_vec4(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int W, int H) {
    __shared__ float tile[TILE][TILE + 1];
    int tx4 = threadIdx.x;          // 0..7
    int ty  = threadIdx.y;          // 0..31
    int x0  = blockIdx.x * TILE + tx4 * 4;
    int y   = blockIdx.y * TILE + ty;

    if (x0 < W && y < H) {
        const float4* in4 = reinterpret_cast<const float4*>(&in[y * W + x0]);
        float4 v = *in4;
        tile[ty][tx4 * 4 + 0] = v.x;
        tile[ty][tx4 * 4 + 1] = v.y;
        tile[ty][tx4 * 4 + 2] = v.z;
        tile[ty][tx4 * 4 + 3] = v.w;
    }
    __syncthreads();

    int y_t = blockIdx.x * TILE + ty;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int x_t = blockIdx.y * TILE + tx4 * 4 + i;
        if (x_t < H && y_t < W) out[y_t * H + x_t] = tile[tx4 * 4 + i][ty];
    }
}

static int verify(const std::vector<float>& a, const std::vector<float>& b, int W, int H) {
    int errors = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (a[y * W + x] != b[x * H + y]) ++errors;
    return errors;
}

int main() {
    constexpr int W     = 8192;
    constexpr int H     = 8192;
    constexpr int ITERS = 50;
    const     long bytes = static_cast<long>(W) * H * 2 * sizeof(float); // read + write

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Matrix: %d x %d = %.0f MB,  bytes/launch = %.0f MB\n\n",
                W, H, W * H * sizeof(float) / 1.0e6, bytes / 1.0e6);

    std::vector<float> h_a(W * H), h_b(W * H);
    for (int i = 0; i < W * H; ++i) h_a[i] = static_cast<float>(i);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));

    auto verify_then_bench = [&](const char* name, auto&& launch) {
        CUDA_CHECK(cudaMemset(d_b, 0, W * H * sizeof(float)));
        launch();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, W * H * sizeof(float), cudaMemcpyDeviceToHost));
        int errs = verify(h_a, h_b, W, H);
        const char* tag = (errs == 0) ? "PASS" : "FAIL";
        float ms = bench_min_ms(ITERS, launch);
        float gbs = bytes / (ms * 1.0e6);
        std::printf("  %-32s [%s] %.3f ms   %.1f GB/s   (%.0f%% of peak)\n",
                    name, tag, ms, gbs, gbs / 1008.0f * 100.0f);
    };

    {
        dim3 block(TILE, TILE);
        dim3 grid(W / TILE, H / TILE);
        verify_then_bench("transpose_naive",
            [&] { transpose_naive<<<grid, block>>>(d_a, d_b, W, H); });
        verify_then_bench("transpose_shared",
            [&] { transpose_shared<<<grid, block>>>(d_a, d_b, W, H); });
        verify_then_bench("transpose_shared_padded",
            [&] { transpose_shared_padded<<<grid, block>>>(d_a, d_b, W, H); });
    }
    {
        dim3 block(TILE, 8);
        dim3 grid(W / TILE, H / TILE);
        verify_then_bench("transpose_shared_4rows  (32x8)",
            [&] { transpose_shared_4rows<<<grid, block>>>(d_a, d_b, W, H); });
    }
    {
        dim3 block(TILE, TILE);
        dim3 grid(W / TILE, H / TILE);
        size_t shmem = TILE * (TILE + 1) * sizeof(float);
        verify_then_bench("transpose_shared_dynamic",
            [&] { transpose_shared_dynamic<<<grid, block, shmem>>>(d_a, d_b, W, H); });
    }
    {
        dim3 block(TILE / 4, TILE);
        dim3 grid(W / TILE, H / TILE);
        verify_then_bench("transpose_shared_vec4   (8x32)",
            [&] { transpose_shared_vec4<<<grid, block>>>(d_a, d_b, W, H); });
    }

    std::printf("\nRTX 4090 peak DRAM bandwidth: ~1008 GB/s.\n");
    std::printf("(For reference, plain copy from Module 2 hit ~95%% of peak.)\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
