// Module 3 — reference solution.
//
// Six matrix-transpose kernels:
//   1. transpose_naive          — uncoalesced writes
//   2. transpose_shared         — coalesced via shared mem, but with bank conflicts
//   3. transpose_shared_padded  — same plus +1 padding to kill bank conflicts
//   4. transpose_shared_4rows   — canonical 32×8 block, 4 rows per thread
//   5. transpose_shared_dynamic — same as (3) but with dynamic shared memory
//   6. transpose_shared_vec4    — float4 global loads, scalar shared writes

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

constexpr int TILE = 32;

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int W, int H) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) {
        out[x * H + y] = in[y * W + x];   // write is strided by H — uncoalesced
    }
}

__global__ void transpose_shared(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int W, int H) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) {
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    }
    __syncthreads();

    int x_t = blockIdx.y * TILE + threadIdx.x;     // swap block x/y for the write
    int y_t = blockIdx.x * TILE + threadIdx.y;
    if (x_t < H && y_t < W) {
        out[y_t * H + x_t] = tile[threadIdx.x][threadIdx.y];   // bank conflict
    }
}

__global__ void transpose_shared_padded(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int W, int H) {
    __shared__ float tile[TILE][TILE + 1];          // <-- +1 kills the bank conflict

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) {
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];
    }
    __syncthreads();

    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;
    if (x_t < H && y_t < W) {
        out[y_t * H + x_t] = tile[threadIdx.x][threadIdx.y];
    }
}

// 4 rows per thread, block dim (32, 8). Padded shared.
__global__ void transpose_shared_4rows(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int W, int H) {
    constexpr int ROWS = 8;        // blockDim.y
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < W && (y + j) < H) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * W + x];
        }
    }
    __syncthreads();

    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += ROWS) {
        if (x_t < H && (y_t + j) < W) {
            out[(y_t + j) * H + x_t] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Dynamic shared memory: same shape as TODO 3 but sized at launch.
__global__ void transpose_shared_dynamic(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int W, int H) {
    extern __shared__ float tile[];          // size = TILE*(TILE+1)*sizeof(float)
    constexpr int LD = TILE + 1;             // logical "stride" between rows

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H) {
        tile[threadIdx.y * LD + threadIdx.x] = in[y * W + x];
    }
    __syncthreads();

    int x_t = blockIdx.y * TILE + threadIdx.x;
    int y_t = blockIdx.x * TILE + threadIdx.y;
    if (x_t < H && y_t < W) {
        out[y_t * H + x_t] = tile[threadIdx.x * LD + threadIdx.y];
    }
}

// float4 global loads: block dim (TILE/4, TILE) = (8, 32). Each thread loads ONE
// float4 (= 4 contiguous floats) per row of the tile.
__global__ void transpose_shared_vec4(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int W, int H) {
    __shared__ float tile[TILE][TILE + 1];

    int tx4 = threadIdx.x;          // 0..7, each owns 4 columns
    int ty  = threadIdx.y;          // 0..31, one row per thread
    int x0  = blockIdx.x * TILE + tx4 * 4;   // first of 4 contiguous columns
    int y   = blockIdx.y * TILE + ty;

    // Vectorized global load: one float4 per thread.
    if (x0 < W && y < H) {
        const float4* in4 = reinterpret_cast<const float4*>(&in[y * W + x0]);
        float4 v = *in4;
        tile[ty][tx4 * 4 + 0] = v.x;
        tile[ty][tx4 * 4 + 1] = v.y;
        tile[ty][tx4 * 4 + 2] = v.z;
        tile[ty][tx4 * 4 + 3] = v.w;
    }
    __syncthreads();

    // Scalar transposed write (4 outputs per thread).
    // Output row in B is x_t = blockIdx.y * TILE + (tx4*4 + i)
    // Output col in B is y_t = blockIdx.x * TILE + ty
    int y_t = blockIdx.x * TILE + ty;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int x_t = blockIdx.y * TILE + tx4 * 4 + i;
        if (x_t < H && y_t < W) {
            out[y_t * H + x_t] = tile[tx4 * 4 + i][ty];
        }
    }
}

static int verify(const std::vector<float>& a, const std::vector<float>& b,
                  int W, int H, const char* name) {
    int errors = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float expected = a[y * W + x];
            float got      = b[x * H + y];
            if (expected != got) {
                if (errors < 3) {
                    std::printf("  [%s] mismatch at (x=%d, y=%d): got %f, expected %f\n",
                                name, x, y, got, expected);
                }
                ++errors;
            }
        }
    }
    return errors;
}

int main() {
    constexpr int W = 8192;
    constexpr int H = 8192;
    static_assert(W % TILE == 0 && H % TILE == 0, "dims must be a multiple of TILE");

    std::vector<float> h_a(W * H), h_b(W * H);
    for (int i = 0; i < W * H; ++i) h_a[i] = static_cast<float>(i);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&](const char* name, auto&& launch) {
        CUDA_CHECK(cudaMemset(d_b, 0, W * H * sizeof(float)));
        launch();
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, W * H * sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::printf("%-30s errors=%d\n", name, verify(h_a, h_b, W, H, name));
    };

    {
        dim3 block(TILE, TILE);
        dim3 grid(W / TILE, H / TILE);
        run("transpose_naive",         [&] { transpose_naive<<<grid, block>>>(d_a, d_b, W, H); });
        run("transpose_shared",        [&] { transpose_shared<<<grid, block>>>(d_a, d_b, W, H); });
        run("transpose_shared_padded", [&] { transpose_shared_padded<<<grid, block>>>(d_a, d_b, W, H); });
    }
    {
        dim3 block(TILE, 8);
        dim3 grid(W / TILE, H / TILE);
        run("transpose_shared_4rows",
            [&] { transpose_shared_4rows<<<grid, block>>>(d_a, d_b, W, H); });
    }
    {
        dim3 block(TILE, TILE);
        dim3 grid(W / TILE, H / TILE);
        size_t shmem = TILE * (TILE + 1) * sizeof(float);
        run("transpose_shared_dynamic",
            [&] { transpose_shared_dynamic<<<grid, block, shmem>>>(d_a, d_b, W, H); });
    }
    {
        dim3 block(TILE / 4, TILE);
        dim3 grid(W / TILE, H / TILE);
        run("transpose_shared_vec4",
            [&] { transpose_shared_vec4<<<grid, block>>>(d_a, d_b, W, H); });
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
