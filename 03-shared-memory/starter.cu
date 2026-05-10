// Module 3 — starter scaffold. Solve the TODOs.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_utils.h"

constexpr int TILE = 32;

// TODO 1: implement the naive transpose.
//   out[x * H + y] = in[y * W + x]
// Reads will be coalesced (lane k of warp reads in[..., x = blockIdx.x*TILE + k]);
// writes will be uncoalesced (lane k writes out[..., x_t * H], stride H apart).
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out,
                                int W, int H) {
    // your code here
}

// TODO 2: implement the shared-memory tiled transpose (no padding).
//   * Declare a __shared__ float tile[TILE][TILE].
//   * Each thread loads in[y*W + x] into tile[threadIdx.y][threadIdx.x].
//   * __syncthreads().
//   * Compute the transposed coordinate: swap block x/y, then write
//     out[y_t * H + x_t] = tile[threadIdx.x][threadIdx.y].
__global__ void transpose_shared(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int W, int H) {
    // your code here
}

// TODO 3: copy your shared-memory transpose and change the tile dim to [TILE][TILE+1].
// Nothing else changes. The +1 staggers each row by 1 bank, eliminating the 32-way
// conflict on the column read-back.
__global__ void transpose_shared_padded(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int W, int H) {
    // your code here
}

// TODO 4 (canonical): 32×8 block; each thread handles 4 input rows.
// Block dim is (TILE, 8) so each block has 256 threads, not 1024. Each thread loops
// over `for (int j = 0; j < TILE; j += 8) tile[ty + j][tx] = in[(y + j) * W + x]`,
// then __syncthreads, then writes the corresponding 4 rows of the transposed tile.
// Use the +1 padding. This is what production transpose looks like.
__global__ void transpose_shared_4rows(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int W, int H) {
    // your code here
}

// TODO 5 (dynamic shared): same as TODO 3, but the shared tile is sized at launch.
// Declare `extern __shared__ float tile[]` and index it as
//   tile[ty * (TILE + 1) + tx]
// Pass (TILE * (TILE + 1) * sizeof(float)) as the third <<<>>> argument.
__global__ void transpose_shared_dynamic(const float* __restrict__ in,
                                         float* __restrict__ out,
                                         int W, int H) {
    // your code here
}

// TODO 6 (float4 global loads): load global tile via float4. Block dim (TILE/4, TILE)
// = (8, 32) means each thread loads one float4 per *row* of the tile (so each block
// covers a TILE×TILE region in 8×32 = 256 threads, each moving 4 floats). Decompose
// the float4 back to 4 floats when storing into shared memory at column positions
// (tx*4 + 0..3). Then __syncthreads + transposed write as in TODO 3.
__global__ void transpose_shared_vec4(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int W, int H) {
    // your code here
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
    constexpr int W = 8192, H = 8192;
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
        int errs = verify(h_a, h_b, W, H, name);
        std::printf("%-30s errors=%d %s\n", name, errs, errs == 0 ? "(PASS)" : "(FAIL)");
    };

    {
        dim3 block(TILE, TILE);
        dim3 grid(W / TILE, H / TILE);
        run("transpose_naive:",         [&] { transpose_naive<<<grid, block>>>(d_a, d_b, W, H); });
        run("transpose_shared:",        [&] { transpose_shared<<<grid, block>>>(d_a, d_b, W, H); });
        run("transpose_shared_padded:", [&] { transpose_shared_padded<<<grid, block>>>(d_a, d_b, W, H); });
    }
    {
        // 32×8 block — each thread does 4 rows.
        dim3 block(TILE, 8);
        dim3 grid(W / TILE, H / TILE);
        run("transpose_shared_4rows:",
            [&] { transpose_shared_4rows<<<grid, block>>>(d_a, d_b, W, H); });
    }
    {
        // Dynamic shared: TILE × (TILE+1) floats.
        dim3 block(TILE, TILE);
        dim3 grid(W / TILE, H / TILE);
        size_t shmem = TILE * (TILE + 1) * sizeof(float);
        run("transpose_shared_dynamic:",
            [&] { transpose_shared_dynamic<<<grid, block, shmem>>>(d_a, d_b, W, H); });
    }
    {
        // float4 loads: block (TILE/4, TILE) = (8, 32) covers a TILE×TILE tile.
        dim3 block(TILE / 4, TILE);
        dim3 grid(W / TILE, H / TILE);
        run("transpose_shared_vec4:",
            [&] { transpose_shared_vec4<<<grid, block>>>(d_a, d_b, W, H); });
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    return 0;
}
