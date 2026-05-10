// Module 13 — minimal `mma.sync` example.
//
// Goal: execute exactly one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
// instruction with a single warp, using 16x16 (A) * 16x8 (B) -> 16x8 (C)
// fragments, then verify against a host reference.
//
// Why this matters: production ML kernels (CUTLASS, FlashAttention) target
// `mma.sync` PTX directly rather than the C++ `wmma::` wrappers, because
// `mma.sync`'s lane-element mapping is documented and stable, while WMMA's
// fragment layout is opaque (you can't do epilogue fusion easily on top of it).
//
// Lane->element mapping for m16n8k16, FP16 inputs, FP32 accumulator
// (PTX ISA 7.0+ documentation, table "MMA .m16n8k16 with .f16 floating-point
// type", reproduced for sm_89):
//
//   A: 16x16 row-major. Each lane holds 8 FP16 elements packed into 4 b32 regs.
//   B: 16x8  col-major. Each lane holds 4 FP16 elements packed into 2 b32 regs.
//   C: 16x8  row-major. Each lane holds 4 FP32 elements in 4 b32 regs.
//
//   For A: lane L (0..31) = group g + tid t  where g = L/4, t = L%4. Lane
//   contains A[g, 2t..2t+1], A[g, 2t+8..2t+9], A[g+8, 2t..2t+1], A[g+8, 2t+8..2t+9].
//   For B: lane contains B[2t..2t+1, g], B[2t+8..2t+9, g].
//   For C: lane contains C[g, 2t..2t+1] and C[g+8, 2t..2t+1].
//
// We load these per-lane, run one mma.sync, store back, and compare to a host
// triple-loop GEMM.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

#include "cuda_utils.h"

constexpr int M = 16;
constexpr int N = 8;
constexpr int K = 16;

// One warp does one mma. A,B in row-major / col-major as required by the
// instruction selector below.
__global__ void mma_m16n8k16_kernel(const __half* __restrict__ A,
                                    const __half* __restrict__ B,
                                    const float*  __restrict__ Cin,
                                    float*        __restrict__ Cout) {
    int lane = threadIdx.x;          // 0..31
    int g    = lane >> 2;            // 0..7
    int t    = lane & 3;             // 0..3

    // -------- pack A fragment (4 b32 regs = 8 FP16 elems) --------
    // a0: A[g,     2t .. 2t+1]   a1: A[g+8, 2t .. 2t+1]
    // a2: A[g,   2t+8 .. 2t+9]   a3: A[g+8, 2t+8 .. 2t+9]
    auto pack2 = [](__half x, __half y) -> uint32_t {
        uint32_t v;
        // little-endian: low 16b = x, high 16b = y
        v  = (uint32_t)*reinterpret_cast<uint16_t*>(&x);
        v |= ((uint32_t)*reinterpret_cast<uint16_t*>(&y)) << 16;
        return v;
    };
    uint32_t a0 = pack2(A[g     * K + 2*t    ], A[g     * K + 2*t + 1]);
    uint32_t a1 = pack2(A[(g+8) * K + 2*t    ], A[(g+8) * K + 2*t + 1]);
    uint32_t a2 = pack2(A[g     * K + 2*t + 8], A[g     * K + 2*t + 9]);
    uint32_t a3 = pack2(A[(g+8) * K + 2*t + 8], A[(g+8) * K + 2*t + 9]);

    // -------- pack B fragment (2 b32 regs = 4 FP16 elems) --------
    // B is col-major (each column is contiguous in memory). Lane holds:
    // b0: B[2t .. 2t+1, g]    b1: B[2t+8 .. 2t+9, g]
    // In our flat array, B is stored row-major NxK transposed -> col-major
    // means B[row, col] = Bmem[col * K + row]. We chose to store B as a flat
    // KxN matrix in column-major order to match this directly:
    // Bmem[col * K + row].
    uint32_t b0 = pack2(B[g * K + 2*t    ], B[g * K + 2*t + 1]);
    uint32_t b1 = pack2(B[g * K + 2*t + 8], B[g * K + 2*t + 9]);

    // -------- pack C accumulator (4 fp32) --------
    // c0: C[g,   2t .. 2t+1]    c2: C[g+8, 2t .. 2t+1]
    float c0 = Cin[g     * N + 2*t    ];
    float c1 = Cin[g     * N + 2*t + 1];
    float c2 = Cin[(g+8) * N + 2*t    ];
    float c3 = Cin[(g+8) * N + 2*t + 1];

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n\t"
        "{%0, %1, %2, %3},\n\t"
        "{%4, %5, %6, %7},\n\t"
        "{%8, %9},\n\t"
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1));

    // -------- store C back --------
    Cout[g     * N + 2*t    ] = c0;
    Cout[g     * N + 2*t + 1] = c1;
    Cout[(g+8) * N + 2*t    ] = c2;
    Cout[(g+8) * N + 2*t + 1] = c3;
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s\n", prop.name);
    std::printf("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 demo\n");
    std::printf("  A: %dx%d  row-major  fp16\n", M, K);
    std::printf("  B: %dx%d  col-major  fp16\n", K, N);
    std::printf("  C: %dx%d  row-major  fp32 (acc + Cin)\n\n", M, N);

    // --- host data ---
    std::vector<__half> hA(M * K), hB(K * N);
    std::vector<float>  hCin(M * N), hCout(M * N), hRef(M * N);

    std::srand(42);
    for (auto& x : hA)   x = __float2half((std::rand() % 200 - 100) * 0.01f);
    for (auto& x : hB)   x = __float2half((std::rand() % 200 - 100) * 0.01f);
    for (auto& x : hCin) x = (std::rand() % 200 - 100) * 0.01f;

    // host reference: C = A * B + Cin (A row-major, B col-major)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = __half2float(hA[i * K + k]);
                float b = __half2float(hB[j * K + k]);   // col-major B
                acc += a * b;
            }
            hRef[i * N + j] = acc + hCin[i * N + j];
        }
    }

    // --- device run ---
    __half *dA, *dB;
    float  *dCin, *dCout;
    CUDA_CHECK(cudaMalloc(&dA,    M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB,    K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dCin,  M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCout, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA,   hA.data(),   M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,   hB.data(),   K * N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCin, hCin.data(), M * N * sizeof(float),  cudaMemcpyHostToDevice));

    mma_m16n8k16_kernel<<<1, 32>>>(dA, dB, dCin, dCout);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(hCout.data(), dCout, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- compare ---
    float max_abs = 0.0f, max_rel = 0.0f;
    int   bad = -1;
    for (int i = 0; i < M * N; ++i) {
        float d = std::fabs(hCout[i] - hRef[i]);
        float r = d / std::fmax(std::fabs(hRef[i]), 1e-6f);
        if (d > 0.05f && bad < 0) bad = i;
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
    }
    std::printf("max_abs=%.3e   max_rel=%.3e   %s\n",
                max_abs, max_rel, bad < 0 ? "PASS" : "FAIL");
    if (bad >= 0) {
        std::printf("  first mismatch at idx %d: got=%.4f ref=%.4f\n",
                    bad, hCout[bad], hRef[bad]);
    }

    std::printf("\nInspect with:  make ptx  &&  grep mma.sync mma_sync_example.ptx\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dCin));
    CUDA_CHECK(cudaFree(dCout));
    return bad < 0 ? 0 : 1;
}
