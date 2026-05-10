// Module 11 — starter scaffold. Build a CUDA Graph manually (without stream
// capture) to drive home what the API does under the hood.

#include <cstdio>

#include "cuda_utils.h"

__global__ void increment_kernel(int* counter, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicAdd(counter, 1);
}

int main() {
    constexpr int N = 1024;

    int* d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ========================================================================
    // TODO: build a graph that does three increment_kernel calls in sequence,
    // *without* using cudaStreamBeginCapture. Instead use the explicit API:
    //
    //   cudaGraph_t graph;
    //   cudaGraphCreate(&graph, 0);
    //
    //   cudaKernelNodeParams kp = {};
    //   kp.func = (void*)increment_kernel;
    //   kp.gridDim = dim3(1);
    //   kp.blockDim = dim3(N);
    //   void* kargs[] = { (void*)&d_counter, (void*)&N };
    //   kp.kernelParams = kargs;
    //
    //   cudaGraphNode_t n0, n1, n2;
    //   cudaGraphAddKernelNode(&n0, graph, nullptr, 0, &kp);
    //   cudaGraphAddKernelNode(&n1, graph, &n0, 1, &kp);   // depends on n0
    //   cudaGraphAddKernelNode(&n2, graph, &n1, 1, &kp);   // depends on n1
    //
    //   cudaGraphExec_t gx;
    //   cudaGraphInstantiate(&gx, graph, nullptr, nullptr, 0);
    //
    // Then launch the graph 1000 times, sync, and read d_counter to verify the
    // count is 3000 * N.
    // ========================================================================

    // your code here

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("counter = %d (expected %d)\n", total, 3 * 1000 * N);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_counter));
    return 0;
}
