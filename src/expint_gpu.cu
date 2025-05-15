#include "expint_gpu.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace gpu {

// 申请 device 内存并 H2D 拷贝
void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples) {
    size_t bytes = samples * sizeof(float);
    cudaError_t err;
    err = cudaMalloc((void**)&d_x, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        d_x = nullptr;
        return;
    }
    err = cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy (H2D) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        d_x = nullptr;
    }
}

void free_device(float* d_x) {
    if (d_x) cudaFree(d_x);
}

} // namespace gpu
