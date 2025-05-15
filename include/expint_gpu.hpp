#pragma once

#ifndef COMPILE_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace gpu {

#ifndef COMPILE_CPU_ONLY
// 分配 device 内存并 H2D
void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples);
void alloc_and_copy_to_device(const double* h_x, double*& d_x, int samples);

// Free
void free_device(float* d_x);
void free_device(double* d_x);

// 新版 kernel：可指定 blockSize
void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples, int blockSize);
void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples, int blockSize);

// 兼容老接口（默认 blockSize = 128）
inline void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples) {
    expint_gpu_float(n, d_x, d_out, samples, 128);
}
inline void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples) {
    expint_gpu_double(n, d_x, d_out, samples, 128);
}

#endif

} // namespace gpu
