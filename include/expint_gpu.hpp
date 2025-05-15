#pragma once

#ifndef COMPILE_CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <vector>

// kernel接口
namespace gpu {

#ifndef COMPILE_CPU_ONLY
// 分配device内存并H2D
void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples);
void alloc_and_copy_to_device(const double* h_x, double*& d_x, int samples);

// Free
void free_device(float* d_x);
void free_device(double* d_x);

// CUDA kernel launchers
void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples);
void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples);
#endif

}  // namespace gpu
