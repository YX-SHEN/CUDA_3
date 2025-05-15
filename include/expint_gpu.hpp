#pragma once
#include <cuda_runtime.h>

namespace gpu {

void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples);
void free_device(float* d_x);

} // namespace gpu
