#include "expint_gpu.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace gpu {

// 内存分配/释放
void alloc_and_copy_to_device(const float* h_x, float*& d_x, int samples) {
    size_t bytes = samples * sizeof(float);
    cudaError_t err = cudaMalloc((void**)&d_x, bytes);
    if (err != cudaSuccess) { d_x = nullptr; return; }
    err = cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_x); d_x = nullptr; }
}
void alloc_and_copy_to_device(const double* h_x, double*& d_x, int samples) {
    size_t bytes = samples * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&d_x, bytes);
    if (err != cudaSuccess) { d_x = nullptr; return; }
    err = cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_x); d_x = nullptr; }
}
void free_device(float* d_x) { if (d_x) cudaFree(d_x); }
void free_device(double* d_x) { if (d_x) cudaFree(d_x); }

// CUDA核函数（float）
__device__ float exponentialIntegralFloat_dev(const int n, const float x) {
    const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n == 0) { ans = expf(-x) / x; }
    else {
        if (x > 1.0f) {
            b = x + n; c = 3.4e38f; d = 1.0f / b; h = d;
            for (i = 1; i <= 10000; i++) {
                a = -i * (nm1 + i);
                b += 2.0f; d = 1.0f / (a * d + b);
                c = b + a / c; del = c * d; h *= del;
                if (fabsf(del - 1.0f) <= epsilon) { ans = h * expf(-x); return ans; }
            }
            ans = h * expf(-x);
        } else {
            ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
            fact = 1.0f;
            for (i = 1; i <= 10000; i++) {
                fact *= -x / i;
                if (i != nm1) del = -fact / (i - nm1);
                else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                    del = fact * (-logf(x) + psi);
                }
                ans += del;
                if (fabsf(del) < fabsf(ans) * epsilon) return ans;
            }
        }
    }
    return ans;
}

__global__ void expint_kernel_float(int n, const float* x, float* out, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < samples)
        out[idx] = exponentialIntegralFloat_dev(n, x[idx]);
}

void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples, int blockSize) {
    int block = blockSize > 0 ? blockSize : 128;
    int grid  = (samples + block - 1) / block;
    expint_kernel_float<<<grid, block>>>(n, d_x, d_out, samples);
    cudaDeviceSynchronize();
}

// CUDA核函数（double）
__device__ double exponentialIntegralDouble_dev(const int n, const double x) {
    const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n == 0) { ans = exp(-x) / x; }
    else {
        if (x > 1.0) {
            b = x + n; c = 1e308; d = 1.0 / b; h = d;
            for (i = 1; i <= 10000; i++) {
                a = -i * (nm1 + i);
                b += 2.0; d = 1.0 / (a * d + b);
                c = b + a / c; del = c * d; h *= del;
                if (fabs(del - 1.0) <= epsilon) { ans = h * exp(-x); return ans; }
            }
            ans = h * exp(-x);
        } else {
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
            fact = 1.0;
            for (i = 1; i <= 10000; i++) {
                fact *= -x / i;
                if (i != nm1) del = -fact / (i - nm1);
                else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
        }
    }
    return ans;
}

__global__ void expint_kernel_double(int n, const double* x, double* out, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < samples)
        out[idx] = exponentialIntegralDouble_dev(n, x[idx]);
}

void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples, int blockSize) {
    int block = blockSize > 0 ? blockSize : 128;
    int grid  = (samples + block - 1) / block;
    expint_kernel_double<<<grid, block>>>(n, d_x, d_out, samples);
    cudaDeviceSynchronize();
}

} // namespace gpu
