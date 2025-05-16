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

// float 内核主体
__device__ float exponentialIntegralFloat_dev(const int n, const float x) {
    const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n == 0) return expf(-x) / x;

    if (x > 1.0f) {
        b = x + n; c = 3.4e38f; d = 1.0f / b; h = d;
        for (i = 1; i <= 10000; i++) {
            a = -i * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= epsilon)
                return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
        fact = 1.0f;
        for (i = 1; i <= 10000; i++) {
            fact *= -x / i;
            if (i != nm1)
                del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon)
                return ans;
        }
        return ans;
    }
}

// float kernel 使用 shared memory tile
__global__ void expint_kernel_float(int n, const float* x, float* out, int samples) {
    extern __shared__ float tile_x_float[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < samples)
        tile_x_float[tid] = x[idx];

    __syncthreads();

    if (idx < samples)
        out[idx] = exponentialIntegralFloat_dev(n, tile_x_float[tid]);
}

void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples, int blockSize) {
    int block = blockSize > 0 ? blockSize : 128;
    int grid = (samples + block - 1) / block;
    size_t shared_mem_bytes = block * sizeof(float);
    expint_kernel_float<<<grid, block, shared_mem_bytes>>>(n, d_x, d_out, samples);
    cudaDeviceSynchronize();
}

// double 内核主体
__device__ double exponentialIntegralDouble_dev(const int n, const double x) {
    const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n == 0) return exp(-x) / x;

    if (x > 1.0) {
        b = x + n; c = 1e308; d = 1.0 / b; h = d;
        for (i = 1; i <= 10000; i++) {
            a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon)
                return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
        fact = 1.0;
        for (i = 1; i <= 10000; i++) {
            fact *= -x / i;
            if (i != nm1)
                del = -fact / (i - nm1);
            else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon)
                return ans;
        }
        return ans;
    }
}

// double kernel 使用 shared memory tile
__global__ void expint_kernel_double(int n, const double* x, double* out, int samples) {
    extern __shared__ double tile_x_double[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < samples)
        tile_x_double[tid] = x[idx];

    __syncthreads();

    if (idx < samples)
        out[idx] = exponentialIntegralDouble_dev(n, tile_x_double[tid]);
}

void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples, int blockSize) {
    int block = blockSize > 0 ? blockSize : 128;
    int grid = (samples + block - 1) / block;
    size_t shared_mem_bytes = block * sizeof(double);
    expint_kernel_double<<<grid, block, shared_mem_bytes>>>(n, d_x, d_out, samples);
    cudaDeviceSynchronize();
}

}  // namespace gpu
