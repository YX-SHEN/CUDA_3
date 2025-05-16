#include "expint_gpu.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <limits>
#include <algorithm>

// ---------- Global constant memory ----------
__constant__ int const_n;

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

// device-side precise En(x) (for float)
__device__ float d_expint_impl_float(int n, float x) {
    const float euler = 0.5772156649015329f;
    const float eps = 1e-30f;
    const float big = 3.4e38f;
    const int maxIter = 10000;
    if (n == 0) return expf(-x)/x;
    int nm1 = n - 1;
    if (x > 1.0f) {
        float b = x + n, c = big, d = 1.0f / b, h = d;
        for (int i = 1; i <= maxIter; ++i) {
            float a = -i * (nm1 + i);
            b += 2;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            float del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= eps) return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        float ans = (nm1 ? 1.0f / nm1 : -logf(x) - euler);
        float fact = 1.0f;
        for (int i = 1; i <= maxIter; ++i) {
            fact *= -x / i;
            float del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                float psi = -euler; for (int k = 1; k <= nm1; ++k) psi += 1.0f / k;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * eps) return ans;
        }
        return ans;
    }
}

__device__ double d_expint_impl_double(int n, double x) {
    const double euler = 0.5772156649015328606;
    const double eps = 1e-30;
    const double big = 1e308;
    const int maxIter = 10000;
    if (n == 0) return exp(-x)/x;
    int nm1 = n - 1;
    if (x > 1.0) {
        double b = x + n, c = big, d = 1.0 / b, h = d;
        for (int i = 1; i <= maxIter; ++i) {
            double a = -i * (nm1 + i);
            b += 2;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            double del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= eps) return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        double ans = (nm1 ? 1.0 / nm1 : -log(x) - euler);
        double fact = 1.0;
        for (int i = 1; i <= maxIter; ++i) {
            fact *= -x / i;
            double del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                double psi = -euler; for (int k = 1; k <= nm1; ++k) psi += 1.0 / k;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * eps) return ans;
        }
        return ans;
    }
}

// float kernel（shared memory + constant + streams）
__global__ void expint_kernel_float(const float* x, float* out, int samples) {
    extern __shared__ float tile_x_float[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < samples)
        tile_x_float[tid] = x[idx];

    __syncthreads();

    if (idx < samples)
        out[idx] = d_expint_impl_float(const_n, tile_x_float[tid]);
}

void expint_gpu_float(const int n, const float* d_x, float* d_out, int samples, int blockSize) {
    cudaMemcpyToSymbol(const_n, &n, sizeof(int));

    int block = blockSize > 0 ? blockSize : 128;
    int grid = (samples + block - 1) / block;
    size_t shared_mem_bytes = block * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    expint_kernel_float<<<grid, block, shared_mem_bytes, stream>>>(d_x, d_out, samples);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

__global__ void expint_kernel_double(const double* x, double* out, int samples) {
    extern __shared__ double tile_x_double[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < samples)
        tile_x_double[tid] = x[idx];

    __syncthreads();

    if (idx < samples)
        out[idx] = d_expint_impl_double(const_n, tile_x_double[tid]);
}

void expint_gpu_double(const int n, const double* d_x, double* d_out, int samples, int blockSize) {
    cudaMemcpyToSymbol(const_n, &n, sizeof(int));

    int block = blockSize > 0 ? blockSize : 128;
    int grid = (samples + block - 1) / block;
    size_t shared_mem_bytes = block * sizeof(double);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    expint_kernel_double<<<grid, block, shared_mem_bytes, stream>>>(d_x, d_out, samples);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

}  // namespace gpu
