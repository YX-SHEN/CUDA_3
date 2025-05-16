//------------------------------------------------------------------------------
// File : main.cpp   (with COMPILE_CPU_ONLY support)
//------------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>

#include "include/expint_cpu.hpp"

#ifndef COMPILE_CPU_ONLY
#include "include/expint_gpu.hpp"
#endif

using namespace std;

/* ---------- prototypes from original CPU code ---------- */
float   exponentialIntegralFloat (int n, float  x);
double  exponentialIntegralDouble(int n, double x);

void outputResultsCpu(const vector<vector<float>>&  resF,
                      const vector<vector<double>>& resD);

int  parseArguments(int argc, char** argv);
void printUsage();

/* ---------- global flags / params ---------- */
bool verbose = false;
bool timing  = false;
bool cpu_on  = true;
bool gpu_on  = true;

int          maxIterations = 2000000000;
unsigned int n             = 10;
unsigned int samples       = 10;
double       a = 0.0, b = 10.0;
int          blockSize     = 128; // 默认block size, 可用-B修改

/* ---------- helper --------------------------------------------------------- */
inline double nowSeconds()
{
    struct timeval tv;  gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ========================================================================== */
int main(int argc, char* argv[])
{
    parseArguments(argc, argv);

    if(!cpu_on && !gpu_on){
        fprintf(stderr,"Error: both -c and -g specified - nothing to do!\n");
        return 1;
    }
    if(!gpu_on && !verbose && !timing){
        puts("[Info] GPU disabled with -g. Use -v or -t to see CPU results.");
    }

    if(a >= b){ puts("Error: a >= b!"); return 1; }
    if(n == 0 || samples == 0){ puts("Error: n or samples = 0!"); return 1; }

    const double dx = (b - a) / double(samples);

    // 1. 修正内存 shape：[n+1][samples]
    vector<vector<float>>  cpuFloat (n+1, vector<float>(samples, 0.f));
    vector<vector<double>> cpuDouble(n+1, vector<double>(samples, 0.0));
    double cpuTime = 0.0;

    if(cpu_on){
        double t0 = nowSeconds();
        // 2. 修正主循环 order=0 ~ n（包括 E0）
        for(unsigned int order = 0; order <= n; ++order){
            for(unsigned int j = 0; j < samples; ++j){
                double x = a + (j+1)*dx;
                cpuFloat [order][j] = exponentialIntegralFloat (order, float (x));
                cpuDouble[order][j] = exponentialIntegralDouble(order,       x );
            }
        }
        cpuTime = nowSeconds() - t0;
    }

    if(timing && cpu_on)
        printf("CPU total time: %.6f s\n", cpuTime);
    if(verbose && cpu_on)
        outputResultsCpu(cpuFloat, cpuDouble);

#ifndef COMPILE_CPU_ONLY
    // 1. GPU 结果 shape 也同步 [n+1][samples]
    vector<vector<float>>  gpuFloat (n+1, vector<float >(samples));
    vector<vector<double>> gpuDouble(n+1, vector<double>(samples));
    double gpu_total_time = 0.0;
    double gpu_time_float = 0.0, gpu_time_double = 0.0;

    if(gpu_on){
        vector<float>  hx (samples);
        vector<double> hxd(samples);
        for(unsigned int j=0;j<samples;++j){
            hx [j] = float (a + (j+1)*dx);
            hxd[j] =        a + (j+1)*dx ;
        }

        float  *dx_d  = nullptr, *dy_d  = nullptr;
        double *dxd_d = nullptr, *dyd_d = nullptr;

        double t0_all = nowSeconds();

        gpu::alloc_and_copy_to_device(hx .data(), dx_d , samples);
        gpu::alloc_and_copy_to_device(hxd.data(), dxd_d, samples);

        cudaMalloc((void**)&dy_d , samples*sizeof(float ));
        cudaMalloc((void**)&dyd_d, samples*sizeof(double));

        // 2. 循环 order=0 ~ n，完整输出
        for(unsigned int order = 0; order <= n; ++order){
            double t1 = nowSeconds();
            gpu::expint_gpu_float (order, dx_d , dy_d , samples, blockSize);
            gpu_time_float += (nowSeconds() - t1);

            double t2 = nowSeconds();
            gpu::expint_gpu_double(order, dxd_d, dyd_d, samples, blockSize);
            gpu_time_double += (nowSeconds() - t2);

            cudaMemcpy(gpuFloat [order].data(), dy_d , samples*sizeof(float ),  cudaMemcpyDeviceToHost);
            cudaMemcpy(gpuDouble[order].data(), dyd_d, samples*sizeof(double), cudaMemcpyDeviceToHost);
        }

        gpu::free_device(dx_d );  gpu::free_device(dy_d );
        gpu::free_device(dxd_d);  gpu::free_device(dyd_d);

        gpu_total_time = nowSeconds() - t0_all;
    }

    if(verbose && gpu_on){
        for(unsigned int order=0; order<=n; ++order){
            for(unsigned int j=0;j<samples;++j){
                double x = a + (j+1)*dx;
                printf("GPU==> n=%2u x=%g  float=%-12.8g  double=%.14g\n",
                       order, x, gpuFloat[order][j], gpuDouble[order][j]);
            }
        }
    }

    if(timing && gpu_on){
        printf("GPU total time (alloc+copy+all kernels+D2H+free): %.6f s\n", gpu_total_time);
        printf("  - Float kernel time (total):  %.6f s\n", gpu_time_float);
        printf("  - Double kernel time (total): %.6f s\n", gpu_time_double);
        if(cpu_on)
            printf("Speed-up (CPU/GPU): %.2fx\n", cpuTime / gpu_total_time);
    }

    // 3. 精度对比也修正为 order=0 ~ n
    if (cpu_on && gpu_on) {
        int bad = 0;
        for (unsigned int order = 0; order <= n; ++order) {
            for (unsigned int j = 0; j < samples; ++j) {
                float  diffF = fabs(gpuFloat[order][j] - cpuFloat[order][j]);
                double diffD = fabs(gpuDouble[order][j] - cpuDouble[order][j]);
                if (diffF > 1e-5f) ++bad;
                if (diffD > 1e-5 ) ++bad;
            }
        }

        // 始终输出一行汇总信息
        printf("[Precision Check] GPU vs CPU comparison: %s (threshold = 1e-5)\n",
               (bad == 0) ? "PASS" : "FAIL");

        // 如需详细 diff，可加 verbose 控制
        if (bad > 0 && verbose) {
            for (unsigned int order = 0; order <= n; ++order) {
                for (unsigned int j = 0; j < samples; ++j) {
                    float  diffF = fabs(gpuFloat[order][j] - cpuFloat[order][j]);
                    double diffD = fabs(gpuDouble[order][j] - cpuDouble[order][j]);
                    if (diffF > 1e-5f)
                        printf("WARNING float n=%u idx=%u diff=%g\n", order, j, diffF);
                    if (diffD > 1e-5)
                        printf("WARNING double n=%u idx=%u diff=%g\n", order, j, diffD);
                }
            }
            printf("Number of mismatches exceeding threshold: %d\n", bad);
        }
    }

#endif

    return 0;
}

// 输出接口同理修正
void outputResultsCpu(const vector<vector<float>>&  resF,
                      const vector<vector<double>>& resD)
{
    for(unsigned int order=0; order<=n; ++order){
        for(unsigned int j=0;j<samples;++j){
            double x = a + (j+1)*(b-a)/double(samples);
            printf("CPU==> n=%2u x=%g  double=%-12.8g  float=%-12.8g\n",
                   order, x, resD[order][j], resF[order][j]);
        }
    }
}

int parseArguments(int argc, char** argv)
{
    int opt;
    while((opt = getopt(argc, argv, "cghn:m:a:b:tvB:")) != -1){
        switch(opt){
            case 'c': cpu_on = false; break;
            case 'g': gpu_on = false; break;
            case 'n': n       = atoi(optarg); break;
            case 'm': samples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing  = true;  break;
            case 'v': verbose = true;  break;
            case 'B': blockSize = atoi(optarg); break;
            case 'h': printUsage(); exit(0);
            default : printUsage(); exit(1);
        }
    }
    return 0;
}

void printUsage()
{
    puts("exponentialIntegral program");
    puts("usage: exponentialIntegral.out [options]");
    puts("  -a value : interval start (default 0.0)");
    puts("  -b value : interval end   (default 10.0)");
    puts("  -c       : skip CPU");
    puts("  -g       : skip GPU");
    puts("  -n N     : highest order   (default 10)");
    puts("  -m M     : samples per order (default 10)");
    puts("  -B value : block size for CUDA kernel (default 128)");
    puts("  -t       : timing");
    puts("  -v       : verbose (print tables)");
    puts("  -h       : this help");
}
