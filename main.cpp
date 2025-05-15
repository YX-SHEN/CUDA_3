//------------------------------------------------------------------------------
// File : main.cpp
//------------------------------------------------------------------------------

#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "include/expint_cpu.hpp"
#include "include/expint_gpu.hpp"

using namespace std;

// 原有声明
float   exponentialIntegralFloat(const int n,const float x);
double  exponentialIntegralDouble(const int n,const double x);
void    outputResultsCpu(const std::vector< std::vector< float  > > &resultsFloatCpu,const std::vector< std::vector< double > > &resultsDoubleCpu);
int     parseArguments(int argc, char **argv);
void    printUsage(void);

bool verbose,timing,cpu;
int maxIterations;
unsigned int n,numberOfSamples;
double a,b;  // The interval that we are going to use

int main(int argc, char *argv[]) {
    unsigned int ui,uj;
    cpu=true;
    verbose=false;
    timing=false;
    n=10;
    numberOfSamples=10;
    a=0.0;
    b=10.0;
    maxIterations=2000000000;

    struct timeval expoStart, expoEnd;

    parseArguments(argc, argv);

    if (verbose) {
        cout << "n=" << n << endl;
        cout << "numberOfSamples=" << numberOfSamples << endl;
        cout << "a=" << a << endl;
        cout << "b=" << b << endl;
        cout << "timing=" << timing << endl;
        cout << "verbose=" << verbose << endl;
    }

    // Sanity checks
    if (a>=b) {
        cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << endl;
        return 0;
    }
    if (n<=0) {
        cout << "Incorrect orders ("<<n<<") have been stated!" << endl;
        return 0;
    }
    if (numberOfSamples<=0) {
        cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
        return 0;
    }

    std::vector< std::vector< float  > > resultsFloatCpu;
    std::vector< std::vector< double > > resultsDoubleCpu;

    double timeTotalCpu=0.0;

    try {
        resultsFloatCpu.resize(n,vector< float >(numberOfSamples));
    } catch (std::bad_alloc const&) {
        cout << "resultsFloatCpu memory allocation fail!" << endl;    exit(1);
    }
    try {
        resultsDoubleCpu.resize(n,vector< double >(numberOfSamples));
    } catch (std::bad_alloc const&) {
        cout << "resultsDoubleCpu memory allocation fail!" << endl;   exit(1);
    }

    double x,division=(b-a)/((double)(numberOfSamples));

    if (cpu) {
        gettimeofday(&expoStart, NULL);
        for (ui=1;ui<=n;ui++) {
            for (uj=1;uj<=numberOfSamples;uj++) {
                x=a+uj*division;
                resultsFloatCpu[ui-1][uj-1]=exponentialIntegralFloat (ui,x);
                resultsDoubleCpu[ui-1][uj-1]=exponentialIntegralDouble (ui,x);
            }
        }
        gettimeofday(&expoEnd, NULL);
        timeTotalCpu=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
    }

    if (timing) {
        if (cpu) {
            printf ("calculating the exponentials on the cpu took: %f seconds\n",timeTotalCpu);
        }
    }

    if (verbose) {
        if (cpu) {
            outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
        }
    }

    // ==================== Step 2: GPU device alloc + H2D ====================
    // （插在 main 尾部，不影响任何 CPU 流程）
    std::vector<float> h_x(numberOfSamples);
    for (unsigned int i = 0; i < numberOfSamples; ++i)
        h_x[i] = a + (i + 1) * division;   // 与 CPU 输入一致

    float* d_x = nullptr;
    gpu::alloc_and_copy_to_device(h_x.data(), d_x, numberOfSamples);

    if (d_x) {
        std::cout << "Step2: GPU device memory allocated and copied OK!" << std::endl;
        gpu::free_device(d_x);
    } else {
        std::cerr << "Step2: Device memory allocation/copy failed!" << std::endl;
    }
    // ==================== End of Step 2 ====================

    return 0;
}

// 完全保留原有 outputResultsCpu
void    outputResultsCpu(const std::vector< std::vector< float  > > &resultsFloatCpu, const std::vector< std::vector< double > > &resultsDoubleCpu) {
    unsigned int ui,uj;
    double x,division=(b-a)/((double)(numberOfSamples));
    for (ui=1;ui<=n;ui++) {
        for (uj=1;uj<=numberOfSamples;uj++) {
            x=a+uj*division;
            std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
            std::cout << "exponentialIntegralFloat  (" << ui << "," << x <<")=" << resultsFloatCpu[ui-1][uj-1] << endl;
        }
    }
}

// 完全保留原有 parseArguments
int parseArguments (int argc, char *argv[]) {
    int c;
    while ((c = getopt (argc, argv, "cghn:m:a:b:tv")) != -1) {
        switch(c) {
            case 'c':
                cpu=false; break;     //Skip the CPU test
            case 'h':
                printUsage(); exit(0); break;
            case 'i':
                maxIterations = atoi(optarg); break;
            case 'n':
                n = atoi(optarg); break;
            case 'm':
                numberOfSamples = atoi(optarg); break;
            case 'a':
                a = atof(optarg); break;
            case 'b':
                b = atof(optarg); break;
            case 't':
                timing = true; break;
            case 'v':
                verbose = true; break;
            default:
                fprintf(stderr, "Invalid option given\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

// 完全保留原有 printUsage (一大段 usage 输出都不能动)
void printUsage () {
    printf("exponentialIntegral program\n");
    printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
    printf("This program will calculate a number of exponential integrals\n");
    printf("usage:\n");
    printf("exponentialIntegral.out [options]\n");
    printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
    printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
    printf("      -c           : will skip the CPU test\n");
    printf("      -g           : will skip the GPU test\n");
    printf("      -h           : will show this usage\n");
    printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
    printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
    printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
    printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
    printf("      -v           : will activate the verbose mode  (default: no)\n");
    printf("     \n");
}
