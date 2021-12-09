#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 10


__global__ void calcf(float* c, const float* a, const float* b)
{
    int i = threadIdx.x;
    c[i] = __fmul_rn(b[i], a[i]);
}

__global__ void calcd(double* c, double* a, double* b)
{
    int i = threadIdx.x;
    c[i] = __dmul_rn(b[i], a[i]);
}


__host__ int kernel1() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float a[] = { 1,1,1,1,1,1,1,1,1,1 };
    float b[] = { 2,2,2,2,2,2,2,2,2,2 };
    float c[N];
    float* ca, * cb, * cc;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&ca, N * sizeof(float));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&cb, N * sizeof(float));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&cc, N * sizeof(float));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }


    cudaStatus = cudaMemcpy(ca, a, N * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(cb, b, N * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaEventRecord(start, 0);

    calcf << <1, N >> > (cc, ca, cb);
    // Copy input vectors from host memory to GPU buffers.


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("KernelTime float: %.2f milliseconds\n",
        KernelTime);
    cudaStatus = cudaMemcpy(c, cc, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    float ansv = 0;
    for (int i = 0; i < N; ++i) {
        ansv += c[i];
    }
    printf("ansv if %f\n", ansv);
    cudaFree(ca);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}


__host__ int kernal2() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double a[] = { 1,1,1,1,1,1,1,1,1,1 };
    double b[] = { 2,2,2,2,2,2,2,2,2,2 };
    double c[N];
    double* ca, * cb, * cc;

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&ca, N * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&cb, N * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&cc, N * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }


    cudaStatus = cudaMemcpy(ca, a, N * sizeof(double), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(cb, b, N * sizeof(double), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaEventRecord(start, 0);

    calcd << <1, N >> > (cc, ca, cb);
    // Copy input vectors from host memory to GPU buffers.


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("KernelTime double: %.2f milliseconds\n",
        KernelTime);
    cudaStatus = cudaMemcpy(c, cc, N * sizeof(double), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    double ansv = 0;
    for (int i = 0; i < N; ++i) {
        ansv += c[i];
    }
    printf("ansv if %f\n", ansv);
    cudaFree(ca);
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

}

int main() {

    kernel1();
    kernal2();

    return 0;
}