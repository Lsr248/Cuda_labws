
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// скалярное произведение векторов

#define N 5
__global__ void addKernel(int* a, int* b, int* c)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int  a[N];
    int b[N];
    int c[N];


    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    for (int i = 0; i < N; i++)
    {
        a[i] = i * i;
        b[i] = -i;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_a, N * sizeof(int));

    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));


    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    addKernel << <1, N >> > (dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < N; ++i)
        sum += c[i];
    printf("%d",sum);
    /*for (int i = 0; i < N; i++)
    {
        printf("%d * %d = %d\n", a[i], b[i], c[i]);
    }*/
    // вывод результатов
    // освобождение памяти
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.