
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#define N 100



__global__ void skalarProduct(double* c, double* cc, const double* a, const double* b)
{
    __shared__ double ash[32];
    __shared__ double bsh[32];
    // Копирование из глобальной памяти
    if (blockIdx.x * 32 + threadIdx.x < N) {
        ash[threadIdx.x] = a[blockIdx.x * 32 + threadIdx.x];
        bsh[threadIdx.x] = b[blockIdx.x * 32 + threadIdx.x];
    }
    // Синхронизация нитей

    __syncthreads();

    if (blockIdx.x * 32 + threadIdx.x < N) {
        //printf("NUM %d = threadash[%d] = %f + %f \n", blockIdx.x * 32 + threadIdx.x, threadIdx.x, ash[threadIdx.x], bsh[threadIdx.x]);
        cc[blockIdx.x * 32 + threadIdx.x] = ash[threadIdx.x] * bsh[threadIdx.x];
        //printf("cc = %f\n", cc[blockIdx.x * 32 + threadIdx.x]);
    }


    __syncthreads();

    if (blockIdx.x * 32 + threadIdx.x == 0) {

        for (int i = 0; i < N; ++i)
            *c += cc[i];
        // printf("c = %f\n", *c);
    }


    // Вычисление скалярного произведения

}

int main()
{

    double a[N];
    double c = 0;

    for (int i = 0; i < N; ++i) 
        a[i] = 1;

    double* dev_a;
    double* dev_c;
    double* dev_cc;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 2;
    }




    cudaStatus = cudaMalloc((void**)&dev_cc, sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 4;
    }

    cudaStatus = cudaMemcpy(dev_c, &c, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 45;
    }


    int blockSize;

    blockSize = N / 32 + 1;


    cudaEvent_t start, stop;
    float elapsedTime;
    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // запись события
    cudaEventRecord(start, 0);

    // Launch a kernel on the GPU with one thread for each element.
    skalarProduct << <blockSize, 32 >> > (dev_c, dev_cc, dev_a, dev_a);

    cudaEventRecord(stop, 0);
    // ожидание завершения работы ядра
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // вывод информации
    printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);
    // уничтожение события
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStatus = cudaMemcpy(&c, dev_c, sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 6;
    }
    printf("skal product = %f\n", sqrt(c));

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 7;
    }



    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 8;
    }


    cudaFree(dev_c);
    cudaFree(dev_a);
  



    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 10;
    }



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 11;
    }

    return 0;
}

