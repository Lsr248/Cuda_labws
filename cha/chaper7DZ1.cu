
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#define N 100


__device__ double f(double* x) {
    return *x * *x;
}


__global__ void integralByTrapecoid(double* c, const double* h, const double* a)
{
    __shared__ double csh[32];

    double x1 = *a + (double)(blockIdx.x * 32 + threadIdx.x) * *h;



    double x2 = x1 + *h;
    csh[threadIdx.x] = (f(&x1) + f(&x2)) / 2;



    __syncthreads();


    if (threadIdx.x == 0) {

        for (int i = 0; i < 32 && blockIdx.x * 32 + i < N; ++i) {
            c[blockIdx.x] += csh[i];
        }
    }

}
__global__ void integralByMiddleQuad(double* c, const double* h, const double* a)
{
    __shared__ double csh[32];

    double x1 = *a + (double)(blockIdx.x * 32 + threadIdx.x) * *h;



    double x2 = x1 + *h/2;
    csh[threadIdx.x] =  f(&x2) ;



    __syncthreads();


    if (threadIdx.x == 0) {

        for (int i = 0; i < 32 && blockIdx.x * 32 + i < N; ++i) {
            c[blockIdx.x] += csh[i];
        }
    }

}
__global__ void integralBySimpson(double* c, const double* h, const double* a)
{
    __shared__ double csh[32];

    double x1 = *a + (double)(blockIdx.x * 32 + threadIdx.x) * *h;



    double x2 = x1 + *h ;
    double x_center = (x2 + x1) / 2;
    csh[threadIdx.x] = (f(&x1)+f(&x2)+4*f(&x_center));



    __syncthreads();


    if (threadIdx.x == 0) {

        for (int i = 0; i < 32 && blockIdx.x * 32 + i < N; ++i) {
            c[blockIdx.x] += csh[i];
        }
    }

}

int main()
{
    const int blockSize = N / 32 + 1;

    double a = 3;
    double b = 6;
    double h = (b - a) / N;
    double c_center[blockSize];
    double c_trap[blockSize];
    double c_Simps[blockSize];
    double ans = 63;


    for (int i = 0; i < blockSize; ++i) {
        c_center[i] = 0;
        c_trap[i] = 0;
        c_Simps[i] = 0;
    }

    double* dev_a = 0;
    double* dev_h = 0;
    double* dev_center = 0;
    double* dev_trap = 0;
    double* dev_Simps = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_center, blockSize * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_trap, blockSize * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_Simps, blockSize * sizeof(double));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 2;
    }

    cudaStatus = cudaMalloc((void**)&dev_h, sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 3;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, &a, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 4;
    }

    cudaStatus = cudaMemcpy(dev_h, &h, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 5;
    }
    cudaStatus = cudaMemcpy(dev_center, c_center, blockSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_trap, c_trap, blockSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_Simps, c_Simps, blockSize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 55;
    }

    // Launch a kernel on the GPU with one thread for each element.

    // Launch a kernel on the GPU with one thread for each element.


    integralByTrapecoid << <blockSize, 32 >> > (dev_trap, dev_h, dev_a);
    integralByMiddleQuad << <blockSize, 32 >> > (dev_center, dev_h, dev_a);
    integralBySimpson << <blockSize, 32 >> > (dev_Simps, dev_h, dev_a);




    cudaStatus = cudaMemcpy(&c_center, dev_center, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 6;
    }

    double sum = 0;
    for (int i = 0; i < blockSize; ++i) {
        sum += c_center[i];
    }
    sum /= N / (b - a);

    printf("integralByMiddleQuad = %f\n", sum);



    cudaStatus = cudaMemcpy(&c_trap, dev_trap, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 7;
    }
    sum = 0;
    for (int i = 0; i < blockSize; ++i) {
        sum += c_trap[i];
    }
    sum /= N / (b - a);

    printf("integralByTrapecoid = %f\n", sum);


    cudaStatus = cudaMemcpy(&c_Simps, dev_Simps, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 8;
    }

    sum = 0;
    for (int i = 0; i < blockSize; ++i) {
        sum += c_Simps[i];
    }
    sum = (b - a) * sum / 6 / N;
    printf("integralBySimps = %f\n", sum);
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



    cudaFree(dev_center);
    cudaFree(dev_trap);
    cudaFree(dev_Simps);
    cudaFree(dev_a);
    cudaFree(dev_h);



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

