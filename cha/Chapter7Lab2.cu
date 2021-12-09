
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#define N 100


__device__ double f(double* x) {
    return *x * *x;
}


__global__ void integralByMiddleQuad(double* c, const double* h, const double* a)
{
    __shared__ double csh[32];

    double x1 = *a + (double)(blockIdx.x * 32 + threadIdx.x) * *h;



    double x2 = x1 + *h / 2;
    csh[threadIdx.x] = f(&x2);

    

    __syncthreads();


    if (threadIdx.x == 0) {

        for (int i = 0; i < 32 && blockIdx.x * 32 + i < N ; ++i) {
            c[blockIdx.x] += csh[i];
        }
    }

}

int main()
{
    const int blockSize = N / 32 + 1;

    double a=3;
    double b=6;
    double h= (b - a) / N;
    double c[blockSize];

    for (int i = 0; i < blockSize; ++i) {
        c[i] = 0;
    }

    double* dev_a = 0;
    double* dev_h = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, blockSize*sizeof(double));

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
    cudaStatus = cudaMemcpy(dev_c, c, blockSize *sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 55;
    }

    // Launch a kernel on the GPU with one thread for each element.
    
    // Launch a kernel on the GPU with one thread for each element.
    integralByMiddleQuad << <blockSize, 32 >> > (dev_c, dev_h, dev_a);




    cudaStatus = cudaMemcpy(&c, dev_c, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 6;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 7;
    }

    double sum = 0;
    for (int i = 0; i < blockSize; ++i) {
        sum += c[i];
    }
    sum /= N / (b-a);
    printf("integralByMiddleQuad = %f\n", sum);


    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return 8;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 9;
    }



    cudaFree(dev_c);
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

