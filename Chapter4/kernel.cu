
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *nin, double h);

#define N 1000



__device__ bool ChechPoint(double* k, double *p)
{
    if ((*k) * (*k) + (*p) * (*p) <= 1)
        return true;
    return false;
}

__global__ void add(double* nin)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    double k = double(i)/N;
    double p = double(j)/N;

    if (ChechPoint(&k, &p))
        nin[threadIdx.x * N + blockIdx.x] = 1;
    else
        nin[threadIdx.x * N + blockIdx.x] = 0;
}

int main()
{

    double* nin = new double[N * N];
    double* dev_nin;
    //Скопировать в gpu
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_nin, N * N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
        return 1;
    }

    //передать в гпу
    add <<<N, N >>>(dev_nin) ;
    cudaStatus = cudaMemcpy(nin, dev_nin, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    //освободить память

    double q = 0;
    for (int i = 0; i < N * N; ++i) {
        q += nin[i];
    }
    printf("%f\n", q * 4 / N / N);

    cudaFree(dev_nin);
    return 0;

}
