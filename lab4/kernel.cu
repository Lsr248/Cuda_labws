
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

__global__ void add(unsigned int* nin)
{
    int i = threadIdx.x;
    int j = blockIdx.x;

    double k = double(i)/N;
    double p = double(j)/N;

    if (ChechPoint(&k, &p))
        atomicAdd(nin, 1);
}

int main()
{


    unsigned int nin = 0;

    unsigned int* dev_nin;
    //Скопировать в gpu
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_nin, sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMemcpy(dev_nin, &nin, sizeof(unsigned int), cudaMemcpyHostToDevice);
 

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
        return 1;
    }

    //передать в гпу
    add <<<N, N >>>(dev_nin) ;
    cudaStatus = cudaMemcpy(&nin, dev_nin, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //освободить память

    printf("%f\n", double(nin) * 4 / N / N);

    cudaFree(dev_nin);
    return 0;

}
