#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 10


__global__ void matrixAdd(const int* A, const
    int* B, int* C)
{
    // Âû÷èñëåíèå èíäåêñà ýëåìåíòà ìàòðèöû íà GPU
    int i = blockIdx.x;
    int j = threadIdx.x;

    C[i * N + j] = A[i * N + j] + B[i * N + j];
    // printf("%d,%d,%d=%d+%d\n", i, j, C[i * N + j], A[i * N + j], B[i * N + j]);
}

int main()
{
    int a[N][N];
    int b[N][N];
    int c[N][N];
    int* ca;
    int* cb;
    int* cc;
    for (int i = 0; i < N * N; ++i) {
        *(*a + i) = 1;
    }

    for (int i = 0; i < N * N; ++i) {
        *(*b + i) = 2;
    }

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&ca, N * N * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&cb, N * N * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }


    cudaStatus = cudaMalloc((void**)&cc, N * N * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }





    cudaStatus = cudaMemcpy(ca, &a, N * N * sizeof(int), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
        return 1;
    }

    cudaStatus = cudaMemcpy(cb, &b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed!");
        return 1;
    }

    // Íà÷àòü îòñ÷åòà âðåìåíè
    // Çàïóñê ÿäðà
    matrixAdd << <N, N >> > (ca, cb, cc);

    cudaStatus = cudaMemcpy(&c, cc, N * N * sizeof(int), cudaMemcpyDeviceToHost);


    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    cudaFree(ca);
    cudaFree(cb);
    cudaFree(cc);
    return 0;
}