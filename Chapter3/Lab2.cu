
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath> 
#include <stdio.h>

#define N  100


__global__ void add(double h, double* c) {
	int t = blockIdx.x;
	double x = h * t;
	double x1 = h * (t + 1);
	if (t < N)
	{
		c[t] = (sqrtf(1.0 - x * x) + sqrtf(1 - x1 * x1)) / 2;
	}
}

int main()
{
	// переменные на CPU
	double  pi = 0;
	double c[N];
	double* dev_c;
	double h = 1 / (double)N;



	cudaMalloc((void**)&dev_c, N * sizeof(double));



	add << <N, 1 >> > (h, dev_c);
	cudaMemcpy(c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
	{
		pi += c[i];
	}
	pi = 4 * pi / N;
	// вывод информации
	printf("%f", pi);
	// очищение памяти на GPU

	cudaFree(dev_c);

	return 0;
}