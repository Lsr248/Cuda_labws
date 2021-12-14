#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <exception>
#include <string>
#include <chrono>
#include <type_traits>

const size_t NUM_THREADS = 512;

void HandleCudaStatus(cudaError status) {
	switch (status)
	{
	case cudaSuccess: break;
	case cudaErrorMemoryAllocation: throw std::exception("Error in memory allocation");
	case cudaErrorInvalidValue: throw std::exception("Invalid argument value");
	case cudaErrorInvalidDevicePointer: throw std::exception("Invalid device pointer");
	case cudaErrorInvalidMemcpyDirection: throw std::exception("Invalid copy dirrection");
	case cudaErrorInitializationError: throw std::exception("Error during initialization");
	case cudaErrorPriorLaunchFailure: throw std::exception("Error in previous launch");
	case cudaErrorInvalidResourceHandle: throw std::exception("Invalid resource handler");
	default: throw std::exception(("Unrecognized cuda status: " + std::to_string(static_cast<int>(status))).c_str());
	}
}

__global__ void matrix_mult(double* left, double* right, double* result, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < k && row < m)
	{
		double sum = 0;
		for (int i = 0; i < n; i++)
		{
			sum += left[row * n + i] * right[i * k + col];
		}
		result[row * k + col] = sum;
	}
}


void CheckCommutation()
{
	const int N = 10;
	const int array_size = N * N;
	const int NUM_THREADS = 5;




	double  a [array_size];
	double b [array_size];

	double* leftToRightCpu = new double[array_size];
	double* rightToLeftCpu = new double[array_size];

	double* dev_a;
	double* dev_b;
	double* dev_clr ;
	double* dev_crl;
	

	for (int i = 0; i < array_size; ++i)
	{
		a[i] = i+1.5;
		b[i] = i;
	}



	cudaMalloc((void**)&dev_a, array_size * sizeof(double));

	cudaMalloc((void**)&dev_b, array_size * sizeof(double));
	cudaMalloc((void**)&dev_clr, array_size * sizeof(double));
	cudaMalloc((void**)&dev_crl, array_size * sizeof(double));

	cudaMemcpy(dev_a, a, array_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, array_size * sizeof(double), cudaMemcpyHostToDevice);




	const int BLOCK_SIZE = 16;
	int gridRows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int gridCols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 dimGrid(gridCols, gridRows);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	matrix_mult << <dimGrid, dimBlock >> > (dev_a, dev_b, dev_clr, N, N, N);
	HandleCudaStatus(cudaGetLastError());
	matrix_mult << <dimGrid, dimBlock >> > (dev_b, dev_a, dev_crl, N, N, N);
	HandleCudaStatus(cudaGetLastError());


	HandleCudaStatus(cudaMemcpy((void*)leftToRightCpu, (void*)dev_clr, array_size * sizeof(double), cudaMemcpyDeviceToHost));
	
	HandleCudaStatus(cudaMemcpy((void*)rightToLeftCpu, (void*)dev_crl, array_size * sizeof(double), cudaMemcpyDeviceToHost));


	for (int i = 0; i < array_size; i++)
	{
		printf("%f ,%f, %f, %f \n", a[i], b[i], leftToRightCpu[i], rightToLeftCpu[i]);
	}

	for (int i = 0; i < array_size; ++i)
	{
		if (leftToRightCpu[i] != rightToLeftCpu[i])
		{
			std::cout << "Matrix are not commutative\n";
			return;
		}
	}
	std::cout << "Matrix are commutative\n";
}

int main()
{
	try
	{
		CheckCommutation();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}
