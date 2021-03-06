#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <exception>
#include <string>

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


__global__ void dzeta_fucntion(float* sum, float s)
{

	sum[threadIdx.x] = 1.f / powf(float(threadIdx.x + 1), s);

}

int main()
{
	const int NUM_THREADS = 512;
	double dzeta = 0;
	float sum[NUM_THREADS] = {};
	float* dev_sum;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&dev_sum, NUM_THREADS * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}
	cudaStatus = cudaMemcpy(dev_sum, sum, sizeof(float) * NUM_THREADS, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 2;
	}
	dzeta_fucntion << <1, NUM_THREADS >> > (dev_sum, 5.f);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 3;
	}
	cudaStatus = cudaMemcpy(sum, dev_sum, sizeof(float) * NUM_THREADS, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 4;
	}
	for (int i = 0; i < NUM_THREADS; ++i)
		dzeta += sum[i];
	std::cout << "Dzeta: " << dzeta << std::endl;

	cudaFree(dev_sum);
}