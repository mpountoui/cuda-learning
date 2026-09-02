#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.hpp"
#include "Auxiliaries.cuh"

/*------------------------------------------------------------------------------------------*/

#define Check(Call)                                                                 \
{                                                                                   \
    cudaError_t error = Call;                                                       \
    if( error != cudaSuccess )                                                      \
    {                                                                               \
        printf("Error: %s: %d\n", __FILE__, __LINE__);                              \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));         \
        exit(1);                                                                    \
    }                                                                               \
}

namespace
{
/*------------------------------------------------------------------------------------------*/
    
    void checkResult(float* hostRef, float* gpuRef, int const N)
    {
        double const tol = 1e-8;
        
        for(int i = 0; i < N; ++i)
        {
            if( std::fabs(hostRef[i] - gpuRef[i]) > tol )
            {
                printf("Arrays does not match\n");
                return;
            }
        }
        
        printf("Arrays match\n");
    }
    
/*------------------------------------------------------------------------------------------*/
    
    void SumArraysOnHost(float* a, float* b, float* c, int size)
    {
        for(int i = 0; i < size; i++)
        {
            c[i] = a[i] + b[i];
        }
    }
    
/*------------------------------------------------------------------------------------------*/
    
    __global__ void SumArraysOnGPU(float* a, float* b, float* c, size_t n)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if (index < n){ c[index] = a[index] + b[index]; }
    }
    
/*------------------------------------------------------------------------------------------*/
    
    void executeGPUKernel(float* d_A, float* d_B, float* d_C, size_t nElem)
    {
        dim3 block(256, 1, 1);
        dim3 grid((nElem + block.x - 1) / block.x);
        
        Timer timer("SumArraysOnGPU");
        timer.start();
        SumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) printf("Launch error: %s\n", cudaGetErrorString(err));
        cudaDeviceSynchronize();
        timer.elapsedSeconds();
    }
}

/*------------------------------------------------------------------------------------------*/

int ArraysSum(size_t nElem)
{
    cudaSetDevice(0);
    size_t nBytes = nElem * sizeof(float);
    
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* hostRef = nullptr;
    float* gpuRef = nullptr;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    allocateAndInitializeHostMemory(nElem, h_A, h_B, hostRef, gpuRef);
    allocateDeviceMemory(nBytes, d_A, d_B, d_C);
    copyInputsToDevice(h_A, h_B, d_A, d_B, nBytes);
    
    executeGPUKernel(d_A, d_B, d_C, nElem);
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    Timer timerHost("SumArraysOnHost");
    timerHost.start();
    SumArraysOnHost(h_A, h_B, hostRef, nElem);
    timerHost.elapsedSeconds();
    
    checkResult(hostRef, gpuRef, nElem);
    freeMemory(h_A, h_B, hostRef, gpuRef, d_A, d_B, d_C);
    
    return 0;
}