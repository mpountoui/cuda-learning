#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include "timer.hpp"
#include "Auxiliaries.h"

namespace
{
/*------------------------------------------------------------------------------------------*/
    
    void checkResult(float* hostRef, float* gpuRef, int const nx, int const ny)
    {
        double const tol = 1e-8;
        int const nxy = nx * ny;
        
        for (int i = 0; i < nxy; ++i)
        {
            if (std::fabs(hostRef[i] - gpuRef[i]) > tol)
            {
                printf("Matrix does not match\n");
                return;
            }
        }
        
        printf("Matrix matches\n");
    }
    
/*------------------------------------------------------------------------------------------*/
    
    void SumMatrixOnHost(float* a, float* b, float* c, int nx, int ny)
    {
        for (int row = 0; row < ny; ++row)
        {
            for (int col = 0; col < nx; ++col)
            {
                int const idx = row * nx + col;
                c[idx] = a[idx] + b[idx];
            }
        }
    }
    
/*------------------------------------------------------------------------------------------*/
    
    __global__ void SumMatrixOnGPU(float* a, float* b, float* c, size_t n)
    {
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t index = y * gridDim.x * blockDim.x + x;
        if (index < n){ c[index] = a[index] + b[index]; }
    }
    
/*------------------------------------------------------------------------------------------*/
    
    void executeGPUKernel(float* d_A, float* d_B, float* d_C, size_t nx, size_t ny)
    {
        dim3 block(16, 16, 1);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        
        Timer timer("SumMatrixOnGPU");
        timer.start();
        SumMatrixOnGPU<<<grid, block>>>(d_A, d_B, d_C, nx * ny);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){ printf("Launch error: %s\n", cudaGetErrorString(err)); }
        cudaDeviceSynchronize();
        timer.elapsedSeconds();
    }
}

/*------------------------------------------------------------------------------------------*/

int MatrixSum(size_t nx, size_t ny)
{
    cudaSetDevice(0);
    size_t nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);
    
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* hostRef = nullptr;
    float* gpuRef = nullptr;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    allocateAndInitializeHostMemory(nxy, h_A, h_B, hostRef, gpuRef, nBytes);
    allocateDeviceMemory(nBytes, d_A, d_B, d_C);
    copyInputsToDevice(h_A, h_B, d_A, d_B, nBytes);
    
    executeGPUKernel(d_A, d_B, d_C, nx, ny);
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    Timer timerHost("SumMatrixOnHost");
    timerHost.start();
    SumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    timerHost.elapsedSeconds();
    
    checkResult(hostRef, gpuRef, nx, ny);
    freeMemory(h_A, h_B, hostRef, gpuRef, d_A, d_B, d_C);
    
    return 0;
}