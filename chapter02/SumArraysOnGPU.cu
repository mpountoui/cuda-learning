#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.hpp"

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

void InitialData(float* array, int size)
{
    time_t t;
    srand((unsigned) time(&t));
    
    for(int i = 0; i < size; ++i)
    {
        array[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
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
    size_t blockArrayThreads = blockDim.x * blockDim.y;
    size_t blockThreads      = blockArrayThreads * blockDim.z;
    
    size_t gridRowThreads    = gridDim.x * blockThreads;
    size_t gridArrayThreads  = gridDim.y * gridRowThreads;
    
    size_t index = blockIdx.z * gridArrayThreads +
                   blockIdx.y * gridRowThreads +
                   blockIdx.x * blockThreads +
                   threadIdx.z * blockArrayThreads +
                   threadIdx.y * blockDim.x +
                   threadIdx.x;
    
    if (index < n){ c[index] = a[index] + b[index]; }
}

/*------------------------------------------------------------------------------------------*/

int ArraysSum(size_t nElem)
{
    int dev = 0;
    cudaSetDevice(dev);
    
    size_t nBytes = nElem * sizeof(float);
    
    float* h_A = (float*) malloc(nBytes);
    float* h_B = (float*) malloc(nBytes);
    float* hostRef = (float*) calloc(nElem, sizeof(float));
    float* gpuRef  = (float*) calloc(nElem, sizeof(float));
    
    InitialData(h_A, nElem);
    InitialData(h_B, nElem);
    
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc( &d_A, nBytes );
    cudaMalloc( &d_B, nBytes );
    cudaMalloc( &d_C, nBytes );
    
    cudaMemcpy( d_A, h_A, nBytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, nBytes, cudaMemcpyHostToDevice );
    
    dim3 block(256);
    size_t blockThreads = block.x * block.y * block.z;
    dim3 grid( (nElem + blockThreads - 1) / blockThreads );
    
    Timer timer("SumArraysOnGPU");
    timer.start();
    SumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    timer.elapsedSeconds();
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    Timer timerHost("SumArraysOnHost");
    timerHost.start();
    SumArraysOnHost(h_A, h_B, hostRef, nElem);
    timerHost.elapsedSeconds();
    
    checkResult(hostRef, gpuRef, nElem);
    
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}