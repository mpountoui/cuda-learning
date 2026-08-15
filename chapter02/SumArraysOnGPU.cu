#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.hpp"

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

void allocateAndInitializeHostMemory(size_t nElem, float*& h_A, float*& h_B, float*& hostRef, float*& gpuRef, size_t nBytes)
{
    Timer timer("AllocateAndInitializeHostMemory");
    timer.start();
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    hostRef = (float*) calloc(nElem, sizeof(float));
    gpuRef  = (float*) calloc(nElem, sizeof(float));
    
    InitialData(h_A, nElem);
    InitialData(h_B, nElem);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

void allocateDeviceMemory(size_t nBytes, float*& d_A, float*& d_B, float*& d_C)
{
    Timer timer("AllocateDeviceMemory");
    timer.start();
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaMalloc(&d_C, nBytes);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

void copyInputsToDevice(float* h_A, float* h_B, float* d_A, float* d_B, size_t nBytes)
{
    Timer timer("CopyInputsToDevice");
    timer.start();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

void executeGPUKernel(float* d_A, float* d_B, float* d_C, size_t nElem)
{
    dim3 block(1023);
    size_t blockThreads = block.x * block.y * block.z;
    dim3 grid((nElem + blockThreads - 1) / blockThreads);
    
    Timer timer("SumArraysOnGPU");
    timer.start();
    SumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Launch error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

void freeMemory(float* h_A, float* h_B, float* hostRef, float* gpuRef,
                float* d_A, float* d_B, float* d_C)
{
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    
    allocateAndInitializeHostMemory(nElem, h_A, h_B, hostRef, gpuRef, nBytes);
    allocateDeviceMemory(nBytes, d_A, d_B, d_C);
    copyInputsToDevice(h_A, h_B, d_A, d_B, nBytes);
    
    executeGPUKernel(d_A, d_B, d_C, nElem);
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    Timer timerHost("SumArraysOnHost");
    timerHost.start();`
    SumArraysOnHost(h_A, h_B, hostRef, nElem);
    timerHost.elapsedSeconds();
    
    checkResult(hostRef, gpuRef, nElem);
    freeMemory(h_A, h_B, hostRef, gpuRef, d_A, d_B, d_C);
    
    return 0;
}