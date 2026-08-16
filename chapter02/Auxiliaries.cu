#include "timer.hpp"

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

void freeMemory(float* h_A, float* h_B, float* hostRef, float* gpuRef, float* d_A, float* d_B, float* d_C)
{
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
