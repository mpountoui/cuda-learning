#ifndef AUXILIARIES_CUH
#define AUXILIARIES_CUH

#include "timer.hpp"

/*------------------------------------------------------------------------------------------*/

template <typename T>
void InitialData(T* array, int size)
{
    time_t t;
    srand((unsigned) time(&t));
    
    for(int i = 0; i < size; ++i)
    {
        array[i] = static_cast<T>((rand() & 0xFF) / 10.0f);
    }
}

/*------------------------------------------------------------------------------------------*/

template <typename T>
void allocateAndInitializeHostMemory(size_t nElem, T*& h_A, T*& hostRef, T*& gpuRef)
{
    Timer timer("AllocateAndInitializeHostMemory");
    timer.start();
    
    h_A     = (T*) malloc(nElem * sizeof(T));
    hostRef = (T*) calloc(nElem, sizeof(T));
    gpuRef  = (T*) calloc(nElem, sizeof(T));
    
    InitialData(h_A, nElem);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

template <typename T>
void allocateAndInitializeHostMemory(size_t nElem, T*& h_A, T*& h_B, T*& hostRef, T*& gpuRef)
{
    Timer timer("AllocateAndInitializeHostMemory");
    timer.start();
    
    h_A     = (T*) malloc(nElem * sizeof(T));
    h_B     = (T*) malloc(nElem * sizeof(T));
    hostRef = (T*) calloc(nElem, sizeof(T));
    gpuRef  = (T*) calloc(nElem, sizeof(T));
    
    InitialData(h_A, nElem);
    InitialData(h_B, nElem);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

template <typename T>
void allocateDeviceMemory(size_t nBytes, T*& d_A, T*& d_C)
{
    Timer timer("AllocateDeviceMemory");
    timer.start();
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_C, nBytes);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

template <typename T>
void allocateDeviceMemory(size_t nBytes, T*& d_A, T*& d_B, T*& d_C)
{
    Timer timer("AllocateDeviceMemory");
    timer.start();
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaMalloc(&d_C, nBytes);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

template <typename T>
void copyInputsToDevice(T* h_A, T* h_B, T* d_A, T* d_B, size_t nBytes)
{
    Timer timer("CopyInputsToDevice");
    timer.start();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    timer.elapsedSeconds();
}

/*------------------------------------------------------------------------------------------*/

template <typename T>
void freeMemory(T* h_A, T* h_B, T* hostRef, T* gpuRef, T* d_A, T* d_B, T* d_C)
{
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#endif