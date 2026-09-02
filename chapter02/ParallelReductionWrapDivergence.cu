#include <stdio.h>
#include <cuda_runtime.h>
#include "Auxiliaries.cuh"

/*------------------------------------------------------------------------------------------*/

void RecursiveRuduce(int* h_data, int* h_ref, size_t nElem)
{
    if (nElem == 1)
    {
        h_ref[0] = h_data[0];
        return;
    }
    
    size_t stride = nElem / 2;
    for (size_t i = 0; i < stride; i++)
    {
        h_ref[i] = h_data[i] + h_data[i + stride];
    }
    
    RecursiveRuduce(h_ref, h_ref, stride);
}

/*------------------------------------------------------------------------------------------*/

__global__ void ReduceNeighboredOnGPU(int* d_data, int* d_ref, size_t nElem)
{
    int tid = threadIdx.x;
    int* block_ptr = d_data + blockIdx.x * blockDim.x;
    
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if( tid % (2 * stride) == 0 && tid + stride < nElem )
        {
            block_ptr[tid] += block_ptr[tid + stride];
        }
        __syncthreads();
    }
    
    if( tid == 0 )
    {
        d_ref[blockIdx.x] = block_ptr[0];
    }
}

/*------------------------------------------------------------------------------------------*/

int ParallelReductionWrapDivergence(size_t nElem)
{
    cudaSetDevice(0);
    size_t nBytes = nElem * sizeof(int);
    
    int* h_A = nullptr;
    int* hostRef = nullptr;
    int* gpuRef = nullptr;
    int* d_A = nullptr;
    int* d_C = nullptr;
    
    allocateAndInitializeHostMemory(nElem, h_A, hostRef, gpuRef);
    allocateDeviceMemory(nBytes, d_A, d_C);
    
    RecursiveRuduce(h_A, hostRef, nElem);
    
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    dim3 block(512);
    dim3 grid((nElem + block.x - 1) / block.x);
    ReduceNeighboredOnGPU<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    for(int i = 1; i < grid.x; i++)
    {
        gpuRef[0] += gpuRef[i];
    }
    
    printf("ParallelReductionWrapDivergence: Host %d, GPU %d\n", hostRef[0], gpuRef[0]);
    if( gpuRef[0] == hostRef[0] )
    {
        printf("ParallelReductionWrapDivergence: Test PASSED\n");
    }
    else
    {
        printf("ParallelReductionWrapDivergence: Test FAILED\n");
    }
    
    return 0;
}