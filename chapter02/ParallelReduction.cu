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
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if( tid % (2 * stride) == 0 && global_tid < nElem )
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

__global__ void ReduceNeighboredLessOnGPU(int* d_data, int* d_ref, size_t nElem)
{
    int tid = threadIdx.x;
    int* block_ptr = d_data + blockIdx.x * blockDim.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if( index < blockDim.x && global_tid < nElem )
        {
            block_ptr[index] += block_ptr[index + stride];
        }
        __syncthreads();
    }
    
    if( tid == 0 )
    {
        d_ref[blockIdx.x] = block_ptr[0];
    }
}

/*------------------------------------------------------------------------------------------*/

__global__ void ReduceInterleavedOnGPU(int* d_data, int* d_ref, size_t nElem)
{
    int tid = threadIdx.x;
    int* block_ptr = d_data + blockIdx.x * blockDim.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if( tid < stride && global_tid < nElem )
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

int ParallelReduction(size_t nElem)
{
    cudaSetDevice(0);
    size_t nBytes = nElem * sizeof(int);
    
    int* h_A = nullptr;
    int* hostRef = nullptr;
    int* gpuRef_A = nullptr;
    int* gpuRef_B = nullptr;
    int* gpuRef_C = nullptr;
    int* d_A = nullptr;
    int* d_C = nullptr;
    int* d_B = nullptr;
    int* d_D = nullptr;
    int* d_E = nullptr;
    int* d_F = nullptr;
    
    allocateAndInitializeHostMemory(nElem, h_A, hostRef);
    gpuRef_A  = (int*) calloc(nElem, sizeof(int));
    gpuRef_B  = (int*) calloc(nElem, sizeof(int));
    gpuRef_C  = (int*) calloc(nElem, sizeof(int));
    
    allocateDeviceMemory(nBytes, d_A, d_C);
    allocateDeviceMemory(nBytes, d_B, d_D);
    allocateDeviceMemory(nBytes, d_E, d_F);
    
    Timer timer("RecursiveRuduce");
    timer.start();
    RecursiveRuduce(h_A, hostRef, nElem);
    timer.stop();
    
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, h_A, nBytes, cudaMemcpyHostToDevice);
    
    dim3 block(512);
    dim3 grid((nElem + block.x - 1) / block.x);
    
    Timer timerGPU("ReduceNeighboredOnGPU");
    timerGPU.start();
    ReduceNeighboredOnGPU<<<grid, block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    timerGPU.stop();
    
    Timer timerGPU2("ReduceInterleavedOnGPU");
    timerGPU2.start();
    ReduceInterleavedOnGPU<<<grid, block>>>(d_B, d_D, nElem);
    cudaDeviceSynchronize();
    timerGPU2.stop();
    
    Timer timerGPU3("ReduceNeighboredLessOnGPU");
    timerGPU3.start();
    ReduceNeighboredLessOnGPU<<<grid, block>>>(d_E, d_F, nElem);
    cudaDeviceSynchronize();
    timerGPU3.stop();
    
    cudaMemcpy(gpuRef_A, d_C, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuRef_B, d_D, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpuRef_C, d_F, nBytes, cudaMemcpyDeviceToHost);
    
    for(int i = 1; i < grid.x; i++)
    {
        gpuRef_A[0] += gpuRef_A[i];
        gpuRef_B[0] += gpuRef_B[i];
        gpuRef_C[0] += gpuRef_C[i];
    }
    
    printf("ParallelReductionWrapDivergence: Host %d, GPU_A %d, GPU_B %d, GPU_C %d\n", hostRef[0], gpuRef_A[0], gpuRef_B[0], gpuRef_C[0]);
    gpuRef_A[0] == hostRef[0] ? printf("ParallelReductionWrapDivergence: Test PASSED\n"):
                                printf("ParallelReductionWrapDivergence: Test FAILED\n");
    
    gpuRef_B[0] == hostRef[0] ? printf("ParallelReductionWrapDivergence: Test PASSED\n"):
                                printf("ParallelReductionWrapDivergence: Test FAILED\n");
    
    gpuRef_C[0] == hostRef[0] ? printf("ParallelReductionWrapDivergence: Test PASSED\n"):
                                printf("ParallelReductionWrapDivergence: Test FAILED\n");
    
    return 0;
}