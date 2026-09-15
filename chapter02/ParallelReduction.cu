#include <stdio.h>
#include <cuda_runtime.h>
#include "Auxiliaries.cuh"

namespace
{
/*------------------------------------------------------------------------------------------*/
    
    using ReductionKernel = void (*)(int*, int*, size_t);
    
/*------------------------------------------------------------------------------------------*/
    
    struct ReductionBuffers
    {
        int* data   = nullptr;
        int* result = nullptr;
    };
    
/*------------------------------------------------------------------------------------------*/
    
    struct ReductionRun
    {
        ReductionBuffers buffers;
        long long elapsedMicroseconds = 0;
    };
    
/*------------------------------------------------------------------------------------------*/
    
    ReductionRun RunReduction(int* input, size_t nElem, dim3 grid, dim3 block, ReductionKernel kernel, const char* timerName)
    {
        ReductionBuffers buffers{nullptr, nullptr};
        size_t nBytes = nElem * sizeof(int);
        cudaMalloc(&buffers.data, nBytes);
        cudaMalloc(&buffers.result, nBytes);
        cudaMemcpy(buffers.data, input, nBytes, cudaMemcpyHostToDevice);
        
        Timer timer(timerName);
        timer.start();
        kernel<<<grid, block>>>(buffers.data, buffers.result, nElem);
        cudaDeviceSynchronize();
        timer.stop();
        
        return {buffers, timer.elapsedMicroseconds()};
    }
    
/*------------------------------------------------------------------------------------------*/
    
    int CollectResult(ReductionRun run, int* result, dim3 grid, size_t nElem)
    {
        cudaMemcpy(result, run.buffers.result, nElem * sizeof(int), cudaMemcpyDeviceToHost);
        for (unsigned int i = 1; i < grid.x; ++i)
        {
            result[0] += result[i];
        }
        cudaFree(run.buffers.data);
        cudaFree(run.buffers.result);
        return result[0];
    }
    
/*------------------------------------------------------------------------------------------*/
    
    void PrintResult(const char* name, int hostResult, int gpuResult, long long elapsedMicroseconds)
    {
        const char* status = gpuResult == hostResult ? "PASSED" : "FAILED";
        printf("  %-28s %8lld μs | Host: %d | GPU: %d | %s\n", name, elapsedMicroseconds, hostResult, gpuResult, status);
    }
    
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
            if( tid % (2 * stride) == 0 && global_tid + stride < nElem )
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
        
        for(int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int index = 2 * stride * tid;
            if( index < blockDim.x && blockIdx.x * blockDim.x + index + stride < nElem )
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
            if( tid < stride && global_tid + stride < nElem )
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
    
    template <unsigned int UnrollFactor>
    __device__ void ReduceUnrolling(int* d_data, int* d_ref, size_t nElem)
    {
        int tid = threadIdx.x;
        int* block_ptr = d_data + UnrollFactor * blockIdx.x * blockDim.x;
        size_t global_tid = UnrollFactor * blockIdx.x * blockDim.x + threadIdx.x;
        
        if( global_tid < nElem )
        {
            for( unsigned int factor = 1; factor < UnrollFactor; ++factor )
            {
                size_t index = global_tid + factor * blockDim.x;
                if( index < nElem )
                {
                    block_ptr[tid] += d_data[index];
                }
            }
        }
        __syncthreads();
        
        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if( tid < stride )
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
    
    __global__ void ReduceUnrolling2(int* d_data, int* d_ref, size_t nElem)
    {
        ReduceUnrolling<2>(d_data, d_ref, nElem);
    }
    
/*------------------------------------------------------------------------------------------*/
    
    __global__ void ReduceUnrolling4(int* d_data, int* d_ref, size_t nElem)
    {
        ReduceUnrolling<4>(d_data, d_ref, nElem);
    }
    
/*------------------------------------------------------------------------------------------*/
    
    void RunAndReport(int* input, int hostResult, size_t nElem, dim3 grid, dim3 block)
    {
        ReductionRun divergent     = RunReduction(input, nElem, grid, block, ReduceNeighboredOnGPU    , "ReduceNeighboredOnGPU"    );
        ReductionRun interleaved   = RunReduction(input, nElem, grid, block, ReduceInterleavedOnGPU   , "ReduceInterleavedOnGPU"   );
        ReductionRun lessDivergent = RunReduction(input, nElem, grid, block, ReduceNeighboredLessOnGPU, "ReduceNeighboredLessOnGPU");
        dim3 unrolling2Grid((nElem + 2 * block.x - 1) / (2 * block.x));
        ReductionRun unrolling2 = RunReduction(input, nElem, unrolling2Grid, block, ReduceUnrolling2, "ReduceUnrolling2");
        dim3 unrolling4Grid((nElem + 4 * block.x - 1) / (4 * block.x));
        ReductionRun unrolling4 = RunReduction(input, nElem, unrolling4Grid, block, ReduceUnrolling4, "ReduceUnrolling4");
        
        int* gpuResult = static_cast<int*>(calloc(nElem, sizeof(int)));
        printf("\nReduction results (blocks: %u, threads/block: %u):\n", grid.x, block.x);
        PrintResult("Neighbor divergence"   , hostResult, CollectResult(divergent    , gpuResult, grid          , nElem), divergent.elapsedMicroseconds    );
        PrintResult("Reduced divergence"    , hostResult, CollectResult(lessDivergent, gpuResult, grid          , nElem), lessDivergent.elapsedMicroseconds);
        PrintResult("Interleaved addressing", hostResult, CollectResult(interleaved  , gpuResult, grid          , nElem), interleaved.elapsedMicroseconds  );
        PrintResult("Unrolling x2"          , hostResult, CollectResult(unrolling2   , gpuResult, unrolling2Grid, nElem), unrolling2.elapsedMicroseconds   );
        PrintResult("Unrolling x4"          , hostResult, CollectResult(unrolling4   , gpuResult, unrolling4Grid, nElem), unrolling4.elapsedMicroseconds   );
        free(gpuResult);
    }
}

/*------------------------------------------------------------------------------------------*/

int ParallelReduction(size_t nElem)
{
    cudaSetDevice(0);
    int* values     = nullptr;
    int* hostResult = nullptr;
    allocateAndInitializeHostMemory(nElem, values, hostResult);
    
    Timer timer("Host recursive reduction");
    timer.start();
    RecursiveRuduce(values, hostResult, nElem);
    timer.stop();
    long long hostMicroseconds = timer.elapsedMicroseconds();
    
    dim3 block(512);
    dim3 grid((nElem + block.x - 1) / block.x);
    
    printf("\nParallel reduction report (elements: %zu):\n", nElem);
    printf("  %-28s %8lld μs | Result: %d\n", "Host recursive reduction", hostMicroseconds, hostResult[0]);
    RunAndReport(values, hostResult[0], nElem, grid, block);
    free(hostResult);
    free(values);
    return 0;
}