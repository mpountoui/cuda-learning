#include "MatrixTranspose.h"

namespace
{
/*------------------------------------------------------------------------------------------*/
    
    int MatrixTranspose(float* input, float* output, size_t rows, size_t columns)
    {
        if (input == nullptr || output == nullptr)
        {
            return -1;
        }
        
        for (size_t row = 0; row < rows; ++row)
        {
            for (size_t column = 0; column < columns; ++column)
            {
                output[column * rows + row] = input[row * columns + column];
            }
        }
        
        return 0;
    }
    
/*------------------------------------------------------------------------------------------*/
    
    __global__ void CopyRow(float* input, float* output, size_t nx, size_t ny)
    {
        size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
        
        if( ix < nx && iy < ny )
        {
            output[iy * nx + ix] = input[iy * nx + ix];
        }
    }
    
/*------------------------------------------------------------------------------------------*/
    
    __global__ void CopyColumn(float* input, float* output, size_t nx, size_t ny)
    {
        size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
        
        if( ix < ny && iy < nx )
        {
            output[ix * ny + iy] = input[ix * ny + iy];
        }
    }
}
