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
    
    __global__ void CopyRow(float* input, float* output, size_t rows, size_t columns)
    {
        size_t index = ( blockDim.x * blockDim.y ) * blockIdx.y + ( blockDim.x * blockDim.y ) * blockIdx.x + ( threadIdx.y * blockDim.x ) + threadIdx.x;
        if (index < rows * columns)
        {
            output[index] = input[index];
        }
    }
}
