#include "cstdio"
#include "cstddef" 
#include "chapter01/hellocuda.h"
#include "chapter02/SumArraysOnGPU.h"
#include "chapter02/SumMatrixOnGPU.h"

int main()
{
    ArraysSum(1e8);
    
    printf("--------------------------------------------------\n");
    
    MatrixSum(1e4, 1e4);
    return 0;
}