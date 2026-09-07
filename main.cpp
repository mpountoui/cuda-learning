#include "cstdio"
#include "cstddef" 
#include "chapter01/hellocuda.h"
#include "chapter02/SumArraysOnGPU.h"
#include "chapter02/SumMatrixOnGPU.h"
#include "chapter02/ParallelReduction.h"

int main()
{
    ParallelReduction(1<<25);
    return 0;
}