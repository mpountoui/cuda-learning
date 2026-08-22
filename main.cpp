#include "cstdio"
#include "cstddef" 
#include "chapter01/hellocuda.h"
#include "chapter02/SumArraysOnGPU.h"
#include "chapter02/SumMatrixOnGPU.h"
#include "chapter02/ParallelReductionWrapDivergence.h"

int main()
{
    ParallelReductionWrapDivergence(1024 * 1024);
    return 0;
}