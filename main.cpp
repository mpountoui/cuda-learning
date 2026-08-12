#include "cstddef" 
#include "chapter01/hellocuda.h"
#include "chapter02/SumArraysOnGPU.h"

int main()
{
    ArraysSum(1<<24);
    return 0;
}