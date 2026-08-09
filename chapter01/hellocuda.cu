#include <stdio.h>

__global__ void HelloFromGPU()
{
    printf("Hello World From GPU\n");
}