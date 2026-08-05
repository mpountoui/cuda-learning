#include <stdio.h>

__global__ void HelloFromGPU()
{
    printf("Hello World From GPU");
}

int main()
{
    HelloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}