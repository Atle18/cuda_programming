#include <stdio.h>

__global__ void helloWorldCuda()
{
    printf("Hello World CUDA!\n");
}

int main()
{
    helloWorldCuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}