#include <stdio.h>

__global__ void hello_world_cuda()
{
    printf("Hello World CUDA!\n");
}

int main(void)
{
    hello_world_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    return 0;
}