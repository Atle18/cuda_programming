#include <stdio.h>

__global__ void myCudaKernel()
{
    const int myblockId = blockIdx.x;
    const int mythreadId = threadIdx.x;

    const int myglobalId = mythreadId + myblockId * blockDim.x;
    /*
    For 3D:
        myblockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y
        mythreadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
        myglobalId = mythreadId + myblockId * blockDim.x * blockDim.y * blockDim.z
    */
    printf("Hello World from block %d and thread %d, global id %d\n", myblockId, mythreadId, myglobalId);
}

int main()
{
    dim3 gridSize(2);    // 2 blocks in the grid, same as gridSize(2,1,1)
    dim3 blockSize(4);   // 4 threads per block, same as blockSize(4,1,1)
    /*
    For 1D:
        gridDim.x = gridSize, gridDim.y = 1, gridDim.z = 1
        blockDim.x = blockSize, blockDim.y = 1, blockDim.z = 1
    */
    myCudaKernel<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}