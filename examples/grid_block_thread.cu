#include <stdio.h>

__global__ void my_kernel()
{
    const int my_block_id = blockIdx.x;
    const int my_thread_id = threadIdx.x;

    const int my_global_id = my_thread_id + my_block_id * blockDim.x;
    /*
    For 3D:
        block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y
        thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
        global_id = thread_id + block_id * blockDim.x * blockDim.y * blockDim.z
    */
    printf("Hello World from block %d and thread %d, global id %d\n", my_block_id, my_thread_id, my_global_id);
}

int main(void)
{
    dim3 grid_size(2);    // 2 blocks in the grid, same as grid_size(2,1,1)
    dim3 block_size(4);   // 4 threads per block, same as block_size(4,1,1)
    /*
    For 1D: 
        gridDim.x = grid_size, gridDim.y = 1, gridDim.z = 1
        blockDim.x = block_size, blockDim.y = 1, blockDim.z = 1
    */
    my_kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}