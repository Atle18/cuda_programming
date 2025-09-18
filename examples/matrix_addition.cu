#include <stdio.h>
#include <random>

__device__ float add(const float x, const float y)
{
    return x + y;
}

__global__ void matrixAdd(float *A, float *B, float*C, const int nx, const int ny)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int id = ix + iy * nx;

    if (ix < nx && iy < ny)
    {
        C[id] = add(A[id], B[id]);
    }
}

void initialData(float *addr, int elementCount)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    for (int i = 0; i < elementCount; i++)
    {
        addr[i] = dist(gen);
    }
}

int main()
{
    // Get GPU devices count, and set GPU device
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0)
    {
        printf("No GPU device found!\n");
        exit(-1);
    }
    else
    {
        printf("The number of GPU devices is %d.\n", deviceCount);
    }

    int rank = 0;
    error = cudaSetDevice(rank);

    if (error != cudaSuccess)
    {
        printf("Fail to set GPU device %d.\n", rank);
        exit(-1);
    }
    else
    {
        printf("Use GPU device %d.\n", rank);
    }

    // Allocate memory for host and device, and initilize
    int nx = 16;
    int ny = 8;
    int elementCount = nx * ny;
    size_t bytesCount = elementCount * sizeof(float);

    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(bytesCount);
    fpHost_B = (float *)malloc(bytesCount);
    fpHost_C = (float *)malloc(bytesCount);

    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, bytesCount);
        memset(fpHost_B, 0, bytesCount);
        memset(fpHost_C, 0, bytesCount);
    }
    else
    {
        printf("Fail to allocate host memory!\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc(&fpDevice_A, bytesCount);
    cudaMalloc(&fpDevice_B, bytesCount);
    cudaMalloc(&fpDevice_C, bytesCount);

    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        cudaMemset(fpDevice_A, 0, bytesCount);
        cudaMemset(fpDevice_B, 0, bytesCount);
        cudaMemset(fpDevice_C, 0, bytesCount);
    }
    else
    {
        printf("Fail to allocate device memory!\n");
        free(fpDevice_A);
        free(fpDevice_B);
        free(fpDevice_C);
        exit(-1);
    }

    // Set values to host memory
    srand(666);
    initialData(fpHost_A, elementCount);
    initialData(fpHost_B, elementCount);

    // Copy data from host to device
    cudaMemcpy(fpDevice_A, fpHost_A, bytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, bytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, bytesCount, cudaMemcpyHostToDevice);

    // Launch kernel function to do matrix addition on device
    dim3 blockSize(4, 4);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    matrixAdd<<<gridSize, blockSize>>>(fpDevice_A, fpDevice_B, fpDevice_C, nx, ny);

    // Copy data from device to host
    // cudaDeviceSynchronize(); // This is not necessary because cudaMemcpy has implicate sync
    cudaMemcpy(fpHost_C, fpDevice_C, bytesCount, cudaMemcpyDeviceToHost);

    // Check the results
    for (int i = 0; i < 10; i++)
    {
        printf("idx=%d, A[idx]=%.1f, B[idx]=%.1f, C[idx]=%.1f\n", i, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    }

    // Free host and device memory
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}