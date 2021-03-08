#include <cuda_runtime.h>
#include <stdio.h>

__global__
void print_kernel()
{
    printf("Hello from kernel !\n");
}

void print()
{
    print_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Failure\n");
        fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(err));
    }
    else
        printf("Success\n");
}

