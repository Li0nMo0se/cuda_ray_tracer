#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

static inline void check_error()
{
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

#ifndef cuda_safe_call
#define cuda_safe_call(call)                                                   \
    do                                                                         \
    {                                                                          \
        cudaError_t cuda_error = call;                                         \
        if (cuda_error != cudaSuccess)                                         \
        {                                                                      \
            std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_error)      \
                      << ", " << __FILE__ << ", line " << __LINE__             \
                      << std::endl;                                            \
        }                                                                      \
    } while (0)
#endif