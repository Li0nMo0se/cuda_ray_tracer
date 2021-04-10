#pragma once

#include "cuda_error.cuh"
#include <cuda_runtime.h>
namespace cuda_tools
{
// Vector to store dynamic objects and have their vtable abailable in kernels.
// Elements are stored in gpu memory.
template <typename T>
class Vector
{
  public:
    // Default with constructor with a minimum capacity
    __device__ __host__ Vector();

    // Constructor with a specified capacity
    __device__ __host__ Vector(int32_t capacity);

    // Deep destructor
    __device__ __host__ ~Vector();

    // Define copy as move constructor to avoid loose of ownership and multiple
    // delete (Can't duplicate vector)
    __device__ __host__ Vector(Vector& other);
    // Move copy constructor. Move the ownership.
    __device__ __host__ Vector(Vector&& other);

    Vector& operator=(const Vector&) = delete;

    // Upgrade the capacity of the vector
    __device__ __host__ void realloc(const int32_t new_capacity);

    // Push back in the vector by constructing the object while pushing (no copy
    // is performed)
    // SubT should be restricted with std::is_base_of. However, nvcc does not
    // support concept yet.
    template <typename SubT, typename... Ts>
    void emplace_back(Ts&&... args);

    // Get the current size of the vector
    __device__ __host__ inline int32_t size_get() const;

    // Get the nth-element
    __device__ __host__ inline const T& operator[](const int32_t pos) const;

    // Get back of the vector
    const T** back_get() const;

  private:
    // Array
    T** __restrict__ data_ = nullptr;

    // Use signed int32_tegers instead of unsigned int32_tegers because it is
    // handled more efficiently in kernels

    // Maximum capacity of the vector
    int32_t capacity_ = begin_capacity;
    // Actual size of the vector
    int32_t size_ = 0;

    // The default capacity of the vector
    static constexpr int32_t begin_capacity = 16;
};

} // namespace cuda_tools

#include "vector.cuhxx"