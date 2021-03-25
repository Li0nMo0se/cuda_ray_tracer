#pragma once

#include "cuda_error.cuh"
#include <cuda_runtime.h>

namespace cuda_tools
{
template <typename T>
class Vector
{
  public:
    // Default with constructor with a minimum capacity
    Vector();

    // Constructor with a specified capacity
    Vector(int32_t capacity);

    // Deep destructor
    ~Vector();

    // Upgrade the capacity of the vector
    void realloc(const int32_t new_capacity);

    // Push back in the vector by constructing the object while pushing (no copy
    // is performed)
    template <typename... Ts>
    void emplace_back(Ts&&... args);

    // Get the current size of the vector
    inline int32_t size_get() const;

  private:
    // Array
    T** data_ = nullptr;

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