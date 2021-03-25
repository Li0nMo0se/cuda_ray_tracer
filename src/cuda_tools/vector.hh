#pragma once

#include "cuda_error.hh"

namespace cuda_tools
{
namespace
{
template <typename T>
__global__ void kernel_dealloc(T** const data, const int32_t size)
{
    for (int32_t i = 0; i < size; i++)
        delete data[i];
}

template <typename T, typename... Ts>
__global__ void kernel_add_object(T** const data, Ts... args)
{
    // Data is already at the location in which the new element is stored
    *data = new T{args...};
}
} // namespace

template <typename T>
class Vector
{
  public:
    Vector()
        : Vector(begin_capacity)
    {
    }

    Vector(int32_t size)
        : capacity_(size)
    {
        realloc(size);
    }

    ~Vector()
    {
        kernel_dealloc<<<1, 1>>>(data_, size_);
        cuda_safe_call(cudaFree(data_));
    }

    void realloc(const int32_t new_capacity)
    {
        if (!data_)
            cuda_safe_call(
                cudaMalloc((void**)&data_, sizeof(T*) * new_capacity));
        else
        {
            T** tmp;
            cuda_safe_call(cudaMalloc((void**)&tmp, sizeof(T*) * new_capacity));
            cuda_safe_call(cudaMemcpy(tmp,
                                      data_,
                                      sizeof(T*) * size_,
                                      cudaMemcpyDeviceToDevice));
            cuda_safe_call(cudaFree(data_));
            data_ = tmp;
        }
        capacity_ = new_capacity;
    }

    template <typename... Ts>
    void emplace_back(Ts&&... args)
    {
        kernel_add_object<<<1, 1>>>(&(data_[size_]), std::forward<Ts>(args)...);
        ++size_;

        // Upgrade the capacity
        if (size_ == capacity_)
            realloc(capacity_ * 2);
    }

    int32_t size_get() const { return size_; }

  private:
    // Array
    T** data_ = nullptr;

    // Use signed int32_tegers instead of unsigned int32_tegers because it is
    // handled more efficiently in kernels

    // Maximum capacity of the vector
    int32_t capacity_ = begin_capacity;
    // Actual size of the vector
    int32_t size_ = 0;

    static constexpr int32_t begin_capacity = 16;
};
} // namespace cuda_tools