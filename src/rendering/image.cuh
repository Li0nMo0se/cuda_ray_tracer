#pragma once

#include "cuda_tools/cuda_error.cuh"
#include <cstdint>
#include <string>
namespace rendering
{
class ISavable
{
  public:
    __host__ virtual void save(const std::string& filename) const = 0;
};

template <typename T>
class Image : public ISavable
{
  public:
    __host__ __device__ Image(const int32_t width, const int32_t height);

    __host__ virtual ~Image();

    __host__ __device__ inline const T* data_get() const;
    __host__ __device__ inline T* data_get();
    __host__ __device__ inline int32_t width_get() const;
    __host__ __device__ inline int32_t height_get() const;

    __host__ virtual void free() = 0;

  protected:
    __host__ void _copy(Image<T>& dst, cudaMemcpyKind memcpy_kind) const;
    // FIXME const

  protected:
    int32_t width_;
    int32_t height_;
    T* data_ = nullptr;
};

template <typename T>
class DeviceImage;
template <typename T>
class HostImage;

/* A device Image can be instatiated by the host (the memory will
 be in the gpu memory)
 */
template <typename T>
class DeviceImage final : public Image<T>
{
  public:
    // Allocate uninitialize memory
    __host__ __device__ DeviceImage(const int32_t width, const int32_t height);

    /* Deep Copy */
    __host__ void copy(HostImage<T>& copy_host) const;

    __host__ void free() override;

    __host__ void save(const std::string& filename) const override;
};

/* A host Image can be instatiated by the host (The memory will be in
 the cpu memory)
 */
template <typename T>
class HostImage final : public Image<T>
{
  public:
    // Allocate uninitialize memory
    __host__ HostImage(const int32_t width, const int32_t height);

    /* Deep Copy */
    __host__ void copy(DeviceImage<T>& copy_device) const;

    __host__ void free() override;

    __host__ inline void save(const std::string& filename) const override;
};

template <typename T>
struct ImageHandler final : public ISavable
{
  public:
    ImageHandler(const int32_t width, const int32_t height);
    ~ImageHandler();

    __host__ void save(const std::string& filename) const override;

    // Copy from host to device
    __host__ void copy_host_to_device();
    // Copy from device to host
    __host__ void copy_device_to_host();

    HostImage<T> host;
    DeviceImage<T> device;
};
} // namespace rendering

#include "rendering/image.cuhxx"