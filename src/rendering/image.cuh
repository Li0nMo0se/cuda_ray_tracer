#pragma once

namespace rendering
{

template <typename T>
class Image
{
  public:
    __host__ Image(const uint32_t width, const uint32_t height);

    __host__ virtual ~Image();

    __host__ __device__ T& operator()(const uint32_t y, const uint32_t x);

    __host__ __device__ const T& operator()(const uint32_t y,
                                            const uint32_t x) const;

    __host__ __device__ inline const T* data_get() const;
    __host__ __device__ inline uint32_t width_get() const;
    __host__ __device__ inline uint32_t height_get() const;

    __host__ virtual void save(const std::string& filename) const = 0;

  protected:
    __host__ void copy(const Image<T>& copy_img, cudaMemcpyKind memcpy_kind);

  protected:
    uint32_t width_;
    uint32_t height_;
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
    __host__ DeviceImage(const uint32_t width, const uint32_t height);

    /* Copy */
    __host__ DeviceImage(const DeviceImage& copy_device);
    __host__ DeviceImage(const HostImage<T>& copy_host);
    __host__ DeviceImage& operator=(const DeviceImage& copy_device);
    __host__ DeviceImage& operator=(const HostImage<T>& copy_host);

    __host__ ~DeviceImage();

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
    __host__ HostImage(const uint32_t width, const uint32_t height);

    /* Copy */
    __host__ HostImage(const HostImage& copy_host);
    __host__ HostImage(const DeviceImage<T>& copy_device);
    __host__ HostImage& operator=(const HostImage& copy_host);
    __host__ HostImage& operator=(const DeviceImage<T>& copy_device);

    __host__ ~HostImage();

    __host__ void save(const std::string& filename) const override;
};
} // namespace rendering

#include "rendering/image.cuhxx"