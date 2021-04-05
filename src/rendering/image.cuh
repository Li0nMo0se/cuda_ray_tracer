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
    __host__ DeviceImage(const uint32_t width, const uint32_t height);

    __host__ ~DeviceImage();

    // FIXME:
    // __host__ DeviceImage(const DeviceImage<T>& copy_device);
    // __host__ DeviceImage& operator=(const DeviceImage<T>& copy_device);

    __host__ DeviceImage(const HostImage<T>& copy_host);
    __host__ DeviceImage& operator=(const HostImage<T>& copy_host);

    __host__ void save(const std::string& filename) const override;

  private:
    __host__ void copy(const Image<T>& copy_img, cudaMemcpyKind memcpy_kind);
};

/* A host Image can be instatiated by the host (The memory will be in
 the cpu memory)
 */
template <typename T>
class HostImage final : public Image<T>
{
    __host__ HostImage(const uint32_t width, const uint32_t height);

    __host__ ~HostImage();

    // FIXME: Imagecopying

    __host__ void save(const std::string filename) const override;
};
} // namespace rendering

#include "rendering/image.cuhxx"