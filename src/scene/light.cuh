#pragma once

namespace scene
{
class Light
{
  public:
    __host__ __device__ Light(const space::Point3& origin,
                              const float intensity)
        : origin_(origin)
        , intensity_(intensity)
    {
    }
    __host__ __device__ ~Light() {}
    __host__ __device__ float intensity_get() const { return intensity_; }
    __host__ __device__ const space::Point3& origin_get() const
    {
        return origin_;
    }

  protected:
    const space::Point3 origin_;
    const float intensity_;
};
} // namespace scene