#pragma once

#include "cuda_tools/optional.cuh"
#include "space/vector.cuh"

namespace space
{

// Minimum value for an intersection
constexpr float T_MIN = 1e-5;

class Ray
{
  public:
    __host__ __device__ Ray(const Point3& origin, const Vector3& direction)
        : origin_(origin)
        , direction_(direction)
    {
    }

    __host__ __device__ const Point3& origin_get() const { return origin_; }
    __host__ __device__ const Vector3& direction_get() const
    {
        return direction_;
    };

  private:
    const Point3 origin_;
    const Vector3 direction_;
};
} // namespace space