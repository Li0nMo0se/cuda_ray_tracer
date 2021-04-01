#pragma once

#include "scene/light.cuh"
#include "space/vector.cuh"

namespace scene
{
class PointLight final : public Light
{
  public:
    __host__ __device__ PointLight(const space::Point3& origin,
                                   const float intensity)
        : Light(origin, intensity)
    {
    }
};
} // namespace scene