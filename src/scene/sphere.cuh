#pragma once

#include "scene/object.cuh"
#include "space/vector.cuh"

namespace scene
{
class Sphere final : public Object
{
  public:
    __host__ __device__ Sphere(const space::Point3& origin,
                               const float radius,
                               const color::TextureMaterial* const texture);

    __host__ __device__ virtual cuda_tools::Optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const override;

    __host__ __device__ virtual space::Vector3
    normal_get(const space::Ray&,
               const space::IntersectionInfo& intersection) const override;

  private:
    const space::Point3 origin_;
    const float radius_;
};
} // namespace scene