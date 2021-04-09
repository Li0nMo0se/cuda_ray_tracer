#pragma once

#include "scene/object.cuh"
#include "space/vector.cuh"
#include <cstdint>

namespace scene
{
class RayBox final : public Object
{
  public:
    __host__ __device__
    RayBox(const space::Point3& lower_bound,
           const space::Point3& higher_bound,
           const color::TextureMaterial* const texture = nullptr);

    __host__ __device__ virtual cuda_tools::Optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const override;

    __host__ __device__ virtual space::Vector3
    normal_get(const space::Ray&,
               const space::IntersectionInfo& intersection) const override;

    __host__ __device__ inline const space::Point3& lower_bound_get() const
    {
        return lower_bound_;
    }
    __host__ __device__ inline const space::Point3& higher_bound_get() const
    {
        return higher_bound_;
    }

  private:
    const space::Point3 lower_bound_;
    const space::Point3 higher_bound_;
    const space::Point3 center_;
    const space::Vector3 map_to_unit_box_;
};
} // namespace scene