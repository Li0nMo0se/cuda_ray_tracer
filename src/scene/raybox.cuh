#pragma once

#include "scene/object.cuh"
#include "space/vector.cuh"
#include <cstdint>

namespace scene
{
class RayBox final : public Object
{
  public:
    __device__ RayBox(const space::Point3& lower_bound,
                      const space::Point3& higher_bound,
                      const color::TextureMaterial* const texture = nullptr);

    __device__ virtual cuda_tools::Optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const override;

    __device__ virtual space::Vector3
    normal_get(const space::Ray&,
               const space::IntersectionInfo& intersection) const override;

    __device__ void translate() override;

    __device__ inline const space::Point3& lower_bound_get() const
    {
        return lower_bound_;
    }
    __device__ inline const space::Point3& higher_bound_get() const
    {
        return higher_bound_;
    }

  private:
    __device__ space::Point3 compute_center() const;
    __device__ space::Point3 compute_map_to_unit_box() const;

  private:
    space::Point3 lower_bound_;
    space::Point3 higher_bound_;
    space::Point3 center_;
    space::Vector3 map_to_unit_box_;
};
} // namespace scene