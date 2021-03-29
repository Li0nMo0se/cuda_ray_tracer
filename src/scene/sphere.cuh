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

    /*
    // Return positive t value if intersection
    virtual std::optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const override;

    virtual space::Vector3
    normal_get(const space::Ray&,
               const space::IntersectionInfo& intersection) const
    override;*/

  private:
    const space::Point3 origin_;
    const float radius_;
};
} // namespace scene