#pragma once

#include "scene/object.cuh"
#include "space/vector.cuh"

namespace scene
{

class Plan final : public Object
{
  public:
    __device__ Plan(const space::Point3& origin,
                    const space::Vector3& normal,
                    const color::TextureMaterial* const texture,
                    const space::Vector3& translation);

    __device__ virtual cuda_tools::Optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const override;

    __device__ virtual space::Vector3
    normal_get(const space::Ray& ray,
               const space::IntersectionInfo&) const override;

    __device__ void translate() override;

  private:
    space::Point3 origin_;
    space::Vector3 normal_;
    // Store opposite normal to avoid computing every time
    space::Vector3 opposite_normal_;
};
} // namespace scene