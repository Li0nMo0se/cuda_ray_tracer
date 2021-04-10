#pragma once

#include "scene/object.cuh"
#include "space/vector.cuh"

namespace scene
{
class Triangle final : public Object
{
  public:
    __device__ Triangle(const space::Point3& A,
                        const space::Point3& B,
                        const space::Point3& C,
                        const color::TextureMaterial* const texture);

    __device__ virtual cuda_tools::Optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const override;

    __device__ virtual space::Vector3
    normal_get(const space::Ray& ray,
               const space::IntersectionInfo&) const override;

  private:
    __device__ space::Vector3 compute_normal() const;

  private:
    const space::Point3 A_;
    const space::Point3 B_;
    const space::Point3 C_;
    const space::Vector3 normal_;
    const space::Vector3 opposite_normal_;
};
} // namespace scene