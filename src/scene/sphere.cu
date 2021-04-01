#include "sphere.cuh"
#include <cmath>

namespace scene
{
Sphere::Sphere(const space::Point3& origin,
               const float radius,
               const color::TextureMaterial* const texture)
    : Object(texture)
    , origin_(origin)
    , radius_(radius)
{
}

cuda_tools::Optional<space::IntersectionInfo>
Sphere::intersect(const space::Ray& ray) const
{
    // TODO
    return cuda_tools::nullopt;
}

space::Vector3
Sphere::normal_get(const space::Ray&,
                   const space::IntersectionInfo& intersection) const
{
    // TODO
}

// FIXME: intersection and get normal
} // namespace scene