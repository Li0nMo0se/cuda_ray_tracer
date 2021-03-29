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

// FIXME: intersection and get normal
} // namespace scene