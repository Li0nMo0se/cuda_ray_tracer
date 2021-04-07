#include "scene/plan.cuh"

namespace scene
{
__host__ __device__ Plan::Plan(const space::Point3& origin,
                               const space::Vector3& normal,
                               const color::TextureMaterial* const texture)
    : Object(texture)
    , origin_(origin)
    , normal_(normal.normalized())
    , opposite_normal_(-normal_)
{
}

__host__ __device__ space::Vector3
Plan::normal_get(const space::Ray& ray, const space::IntersectionInfo&) const
{
    // if dot product is positive, the angle is lower than pi/2.
    // The normal must have an angle greater than pi/2
    // Thus, the dot product of the ray direction with the normal must
    // be negative
    if (ray.direction_get().dot(normal_) > 0.f)
        return opposite_normal_;

    return normal_;
}

__host__ __device__ cuda_tools::Optional<space::IntersectionInfo>
Plan::intersect(const space::Ray& ray) const
{
    // Let's P be the intersection point such as P = O + tD
    // Let's P0 be the origin of the plan
    // Let's n be the normal vector of the plan
    // Intersection if (P - P0).n = 0 (perpendicular)
    // (O + tD - P0).n = 0
    // t(D.n) = (P0 - O).n
    // t = ((P0 - O).n) / (D.n)

    const float denominator = ray.direction_get().dot(normal_);
    // Consider as no intersection
    if (std::abs(denominator) < epsilone)
        return cuda_tools::nullopt;

    const float numerator = (origin_ - ray.origin_get()).dot(normal_);
    const float t_res = numerator / denominator;
    if (t_res < space::T_MIN)
        return cuda_tools::nullopt;
    return space::IntersectionInfo(t_res, *this);
}
} // namespace scene