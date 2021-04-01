#pragma once

#include "cuda_tools/optional.cuh"
#include "space/ray.cuh"
#include "space/vector.cuh"

namespace scene
{
class Object;
}

namespace space
{
class IntersectionInfo
{
  public:
    __host__ __device__ IntersectionInfo()
        : t_(cuda_tools::nullopt)
        , obj_(cuda_tools::nullopt)
        , intersection_(cuda_tools::nullopt)
    {
    }

    __host__ __device__ IntersectionInfo(const float t,
                                         const scene::Object& obj)
        : t_(t)
        , obj_(&obj)
    {
    }

    __host__ __device__ void compute_intersection(const Ray& ray)
    {
        assert(t_.has_value());
        intersection_ = ray.origin_get() + t_.value() * ray.direction_get();
    }

    // TODO instead of going to the normal, go back until no intersection with
    // yourself
    __host__ __device__ void auto_intersection_correction(const Vector3& normal)
    {
        assert(intersection_.has_value());
        intersection_ =
            intersection_.value() + normal * intersection_correction_ratio;
    }

    /* Getters */
    __host__ __device__ float t_get() const
    {
        assert(t_.has_value());
        return t_.value();
    }

    __host__ __device__ const scene::Object& obj_get() const
    {
        assert(obj_.has_value());
        return *(obj_.value());
    }

    __host__ __device__ const Point3& intersection_get() const
    {
        assert(intersection_.has_value());
        return intersection_.value();
    }

  private:
    cuda_tools::Optional<float> t_ = cuda_tools::nullopt;
    cuda_tools::Optional<const scene::Object*> obj_ = cuda_tools::nullopt;
    cuda_tools::Optional<Point3> intersection_ = cuda_tools::nullopt;

    // FIXME: Instead of using the ratio, move until no intersection with
    // yourself
    // FIXME: static constexpr works with gpu memory?
    const float intersection_correction_ratio = 0.2f;
};
} // namespace space