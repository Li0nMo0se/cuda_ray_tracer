#include "scene/raybox.cuh"

namespace scene
{

__device__ RayBox::RayBox(const space::Point3& lower_bound,
                          const space::Point3& higher_bound,
                          const color::TextureMaterial* const texture,
                          const space::Vector3& translation)
    : Object(texture, translation)
    , lower_bound_(lower_bound)
    , higher_bound_(higher_bound)
    , center_(compute_center())
    , map_to_unit_box_(compute_map_to_unit_box())

{
    assert(lower_bound_[0] <= higher_bound_[0]);
    assert(lower_bound_[1] <= higher_bound_[1]);
    assert(lower_bound_[2] <= higher_bound_[2]);
}

__device__ space::Point3 RayBox::compute_center() const
{
    return (lower_bound_ + higher_bound_) / 2;
}

__device__ space::Point3 RayBox::compute_map_to_unit_box() const
{
    return (higher_bound_ - lower_bound_) / 2;
}

__device__ void RayBox::translate()
{
    lower_bound_ += translation_;
    higher_bound_ += translation_;
    center_ = compute_center();
    map_to_unit_box_ = compute_map_to_unit_box();
}

template <typename T>
static __device__ void swap(T& a, T& b)
{
    T c(a);
    a = b;
    b = c;
}

__device__ cuda_tools::Optional<space::IntersectionInfo>
RayBox::intersect(const space::Ray& ray) const
{
    // The algorithm comes from scratchpixel
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    const space::Point3& ray_origin = ray.origin_get();
    const space::Vector3& ray_direction = ray.direction_get();

    assert(ray_direction[0] != 0.f);
    // Compute t_min and t_max on x axis
    float t_min = (lower_bound_[0] - ray_origin[0]) / ray_direction[0];
    float t_max = (higher_bound_[0] - ray_origin[0]) / ray_direction[0];
    // t_min might be greater than t_max according to the ray direction
    // We must have t_min < t_max for the tests above
    if (t_max < t_min)
        swap(t_max, t_min);

    // Compute t_min and t_max on y axis
    {
        assert(ray_direction[1] != 0.f);
        float t_min_y = (lower_bound_[1] - ray_origin[1]) / ray_direction[1];
        float t_max_y = (higher_bound_[1] - ray_origin[1]) / ray_direction[1];
        if (t_max_y < t_min_y)
            swap(t_max_y, t_min_y);

        if (t_min > t_max_y || t_min_y > t_max) // No intersection
            return cuda_tools::nullopt;

        // Update t according to the x,y plans
        if (t_min_y > t_min)
            t_min = t_min_y;
        if (t_max_y < t_max)
            t_max = t_max_y;
    }

    // Compute t_min and t_max on z axis
    {
        assert(ray_direction[2] != 0.f);
        float t_min_z = (lower_bound_[2] - ray_origin[2]) / ray_direction[2];
        float t_max_z = (higher_bound_[2] - ray_origin[2]) / ray_direction[2];
        if (t_max_z < t_min_z)
            swap(t_max_z, t_min_z);

        if (t_min > t_max_z || t_max < t_min_z)
            return cuda_tools::nullopt;

        if (t_min_z > t_min)
            t_min = t_min_z;
        if (t_max_z < t_max)
            t_max = t_max_z;
    }

    // Intersection found
    // Can t_min be greater than t_max here ?
    assert(t_min <= t_max);
    float t_res = t_min;
    if (t_res < space::T_MIN)
    {
        t_res = t_max;
        if (t_res < space::T_MIN)
            return cuda_tools::nullopt;
    }

    return space::IntersectionInfo(t_res, *this);
}

__device__ space::Vector3
RayBox::normal_get(const space::Ray&,
                   const space::IntersectionInfo& intersection) const
{
    // Compute vector from center to the intersection
    // center_ = (lower + higher) / 2
    space::Vector3 center_to_intersection =
        intersection.intersection_get() - center_;

    // Map this vector in the unit raybox
    // map_to_unit_box_ = (higher - lower) / 2
    center_to_intersection = center_to_intersection / map_to_unit_box_;
    // FIXME: abs on the vector
    // Find the maximum absolute coordinate
    uint8_t coord_max = 0;
    float max = abs(center_to_intersection[0]);
    for (uint8_t i = 1; i < 3; i++)
    {
        const float value = abs(center_to_intersection[i]);
        if (value > max)
        {
            coord_max = i;
            max = value;
        }
    }

    // FIXME: store normal in an array of normals
    space::Vector3 normal(0.f, 0.f, 0.f);
    normal[coord_max] = 1;
    if (center_to_intersection[coord_max] < 0.f)
        normal[coord_max] *= -1;

    return normal;
}

} // namespace scene