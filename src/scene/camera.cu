#include "camera.cuh"

namespace scene
{
Camera::Camera(const space::Vector3& origin,
               const space::Vector3& y_axis,
               const space::Vector3& z_axis,
               const float z_min,
               const float alpha,
               const float beta)
    : origin_(origin)
    , x_axis_(cross_product(y_axis, z_axis))
    , y_axis_(y_axis)
    , z_axis_(z_axis)
    , z_min_(z_min)
    , alpha_(alpha)
    , beta_(beta)
{
}
} // namespace scene