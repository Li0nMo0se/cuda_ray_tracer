#pragma once

#include "space/vector.cuh"

namespace scene
{
class Camera final
{
  public:
    Camera(const space::Vector3& origin,
           const space::Vector3& y_axis,
           const space::Vector3& z_axis,
           const float z_min,
           const float alpha,
           const float beta);

    Camera(const Camera&) = default;
    Camera& operator=(const Camera&) = default;

    ~Camera() = default;

  public:
    // Origin of the camera `C`
    space::Vector3 origin_;

    // Three axis of the camera
    // Unit vectors
    space::Vector3 x_axis_;
    space::Vector3 y_axis_;
    space::Vector3 z_axis_;

    // Focal distance
    float z_min_;

    // alpha angle
    float alpha_;
    // beta angle
    float beta_;
};
} // namespace scene