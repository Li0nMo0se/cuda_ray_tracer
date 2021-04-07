#pragma once

#include "color.cuh"

namespace color
{
class TextureMaterial
{
  public:
    TextureMaterial() = default;

    __host__ __device__ ~TextureMaterial() {}

    __host__ __device__ virtual Color3
    get_color(const space::Point3&) const = 0;
    __host__ __device__ virtual float get_kd(const space::Point3&) const = 0;
    __host__ __device__ virtual float get_ks(const space::Point3&) const = 0;
    __host__ __device__ virtual float get_ns(const space::Point3&) const = 0;
    __host__ __device__ virtual bool is_reflectable() const = 0;
};
} // namespace color