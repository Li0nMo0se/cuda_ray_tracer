#pragma once

#include "texture_material.cuh"

namespace color
{
class UniformTexture final : public TextureMaterial
{
  public:
    __host__ __device__ UniformTexture(const Color3 colors,
                                       const float kd,
                                       const float ks,
                                       const float ns)
        : colors_(colors)
        , kd_(kd)
        , ks_(ks)
        , ns_(ns)
    {
    }

    __host__ __device__ color::Color3
    get_color(const space::Point3&) const override
    {
        return colors_;
    }

    __host__ __device__ float get_kd(const space::Point3&) const override
    {
        return kd_;
    }

    __host__ __device__ float get_ks(const space::Point3&) const override
    {
        return ks_;
    }

    __host__ __device__ float get_ns(const space::Point3&) const override
    {
        return ns_;
    }

  private:
    color::Color3 colors_;
    float kd_;
    float ks_;
    float ns_;
};
} // namespace color