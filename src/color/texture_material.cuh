#pragma once

#include "color.cuh"

namespace color
{
class TextureMaterial
{
  public:
    TextureMaterial() = default;
    virtual ~TextureMaterial() = default;

    TextureMaterial(const TextureMaterial&) = default;
    TextureMaterial& operator=(const TextureMaterial&) = default;

    virtual Color3 get_color(const space::Point3&) const = 0;
    virtual float get_kd(const space::Point3&) const = 0;
    virtual float get_ks(const space::Point3&) const = 0;
    virtual float get_ns(const space::Point3&) const = 0;
};
} // namespace color