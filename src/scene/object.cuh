#pragma once

#include "color/texture_material.cuh"

namespace scene
{

// Near zero value
constexpr float epsilone = 1e-6;

class Object
{
  public:
    __host__ __device__ Object(const color::TextureMaterial* const texture)
        : texture_(texture)
    {
    }

    __host__ __device__ virtual ~Object() {}
    /*
           // If no intersection return nullopt
           virtual std::optional<space::IntersectionInfo>
           intersect(const space::Ray& ray) const = 0;

           virtual space::Vector3
           normal_get(const space::Ray& ray,
                      const space::IntersectionInfo& intersection) const = 0;
     */

    __host__ __device__ const color::TextureMaterial& get_texture() const
    {
        return *texture_;
    }

  protected:
    const color::TextureMaterial* const texture_;
};
} // namespace scene