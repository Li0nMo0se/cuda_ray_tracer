#pragma once

#include "color/texture_material.cuh"
#include "cuda_tools/optional.cuh"
#include "space/intersection_info.cuh"
#include "space/ray.cuh"
#include "space/vector.cuh"

namespace scene
{

class Translatable
{
  public:
    __device__ Translatable(const space::Vector3& translation)
        : translation_(translation)
    {
    }

    __device__ virtual void translate() = 0;

  protected:
    space::Vector3 translation_;
};

// Near zero value
constexpr float epsilone = 1e-6;

class Object : public Translatable
{
  public:
    __device__ Object(const color::TextureMaterial* const texture,
                      const space::Vector3& translation)
        : Translatable(translation)
        , texture_(texture)
    {
    }

    __device__ virtual ~Object() {}

    // If no intersection return nullopt
    __device__ virtual cuda_tools::Optional<space::IntersectionInfo>
    intersect(const space::Ray& ray) const = 0;

    __device__ virtual space::Vector3
    normal_get(const space::Ray& ray,
               const space::IntersectionInfo& intersection) const = 0;

    __device__ const color::TextureMaterial& get_texture() const
    {
        return *texture_;
    }

  protected:
    const color::TextureMaterial* const texture_;
};
} // namespace scene