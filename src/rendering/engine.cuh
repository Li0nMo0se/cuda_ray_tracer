#pragma once

#include "color/color.cuh"
#include "rendering/image.cuh"
#include "scene/scene.cuh"
#include <string>

namespace rendering
{
class Engine final
{
  public:
    /* Render the scene into an image.
     * The engine do not modify the scene, but it is destroyed at the end of the
     * rendering
     */
    static void render(const std::string& filename,
                       const int32_t resolution_width,
                       const int32_t resolution_height,
                       scene::Scene& scene,
                       const int32_t aliasing_level,
                       const int32_t reflection_max_depth);

    Engine() = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(const Engine&) = delete;

    // FIXME: The kernel needs it to be public to call it. But this method
    // should be private

    static __device__ color::Color3
    get_pixel_color(const space::Point3& curr_pixel,
                    const scene::Scene& scene,
                    const int32_t unit_x,
                    const int32_t unit_y,
                    const int32_t aliasing_level,
                    const int32_t reflection_max_depth);

  private:
    static __device__ cuda_tools::Optional<space::IntersectionInfo>
    cast_ray(const space::Ray& ray, const scene::Scene& scene);

    static __device__ bool check_shadow(const scene::Scene& scene,
                                        const scene::Light& light,
                                        const space::Point3& intersection);

    static __device__ inline float distance_attenuation(const float distance);

    static __device__ color::Color3
    cast_ray_color(space::Ray ray,
                   const scene::Scene& scene,
                   const int32_t reflection_max_depth);
};
} // namespace rendering