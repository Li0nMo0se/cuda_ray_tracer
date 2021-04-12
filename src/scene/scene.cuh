#pragma once

#include "color/texture_material.cuh"
#include "cuda_tools/vector.cuh"
#include "scene/camera.cuh"
#include "scene/light.cuh"
#include "scene/object.cuh"

namespace rendering
{
class Engine;
class VideoEngine;
} // namespace rendering

namespace scene
{
class Scene final
{
  public:
    using objects_t = cuda_tools::Vector<Object>;
    using lights_t = cuda_tools::Vector<Light>;
    using textures_t = cuda_tools::Vector<color::TextureMaterial>;

    __host__ __device__ Scene(const Camera& camera,
                              objects_t& objects,
                              lights_t& lights,
                              textures_t& textures)
        : camera_(camera)
        , objects_(objects)
        , lights_(lights)
        , textures_(textures)
    {
    }

    inline __host__ __device__ const Camera& camera_get() const
    {
        return camera_;
    }

    inline void free()
    {
        objects_.free();
        lights_.free();
        textures_.free();
    }

    // The VideoEngine owns the scene
    friend rendering::VideoEngine;
    // The engine handle the scene but do not modify it
    friend rendering::Engine;

  private:
    // The unique camera of the scene
    Camera camera_;

    // Vector of the objects in the scene
    objects_t objects_;

    // Vector of the lights in the scene
    lights_t lights_;
    // Texture can be shared with several objects. Thus, keep track of the
    // memory in this vector
    textures_t textures_;
};
} // namespace scene