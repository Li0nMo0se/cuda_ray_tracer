#pragma once

#include "camera.cuh"
#include "cuda_tools/vector.cuh"
#include "light.cuh"
#include "object.cuh"
#include "scene.cuh"

namespace scene
{
class Scene final
{
  public:
    using objects_t = cuda_tools::Vector<Object>;
    using lights_t = cuda_tools::Vector<Light>;

    Scene(const Camera& camera, objects_t& objects, lights_t& lights)
        : camera_(camera)
        , objects_(objects)
        , lights_(lights)
    {
    }

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    ~Scene() = default;

  private:
    Camera camera_;

    objects_t objects_;
    lights_t lights_;
};
} // namespace scene