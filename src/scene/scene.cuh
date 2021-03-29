#pragma once

#include "camera.cuh"
#include "cuda_tools/vector.cuh"
#include "light.cuh"
#include "object.cuh"

namespace scene
{
class Scene final
{
    using objects_t = cuda_tools::Vector<Object>;
    using lights_t = cuda_tools::Vector<Light>;

  public:
    Scene(const Camera& camera, objects_t&& objects, lights_t&& lights);

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    ~Scene() = default;

  private:
    Camera camera_;

    objects_t objects_;
    lights_t lights_;
};
} // namespace scene