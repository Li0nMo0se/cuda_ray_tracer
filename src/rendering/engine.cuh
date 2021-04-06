#pragma once

#include "scene/scene.cuh"
#include <string>

namespace rendering
{
class Engine final
{
  public:
    static void render(const std::string& filename,
                       const int32_t resolution_width,
                       const int32_t resolution_height,
                       scene::Scene& scene,
                       const int32_t aliasing_level,
                       const int32_t reflection_max_depth);

    Engine() = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(const Engine&) = delete;
};
} // namespace rendering