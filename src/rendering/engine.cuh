#pragma once

#include "../scene/scene.cuh"
#include <string>

namespace rendering
{
class Engine final
{
  public:
    static void render(const std::string& filename,
                       const uint32_t width,
                       const uint32_t height,
                       const scene::Scene& scene,
                       const uint32_t aliasing_level,
                       const uint32_t reflection_max_depth);

    Engine() = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(const Engine&) = delete;
};
} // namespace rendering