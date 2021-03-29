#pragma once

#include "../scene/camera.cuh"
#include "../space/vector.cuh"

#include <string>

namespace parse
{
class Parser
{
  public:
    Parser() = default;
    ~Parser() = default;

    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    // FIXME: return scene
    void parse_scene(const std::string& filename);

  private:
    scene::Camera parse_camera(const std::string& line);

    space::Vector3 parse_vector(const std::string& vector);
};
} // namespace parse