#pragma once

#include "../scene/scene.cuh"

#include <string>

namespace parse
{
scene::Scene parse_scene(const std::string& filename);
} // namespace parse