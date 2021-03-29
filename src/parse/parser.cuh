#pragma once

#include "scene/scene.cuh"

#include <string>

namespace parse
{
// FIXME: Easier if parser is actually a class?
// Would be able to store objects, textures, name_to_texture, nb lines and not
// having to give arguments everytime. I think it will lead to a cleaner code
// (?)
scene::Scene parse_scene(const std::string& filename);
} // namespace parse