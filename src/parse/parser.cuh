#pragma once

#include "scene/scene.cuh"

#include <string>

namespace parse
{
class ParseError : public std::exception
{
  public:
    ParseError(const std::string& msg, const unsigned int nb_line)
        : msg_("Line " + std::to_string(nb_line) + ": " + msg)
    {
    }

    virtual const char* what() const noexcept { return msg_.c_str(); }

  private:
    const std::string msg_;
};

/* Read a file and return the scene according to the description in the file */
scene::Scene parse_scene(const std::string& filename);
} // namespace parse