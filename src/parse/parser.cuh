#pragma once

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
    // FIXME: return space::Vector3
    void parse_vector(const std::string& vector);

  private:
    int32_t nb_line_ = 0;
};
} // namespace parse