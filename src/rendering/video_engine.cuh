#pragma once

#include <string>

namespace rendering
{

class VideoEngine final
{
  public:
    static void render(const std::string& input_path,
                       const std::string& output_path,
                       const uint32_t resolution_width,
                       const uint32_t resolution_height,
                       const uint32_t nb_frames,
                       const uint32_t aliasing_level,
                       const uint32_t reflection_max_depth);

    VideoEngine() = delete;
    VideoEngine& operator=(const VideoEngine&) = delete;
    VideoEngine(const VideoEngine&) = delete;

  private:
    static std::string get_output_filename(const std::string& output_path,
                                           const uint32_t index,
                                           uint32_t max_size);

    static uint32_t count_digits(uint32_t nb);
};
} // namespace rendering