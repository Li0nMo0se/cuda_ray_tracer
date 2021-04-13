#pragma once

#include "color/color.cuh"
#include <string>

namespace rendering
{

class VideoEngine final
{
  public:
    static void render(const std::string& input_path,
                       const std::string& output_path,
                       const int32_t resolution_width,
                       const int32_t resolution_height,
                       const int32_t nb_frames,
                       const int32_t aliasing_level,
                       const int32_t reflection_max_depth);

    VideoEngine() = delete;
    VideoEngine& operator=(const VideoEngine&) = delete;
    VideoEngine(const VideoEngine&) = delete;

    static void save(const color::Color3* const d_frame,
                     const int32_t width,
                     const int32_t height,
                     const std::string& filename);

  private:
    static std::string get_output_filename(const std::string& output_path,
                                           const int32_t index,
                                           int32_t max_size);

    static int32_t count_digits(int32_t nb);
};
} // namespace rendering