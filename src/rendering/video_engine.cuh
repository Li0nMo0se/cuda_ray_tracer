#pragma once

#include <string>
namespace rendering
{

class VideoEngine final
{
  public:
    static void render(const std::string& input_path,
                       const std::string& output_path,
                       const uint32_t width,
                       const uint32_t height,
                       const uint32_t aliasing_level,
                       const uint32_t reflection_max_depth);

    VideoEngine() = delete;
    VideoEngine& operator=(const VideoEngine&) = delete;
    VideoEngine(const VideoEngine&) = delete;

    // FIXME: Shouldn't be public
    static void render_frame(const std::string& input_filename,
                             const std::string& output_filename,
                             const uint32_t width,
                             const uint32_t height,
                             const uint32_t aliasing_level,
                             const uint32_t reflection_max_depth,
                             const cudaStream_t stream);

  private:
    static std::string get_output_filename(const std::string& output_path,
                                           const uint32_t index,
                                           uint32_t max_size);
};
} // namespace rendering