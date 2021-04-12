#include "parse/parser.cuh"
#include "rendering/engine.cuh"
#include "rendering/video_engine.cuh"
#include "rendering/worker.cuh"
#include "scene/scene.cuh"
#include <algorithm>
#include <filesystem>
#include <vector>

namespace rendering
{

void VideoEngine::render_frame(const std::string& input_filename,
                               const std::string& output_filename,
                               const uint32_t width,
                               const uint32_t height,
                               const uint32_t aliasing_level,
                               const uint32_t reflection_max_depth,
                               const cudaStream_t stream)
{
    scene::Scene scene = parse::parse_scene(input_filename, stream);
    Engine::render(output_filename,
                   width,
                   height,
                   scene,
                   aliasing_level,
                   reflection_max_depth,
                   stream);
}

void VideoEngine::render(const std::string& input_path,
                         const std::string& output_path,
                         const uint32_t width,
                         const uint32_t height,
                         const uint32_t aliasing_level,
                         const uint32_t reflection_max_depth)
{
    // Get input files
    std::vector<std::string> filename_scenes;
    for (const auto& entry : std::filesystem::directory_iterator(input_path))
        filename_scenes.push_back(entry.path().string());
    std::sort(filename_scenes.begin(), filename_scenes.end());

    std::vector<Worker> workers;
    int32_t index_frame = 0;
    int32_t max_frames = filename_scenes.size();
    // Foreach file, render a frame
    for (const std::string& filename_scene : filename_scenes)
    {
        std::string output_filename =
            get_output_filename(output_path, index_frame, max_frames);
        // run the worker
        workers.emplace_back(filename_scene,
                             output_filename,
                             width,
                             height,
                             aliasing_level,
                             reflection_max_depth);
        index_frame++;
    }

    for (Worker& worker : workers)
    {
        worker.stop();
    }
}

std::string VideoEngine::get_output_filename(const std::string& output_path,
                                             const uint32_t index,
                                             uint32_t max_size)
{
    uint32_t nb_digits_max = 1;
    while (max_size >= 10)
    {
        max_size /= 10;
        nb_digits_max += 1;
    }

    uint32_t nb_digits_curr = 1;
    uint32_t tmp_index = index;
    while (tmp_index >= 10)
    {
        tmp_index /= 10;
        nb_digits_curr += 1;
    }

    std::string final_string = output_path + "/";
    while (nb_digits_max > nb_digits_curr)
    {
        final_string += "0";
        nb_digits_max--;
    }

    final_string += std::to_string(index) + ".ppm";

    return final_string;
}
} // namespace rendering