#include "parse/parser.cuh"
#include "rendering/engine.cuh"
#include "rendering/video_engine.cuh"
#include "scene/scene.cuh"

namespace rendering
{

static __global__ void update_scene(scene::Scene::objects_t objects)
{
    objects[threadIdx.x * blockIdx.x].translate();
}

void VideoEngine::render(const std::string& input_path,
                         const std::string& output_path,
                         const uint32_t resolution_width,
                         const uint32_t resolution_height,
                         const uint32_t nb_frames,
                         const uint32_t aliasing_level,
                         const uint32_t reflection_max_depth)
{
    const uint32_t max_digits = count_digits(nb_frames);

    scene::Scene scene = parse::parse_scene(input_path);

    for (uint32_t index_frame = 0; index_frame < nb_frames; index_frame++)
    {
        const std::string output_filename =
            get_output_filename(output_path, index_frame, max_digits);

        Engine::render(output_filename,
                       resolution_width,
                       resolution_height,
                       scene,
                       aliasing_level,
                       reflection_max_depth);

        update_scene<<<1, 1>>>(scene.objects_);
    }

    scene.free();
}

uint32_t VideoEngine::count_digits(uint32_t nb)
{
    uint32_t nb_digits = 1;
    while (nb >= 10)
    {
        nb_digits++;
        nb /= 10;
    }
    return nb_digits;
}
std::string VideoEngine::get_output_filename(const std::string& output_path,
                                             const uint32_t index,
                                             uint32_t max_digits)
{
    const uint32_t nb_digits_index = count_digits(index);

    std::string final_string = output_path + "/";
    while (max_digits > nb_digits_index)
    {
        final_string += "0";
        max_digits--;
    }

    final_string += std::to_string(index) + ".ppm";

    return final_string;
}

} // namespace rendering