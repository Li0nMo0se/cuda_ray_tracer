#include "rendering/video_engine.cuh"
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " input.scene output_directory width height nb_frames "
                     "aliasing_level "
                  << "reflection_max_depth\n";
        return EXIT_FAILURE;
    }
    uint32_t width;
    std::stringstream ss_width(argv[3]);
    ss_width >> width;

    uint32_t height;
    std::stringstream ss_height(argv[4]);
    ss_height >> height;

    uint32_t nb_frames;
    std::stringstream ss_nb_frames(argv[5]);
    ss_nb_frames >> nb_frames;

    uint32_t aliasing_level;
    std::stringstream ss_aliasing(argv[6]);
    ss_aliasing >> aliasing_level;

    uint32_t reflection_max_depth;
    std::stringstream ss_reflection(argv[7]);
    ss_reflection >> reflection_max_depth;

    rendering::VideoEngine::render(argv[1],
                                   argv[2],
                                   width,
                                   height,
                                   nb_frames,
                                   aliasing_level,
                                   reflection_max_depth);

    return EXIT_SUCCESS;
}
