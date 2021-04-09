#include "parse/parser.cuh"
#include "rendering/engine.cuh"
#include "scene/scene.cuh"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc != 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " file.scene outputfile.ppm width height aliasing_level "
                  << "reflection_max_depth\n";
        return EXIT_FAILURE;
    }
    uint32_t width;
    std::stringstream ss_width(argv[3]);
    ss_width >> width;

    uint32_t height;
    std::stringstream ss_height(argv[4]);
    ss_height >> height;

    uint32_t aliasing_level;
    std::stringstream ss_aliasing(argv[5]);
    ss_aliasing >> aliasing_level;

    uint32_t reflection_max_depth;
    std::stringstream ss_reflection(argv[6]);
    ss_reflection >> reflection_max_depth;

    std::string path_to_dir = argv[1];
    std::vector<std::string> filename_scenes;

    for (const auto& entry : std::filesystem::directory_iterator(path_to_dir))
        filename_scenes.push_back(entry.path());
    std::sort(filename_scenes.begin(), filename_scenes.end());

    for (const std::string& filename_scene : filename_scenes)
    {
        scene::Scene scene = parse::parse_scene(filename_scene);
        std::string output_file = filename_scene + ".out.ppm";
        rendering::Engine::render(output_file,
                                  width,
                                  height,
                                  scene,
                                  aliasing_level,
                                  reflection_max_depth);
    }

    return EXIT_SUCCESS;
}
