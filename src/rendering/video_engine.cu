#include "parse/parser.cuh"
#include "rendering/engine.cuh"
#include "rendering/save_worker.cuh"
#include "rendering/video_engine.cuh"
#include "scene/scene.cuh"

#include <fstream>
#include <ostream>
#include <vector>

namespace rendering
{

static __global__ void update_scene(scene::Scene::objects_t objects,
                                    const int32_t size)
{
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size)
        objects[x].translate();
}

template <typename T>
static uint8_t get_color_with_boundary(const T val)
{
    if (val < static_cast<T>(0))
        return 0;
    if (val > static_cast<T>(255))
        return 255;
    return static_cast<uint8_t>(val);
}

void VideoEngine::save(const color::Color3* const d_frame,
                       const int32_t width,
                       const int32_t height,
                       const std::string& filename,
                       const cudaStream_t stream)
{
    // Copy frame
    const int32_t total_size = sizeof(color::Color3) * width * height;
    color::Color3* const h_frame =
        static_cast<color::Color3*>(std::malloc(total_size));
    cuda_safe_call(cudaMemcpyAsync(h_frame,
                                   d_frame,
                                   total_size,
                                   cudaMemcpyDeviceToHost,
                                   stream));
    cuda_safe_call(cudaStreamSynchronize(stream));

    // Write file
    std::ofstream of(filename, std::ios_base::out | std::ios_base::binary);

    if (of.fail())
    {
        std::cerr << "Cannot save the image in the file " << filename
                  << std::endl;
        return;
    }

    of << "P6" << std::endl;
    of << width << " " << height << std::endl;
    uint32_t max_value = std::numeric_limits<uint8_t>::max();
    of << max_value << std::endl;

    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            const color::Color3& curr_pixel = h_frame[y * width + x];
            uint8_t r = get_color_with_boundary<float>(curr_pixel[0]);
            uint8_t g = get_color_with_boundary<float>(curr_pixel[1]);
            uint8_t b = get_color_with_boundary<float>(curr_pixel[2]);

            of << r << g << b;
        }
    }

    of.close();
    std::free(h_frame);
}

void VideoEngine::render(const std::string& input_path,
                         const std::string& output_path,
                         const int32_t resolution_width,
                         const int32_t resolution_height,
                         const int32_t nb_frames,
                         const int32_t aliasing_level,
                         const int32_t reflection_max_depth)
{
    const int32_t max_digits = count_digits(nb_frames);

    scene::Scene scene = parse::parse_scene(input_path);
    const int32_t nb_objects = scene.objects_.size_get();

    const int32_t total_resolution = resolution_width * resolution_height;

    color::Color3* frames;
    cuda_safe_call(
        cudaMalloc((void**)&frames,
                   sizeof(color::Color3) * total_resolution * nb_frames));

    cudaStream_t stream_compute;
    cudaStream_t stream_save;
    cuda_safe_call(cudaStreamCreate(&stream_compute));
    cuda_safe_call(cudaStreamCreate(&stream_save));

    std::vector<SaveWorker> workers;
    for (int32_t index_frame = 0; index_frame < nb_frames; index_frame++)
    {
        color::Color3* const curr_frame =
            frames + (total_resolution * index_frame);
        // Synchronization is made after call of the kernel
        Engine::render(curr_frame,
                       resolution_width,
                       resolution_height,
                       scene,
                       aliasing_level,
                       reflection_max_depth,
                       stream_compute);

        workers.emplace_back(
            curr_frame,
            resolution_width,
            resolution_height,
            get_output_filename(output_path, index_frame, max_digits),
            stream_save);

        // FIXME: Correct?
        constexpr int32_t TILE_W = 64;
        const dim3 block(TILE_W);
        const dim3 grid(1 + (nb_objects - 1) / block.x);

        update_scene<<<grid, block, 0, stream_compute>>>(scene.objects_,
                                                         nb_objects);
    }

    for (SaveWorker& worker : workers)
        worker.stop();

    scene.free();
    cuda_safe_call(cudaFree(frames));
}

int32_t VideoEngine::count_digits(int32_t nb)
{
    int32_t nb_digits = 1;
    while (nb >= 10)
    {
        nb_digits++;
        nb /= 10;
    }
    return nb_digits;
}
std::string VideoEngine::get_output_filename(const std::string& output_path,
                                             const int32_t index,
                                             int32_t max_digits)
{
    const int32_t nb_digits_index = count_digits(index);

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