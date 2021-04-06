#include "rendering/engine.cuh"
#include "rendering/image.cuh"

namespace rendering
{
struct FrameInfo
{
    const space::Point3 top_left;
    const float unit_x;
    const float unit_y;
};

static __device__ color::Color3
get_pixel_color(const space::Point3& curr_pixel,
                const scene::Scene& scene,
                const int32_t unit_x,
                const int32_t unit_y,
                const int32_t reflection_max_depth)
{
}

// Copy arguments to have them in gpu registers, cache L1...
static __global__ void kernel_render(DeviceImage<color::Color3> d_img,
                                     const scene::Scene scene,
                                     const FrameInfo frame_info,
                                     const int32_t reflection_max_depth)
{
    const int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= d_img.width_get() || y >= d_img.height_get())
        return;

    const scene::Camera& camera = scene.camera_get();
    const space::Point3 curr_pixel = frame_info.top_left +
                                     x * (frame_info.unit_x * camera.x_axis_) -
                                     y * (frame_info.unit_y * camera.y_axis_);

    color::Color3* const data = d_img.data_get();
    // Ray computation with aliasing
    data[y * d_img.width_get() + x] = get_pixel_color(curr_pixel,
                                                      scene,
                                                      frame_info.unit_x,
                                                      frame_info.unit_y,
                                                      reflection_max_depth);
}

void Engine::render(const std::string& filename,
                    const int32_t resolution_width,
                    const int32_t resolution_height,
                    scene::Scene& scene,
                    const int32_t aliasing_level,
                    const int32_t reflection_max_depth)
{
    // Create Image
    ImageHandler<color::Color3> im(resolution_width, resolution_height);

    // Find width & height of a pixel
    const scene::Camera& camera = scene.camera_get();

    // Compute the height and width of the image in the 3D world
    const float height = std::tan(camera.alpha_) * camera.z_min_ * 2;
    const float width = std::tan(camera.beta_) * camera.z_min_ * 2;

    // Size of a pixel in the 3D world
    const float unit_x = width / resolution_width;
    const float unit_y = height / resolution_height;

    // Find top-left pixel
    // This space::Point3 will be used as a base for vector generation

    // P is the projection of `C` on the image plan
    const space::Point3 p = camera.origin_ + camera.z_axis_ * camera.z_min_;
    // Find the very top left point of the image in the 3D world
    space::Point3 top_left =
        p - (width / 2 * camera.x_axis_) + (height / 2 * camera.y_axis_);
    // Find the center of the top left pixel
    top_left = top_left + (unit_x / 2 * camera.x_axis_) -
               (unit_y / 2 * camera.y_axis_);

    // foreach pixel of the image
    //      Compute the ray from the origin of the camera to the pixel
    //      Find intersections of this ray with every objects of the scene
    //      (Calculate specular & diffuse contribution)
    constexpr int TILE_W = 32;
    constexpr int TILE_H = 8;
    constexpr dim3 block(TILE_W, TILE_H);
    const dim3 grid(1 + (resolution_width - 1) / block.x,
                    1 + (resolution_height - 1) / block.y);

    const FrameInfo frame_info{top_left, unit_x, unit_y};
    kernel_render<<<grid, block>>>(im.device,
                                   scene,
                                   frame_info,
                                   reflection_max_depth);
    cudaDeviceSynchronize();
    check_error();

    // scene not usable because it has been copied

    // Retrive image
    im.save(filename);
}
} // namespace rendering