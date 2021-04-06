#include "rendering/engine.cuh"

namespace rendering
{

__device__ cuda_tools::Optional<space::IntersectionInfo>
Engine::cast_ray(const space::Ray& ray, const scene::Scene& scene)
{
    cuda_tools::Optional<space::IntersectionInfo> closest_intersection =
        cuda_tools::nullopt;

    // Find the closest intersection if it exists
    for (int32_t i = 0; i < scene.objects_.size_get(); i++)
    {
        const cuda_tools::Optional<space::IntersectionInfo> intersection =
            scene.objects_[i].intersect(ray);
        // If an intersection is found, assign this intersection if closer to
        // the origin of the ray
        if (intersection.has_value() &&
            (!closest_intersection || intersection.value().t_get() <
                                          closest_intersection.value().t_get()))
        {
            closest_intersection = intersection;
        }
    }

    // Compute the intersection point if there was an intersection
    if (closest_intersection.has_value())
        closest_intersection.value().compute_intersection(ray);
    return closest_intersection;
}

__device__ bool Engine::check_shadow(const scene::Scene& scene,
                                     const scene::Light& light,
                                     const space::Point3& intersection)
{
    const space::Vector3 vector_to_light = light.origin_get() - intersection;
    const space::Ray ray(intersection, vector_to_light.normalized());

    const cuda_tools::Optional<space::IntersectionInfo> intersection_info =
        cast_ray(ray, scene);

    if (!intersection_info.has_value())
        return false;

    // Is the intersection of the ray between the intersected point and the
    // light?
    // t_intersected is the distance between the intersected point and the
    // origin (that's the definition of a ray)
    const float distance_to_light = vector_to_light.length();
    return intersection_info.value().t_get() < distance_to_light;
}

__device__ color::Color3
Engine::get_object_color(const scene::Scene& scene,
                         const space::Ray& ray,
                         const space::IntersectionInfo& intersection_info)
{
    const scene::Object& obj = intersection_info.obj_get();
    const color::TextureMaterial& texture = obj.get_texture();
    const space::Vector3& intersection = intersection_info.intersection_get();

    const float kd = texture.get_kd(intersection);
    const float ks = texture.get_ks(intersection);
    const float ns = texture.get_ns(intersection);
    const color::Color3 obj_color = texture.get_color(intersection);

    // Normal of the object at the intersection point
    const space::Vector3& normal = obj.normal_get(ray, intersection_info);

    // Compute the reflected vector
    const space::Vector3 S =
        intersection - normal * 2 * intersection.dot(normal);

    color::Color3 color = color::black();

    for (int32_t i = 0; i < scene.lights_.size_get(); i++)
    {
        const scene::Light& light = scene.lights_[i];

        // Compute shadow (+ normal to avoid intersecting with yourself)
        if (check_shadow(scene, light, intersection))
            continue;

        const space::Vector3 L = light.origin_get() - intersection;
        const float intensity = light.intensity_get();
        // Compute the diffuse light
        const float coeff_diffuse =
            kd * normal.dot(L) * intensity * distance_attenuation(L.length());
        color += obj_color * coeff_diffuse;

        // Compute the specular light
        const float coeff_specular = ks * intensity * powf(S.dot(L), ns);
        if (coeff_specular > 0)
            color += coeff_specular;
    }

    return color;
}

__device__ inline float Engine::distance_attenuation(const float distance)
{
    return 1.f / distance;
}

__device__ color::Color3 Engine::cast_ray_color(const space::Ray& ray,
                                                const scene::Scene& scene)
{
    cuda_tools::Optional<space::IntersectionInfo> intersection =
        cast_ray(ray, scene);

    if (intersection.has_value())
    {
        space::IntersectionInfo& intersection_v = intersection.value();
        const scene::Object& intersected_obj = intersection_v.obj_get();
        // FIXME find more elegant way to do this
        intersection_v.auto_intersection_correction(
            intersected_obj.normal_get(ray, intersection_v));
        return get_object_color(scene, ray, intersection_v);
    }
    return color::background();
}

__device__ color::Color3
Engine::get_pixel_color(const space::Point3& curr_pixel,
                        const scene::Scene& scene,
                        const int32_t unit_x,
                        const int32_t unit_y,
                        const int32_t aliasing_level,
                        const int32_t reflection_max_depth)
{
    // FIXME: aliasing
    // FIXME: reflection

    const scene::Camera& camera = scene.camera_get();
    const space::Vector3 ray_direction =
        (curr_pixel - camera.origin_).normalized();
    const space::Ray ray(camera.origin_, ray_direction);

    return cast_ray_color(ray, scene);
}

struct FrameInfo
{
    const space::Point3 top_left;
    const float unit_x;
    const float unit_y;
};

// Copy arguments to have them in gpu registers, cache L1...
__global__ void kernel_render(DeviceImage<color::Color3> d_img,
                              const scene::Scene scene,
                              const FrameInfo frame_info,
                              const int32_t aliasing_level,
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
    data[y * d_img.width_get() + x] =
        Engine::get_pixel_color(curr_pixel,
                                scene,
                                frame_info.unit_x,
                                frame_info.unit_y,
                                aliasing_level,
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
                                   aliasing_level,
                                   reflection_max_depth);
    cuda_safe_call(cudaDeviceSynchronize());
    check_error();

    // scene not usable because it has been copied

    // Retrive image
    im.save(filename);
}
} // namespace rendering