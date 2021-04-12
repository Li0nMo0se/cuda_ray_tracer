#include "parser.cuh"

#include "color/texture_material.cuh"
#include "color/uniform_texture.cuh"
#include "scene/camera.cuh"
#include "scene/plan.cuh"
#include "scene/point_light.cuh"
#include "scene/raybox.cuh"
#include "scene/sphere.cuh"
#include "scene/triangle.cuh"
#include "space/vector.cuh"

#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <unordered_map>

#ifndef M_PI
constexpr float M_PI = 3.141592653589793;
#endif

namespace parse
{
using map_texture_t =
    std::unordered_map<std::string, const color::TextureMaterial*>;

/*** Vector ***/
/* Parse a vector of 3 coordinates */
static space::Vector3 parse_vector(std::string str)
{
    // A vector looks like this: (x,y,z)
    str.erase(0, 1); // skip '('

    float x;
    std::stringstream ss_x(str);
    ss_x >> x; // x

    std::getline(ss_x, str); // get the rest of the streamstring
    str.erase(0, 1); // skip ','

    float y;
    std::stringstream ss_y(str);
    ss_y >> y; // y

    std::getline(ss_y, str); // get the rest of the streamstring
    str.erase(0, 1); // skip ','

    float z;
    std::stringstream ss_z(str);
    ss_z >> z; // z

    // ignore ')'

    return space::Vector3(x, y, z);
}

static space::Vector3 get_translation(std::stringstream& ss)
{
    std::string translation_str;
    ss >> translation_str;

    space::Vector3 translation = space::Vector3(0.f, 0.f, 0.f);
    if (translation_str.size() > 0)
        translation = parse_vector(translation_str);

    return translation;
}

/*** Texture ***/
/* Parse an Uniform Texture */
static void parse_texture(const std::string& line,
                          scene::Scene::textures_t& textures,
                          map_texture_t& name_to_texture,
                          const int32_t nb_line)
{
    std::stringstream ss(line);
    std::string texture_type;
    ss >> texture_type; // UniformTexture
    std::string texture_name;
    ss >> texture_name;

    std::string color_str;
    ss >> color_str;
    const color::Color3 color = parse_vector(color_str);

    float kd, ks, ns;
    ss >> kd >> ks >> ns;

    if (name_to_texture.find(texture_name) ==
        name_to_texture.end()) // Not found
    {
        textures.emplace_back<color::UniformTexture>(color, kd, ks, ns);
        const color::TextureMaterial** texture = textures.back_get();
        name_to_texture.insert({texture_name, *texture});
        delete texture;
    }
    else
        throw ParseError("Redefinition of " + texture_name, nb_line);
}

static const color::TextureMaterial*
get_texture(std::stringstream& ss,
            const map_texture_t& name_to_texture,
            const int32_t nb_line)
{
    std::string texture_name;
    ss >> texture_name;
    auto it = name_to_texture.find(texture_name);
    if (it == name_to_texture.end())
        throw ParseError("No such texture " + texture_name, nb_line);
    return it->second;
}

/*** Camera ***/
/* Parse a line which describe the camera */
static scene::Camera parse_camera(const std::string& line)
{
    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp; // Camera
    ss >> tmp; // origin
    const space::Vector3 origin = parse_vector(tmp);
    ss >> tmp; // y_axis
    const space::Vector3 y_axis = parse_vector(tmp);
    ss >> tmp; // z_axis
    const space::Vector3 z_axis = parse_vector(tmp);
    float z_min;
    ss >> z_min;
    float alpha;
    ss >> alpha;
    // convert to radian
    alpha = alpha * M_PI / 180.f;
    float beta;
    ss >> beta;
    // convert to radian
    beta = beta * M_PI / 180.f;

    return scene::Camera(origin, y_axis, z_axis, z_min, alpha, beta);
}

/*** Parse lights ***/
static void parse_pointlight(const std::string& line,
                             scene::Scene::lights_t& lights)
{
    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp; // PointLight

    std::string origin_str;
    ss >> origin_str;
    space::Vector3 origin = parse_vector(origin_str);

    float intensity;
    ss >> intensity;
    lights.emplace_back<scene::PointLight>(origin, intensity);
}

/*** Parse objects ***/
static void parse_sphere(const std::string& line,
                         scene::Scene::objects_t& objects,
                         map_texture_t& name_to_texture,
                         const int32_t nb_line)
{
    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp; // Sphere

    std::string origin_str;
    ss >> origin_str;
    const space::Vector3 origin = parse_vector(origin_str);

    float radius;
    ss >> radius;

    const color::TextureMaterial* const texture =
        get_texture(ss, name_to_texture, nb_line);

    const space::Vector3 translation = get_translation(ss);

    objects.emplace_back<scene::Sphere>(origin, radius, texture, translation);
}

static void parse_plan(const std::string& line,
                       scene::Scene::objects_t& objects,
                       map_texture_t& name_to_texture,
                       const int32_t nb_line)
{
    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp; // Plan

    std::string origin_str;
    ss >> origin_str;
    const space::Vector3 origin = parse_vector(origin_str);

    std::string normal_str;
    ss >> normal_str;
    const space::Vector3 normal = parse_vector(normal_str);

    const color::TextureMaterial* const texture =
        get_texture(ss, name_to_texture, nb_line);

    const space::Vector3 translation = get_translation(ss);

    objects.emplace_back<scene::Plan>(origin, normal, texture, translation);
}

static void parse_raybox(const std::string& line,
                         scene::Scene::objects_t& objects,
                         map_texture_t& name_to_texture,
                         const int32_t nb_line)
{

    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp; // Raybox

    std::string lower_bound_str;
    ss >> lower_bound_str;
    const space::Vector3 lower_bound = parse_vector(lower_bound_str);

    std::string higher_bound_str;
    ss >> higher_bound_str;
    const space::Vector3 higher_bound = parse_vector(higher_bound_str);

    const color::TextureMaterial* const texture =
        get_texture(ss, name_to_texture, nb_line);

    const space::Vector3 translation = get_translation(ss);

    objects.emplace_back<scene::RayBox>(lower_bound,
                                        higher_bound,
                                        texture,
                                        translation);
}

static void parse_triangle(const std::string& line,
                           scene::Scene::objects_t& objects,
                           map_texture_t& name_to_texture,
                           const int32_t nb_line)
{
    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp; // Triangle

    std::string A_str;
    ss >> A_str;
    const space::Vector3 A = parse_vector(A_str);

    std::string B_str;
    ss >> B_str;
    const space::Vector3 B = parse_vector(B_str);

    std::string C_str;
    ss >> C_str;
    const space::Vector3 C = parse_vector(C_str);

    const color::TextureMaterial* const texture =
        get_texture(ss, name_to_texture, nb_line);

    const space::Vector3 translation = get_translation(ss);

    return objects.emplace_back<scene::Triangle>(A, B, C, texture, translation);
}

scene::Scene parse_scene(const std::string& filename)
{
    int32_t nb_line = 1;
    std::ifstream in(filename);
    if (!in)
    {
        throw std::invalid_argument(
            "Cannot parse the scene: no such file: " + filename + ".");
    }

    // File is valid
    std::optional<scene::Camera> camera;

    scene::Scene::objects_t objects;
    scene::Scene::lights_t lights;
    scene::Scene::textures_t textures;

    map_texture_t name_to_texture;

    std::string line;
    while (std::getline(in, line))
    {
        if (!(line.empty() || line[0] == '#'))
        {
            std::stringstream ss(line);
            std::string curr_token;
            ss >> curr_token;
            if (curr_token == "Camera")
                camera = parse_camera(line);
            else if (curr_token == "UniformTexture")
                parse_texture(line, textures, name_to_texture, nb_line);
            else if (curr_token == "Sphere")
                parse_sphere(line, objects, name_to_texture, nb_line);
            else if (curr_token == "Plan")
                parse_plan(line, objects, name_to_texture, nb_line);
            else if (curr_token == "Raybox")
                parse_raybox(line, objects, name_to_texture, nb_line);
            else if (curr_token == "Triangle")
                parse_triangle(line, objects, name_to_texture, nb_line);
            else if (curr_token == "PointLight")
                parse_pointlight(line, lights);
            else
                throw ParseError("Undefined structure: " + curr_token, nb_line);
        }
        nb_line++;
    }

    if (!camera)
        throw ParseError("Camera is missing", nb_line);

    return scene::Scene(camera.value(), objects, lights, textures);
}
} // namespace parse