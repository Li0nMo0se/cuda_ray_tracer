#include "parser.cuh"

#include "color/texture_material.cuh"
#include "color/uniform_texture.cuh"
#include "scene/camera.cuh"
#include "scene/point_light.cuh"
#include "scene/sphere.cuh"
#include "space/vector.cuh"

#include <cmath>
#include <fstream>
#include <optional>
#include <sstream>
#include <unordered_map>

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
        name_to_texture.insert({texture_name, textures.back_get()});
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
    objects.emplace_back<scene::Sphere>(origin, radius, texture);
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