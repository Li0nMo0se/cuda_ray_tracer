#include "parser.cuh"

#include "color/texture_material.cuh"
#include "color/uniform_texture.cuh"
#include "scene/camera.cuh"
#include "space/vector.cuh"

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace parse
{

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

/* Parse an Uniform Texture */
void parse_texture(
    const std::string& line,
    scene::Scene::textures_t& textures,
    std::unordered_map<std::string, const color::TextureMaterial*>&
        name_to_texture)
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
    // else
    // FIXME: Parse error if textures already exists
}

/* Parse a line which describe the camera */
static scene::Camera parse_camera(const std::string& line)
{
    std::stringstream ss(line);
    std::string tmp;
    ss >> tmp;
    ss >> tmp;
    const space::Vector3 origin = parse_vector(tmp);
    ss >> tmp;
    const space::Vector3 y_axis = parse_vector(tmp);
    ss >> tmp;
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
    // Parse Camera first
    std::string line;
    while (std::getline(in, line))
    {
        nb_line++;
        if (!(line.empty() || line[0] == '#'))
            break;
    }
    // FIXME: Error if camera not found
    scene::Camera camera = parse_camera(line);

    scene::Scene::objects_t objects;
    scene::Scene::lights_t lights;
    scene::Scene::textures_t textures;

    std::unordered_map<std::string, const color::TextureMaterial*>
        name_to_texture;

    while (std::getline(in, line))
    {
        if (!(line.empty() || line[0] == '#'))
        {
            std::stringstream ss(line);
            std::string curr_token;
            ss >> curr_token;
            if (curr_token == "UniformTexture")
                parse_texture(line, textures, name_to_texture);
            // else
            //   throw ParseError("Undefined structure: " + curr_token,
            //                    nb_line_);
        }
        nb_line++;
    }

    return scene::Scene(camera, objects, lights, textures);
}
} // namespace parse