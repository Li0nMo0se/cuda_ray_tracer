#include "parser.cuh"

#include <cmath>
#include <fstream>
#include <sstream>

namespace parse
{

scene::Scene Parser::parse_scene(const std::string& filename)
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

    while (std::getline(in, line))
    {
        if (!(line.empty() || line[0] == '#'))
        {
            std::stringstream ss(line);
            std::string curr_token;
            ss >> curr_token;
            //   throw ParseError("Undefined structure: " + curr_token,
            //                    nb_line_);
        }
        nb_line++;
    }

    return scene::Scene(camera, objects, lights);
}

/* Parse a line which describe the camera */
scene::Camera Parser::parse_camera(const std::string& line)
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

space::Vector3 Parser::parse_vector(const std::string& vector)
{
    std::string tmp = vector;
    // A vector looks like this: (x,y,z)
    tmp.erase(0, 1); // skip '('

    float x;
    std::stringstream ss_x(tmp);
    ss_x >> x; // x

    std::getline(ss_x, tmp); // get the rest of the streamstring
    tmp.erase(0, 1); // skip ','

    float y;
    std::stringstream ss_y(tmp);
    ss_y >> y; // y

    std::getline(ss_y, tmp); // get the rest of the streamstring
    tmp.erase(0, 1); // skip ','

    float z;
    std::stringstream ss_z(tmp);
    ss_z >> z; // z

    // ignore ')'

    return space::Vector3(x, y, z);
}
} // namespace parse