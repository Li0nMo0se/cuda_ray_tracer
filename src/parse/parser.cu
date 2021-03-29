#include "parser.cuh"

#include <fstream>
#include <sstream>

// FIXME: To delete
#include <iostream>

namespace parse
{
void Parser::parse_scene(const std::string& filename)
{
    nb_line_ = 1;
    std::ifstream in(filename);
    if (!in)
    {
        throw std::invalid_argument(
            "Cannot parse the scene: no such file: " + filename + ".");
    }

    // File is valid
    std::string line;
    while (std::getline(in, line))
    {
        nb_line_++;
        if (!(line.empty() || line[0] == '#'))
        {
            parse_vector(line);
        }
    }
}

void Parser::parse_vector(const std::string& vector)
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

    // return space::Vector3(x, y, z);
}
} // namespace parse