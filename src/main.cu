#include "cuda_tools/cuda_error.cuh"
#include "cuda_tools/optional.hh"
#include "cuda_tools/vector.cuh"
#include "parse/parser.cuh"
#include "space/vector.cuh"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
        return 1;
    }

    // Cuda tools stuff
    cuda_tools::Vector<int> vect;
    vect.emplace_back<int>(10);
    vect.emplace_back<int>(20);

    cuda_safe_call(cudaDeviceSynchronize());
    check_error();

    cuda_tools::Optional<int> a = cuda_tools::nullopt;
    std::cout << std::boolalpha << a.has_value() << std::endl;
    a = 2;
    std::cout << std::boolalpha << a.has_value() << ' ' << *a << std::endl;

    // Parser
    parse::Parser parser;
    parser.parse_scene(argv[1]);

    // Vector
    space::Vector3 vect3(1.f, 2.f, 3.f);
    vect3 = space::Vector3(2.f, 3.f, 4.f);

    std::cout << vect3[0] << " " << vect3[1] << " " << vect3[2] << "\n";

    return 0;
}
