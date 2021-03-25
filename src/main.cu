#include "cuda_tools/cuda_error.cuh"
#include "cuda_tools/optional.hh"
#include "cuda_tools/vector.cuh"
#include "parse/parser.hh"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 2)
        std::cerr << "Usage: " << argv[0] << " filename" << std::endl;

    // Cuda tools stuff
    cuda_tools::Vector<int> vect;
    vect.emplace_back(10);
    vect.emplace_back(20);

    cuda_safe_call(cudaDeviceSynchronize());
    check_error();

    cuda_tools::Optional<int> a = cuda_tools::nullopt{};
    std::cout << std::boolalpha << a.has_value() << std::endl;
    a = 2;
    std::cout << std::boolalpha << a.has_value() << ' ' << *a << std::endl;

    // Parser
    parse::Parser parser;
    parser.parse_scene(argv[1]);

    return 1;
}
