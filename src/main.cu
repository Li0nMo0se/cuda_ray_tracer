#include "cuda_tools/cuda_error.hh"
#include "cuda_tools/optional.hh"
#include "cuda_tools/vector.hh"
int main()
{
    cuda_tools::Vector<int> vect;
    vect.emplace_back(10);
    vect.emplace_back(20);

    cuda_safe_call(cudaDeviceSynchronize());
    check_error();

    cuda_tools::Optional<int> a = cuda_tools::nullopt{};
    std::cout << std::boolalpha << a.has_value() << std::endl;
    a = 2;
    std::cout << std::boolalpha << a.has_value() << ' ' << *a << std::endl;
    return 1;
}
