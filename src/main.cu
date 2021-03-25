#include "cuda_tools/cuda_error.hh"
#include "cuda_tools/vector.hh"
int main()
{
    cuda_tools::Vector<int> vect;
    vect.emplace_back(10);
    vect.emplace_back(20);

    cuda_safe_call(cudaDeviceSynchronize());
    check_error();
    return 1;
}
