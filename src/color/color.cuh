#pragma once

#include "../space/vector.cuh"

namespace color
{
using Color3 = space::Vector<3, float>;

const Color3 black = Color3(0.f, 0.f, 0.f);
const Color3 white = Color3(255.f, 255.f, 255.f);
} // namespace color