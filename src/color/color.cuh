#pragma once

#include "../space/vector.cuh"

namespace color
{
using Color3 = space::Vector<3, float>;

// FIXME: Make those variables accessible from the device
__device__ constexpr Color3 black() { return Color3(0.f, 0.f, 0.f); }
__device__ constexpr Color3 white() { return Color3(255.f, 255.f, 255.f); }
__device__ constexpr Color3 background() { return black(); }
} // namespace color