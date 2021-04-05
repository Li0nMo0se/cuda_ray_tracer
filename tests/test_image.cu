#include "color/color.cuh"
#include "rendering/image.cuh"
#include "gtest/gtest.h"

TEST(Image, CreateDeviceImage)
{
    constexpr uint32_t width = 5;
    constexpr uint32_t height = 10;
    rendering::DeviceImage<color::Color3> d_img(width, height);
    ASSERT_TRUE(true);
}
int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}