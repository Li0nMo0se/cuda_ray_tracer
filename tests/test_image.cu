#include "color/color.cuh"
#include "cuda_tools/cuda_error.cuh"
#include "rendering/image.cuh"
#include "gtest/gtest.h"

TEST(RenderingImage, CreateDeviceImage)
{
    constexpr uint32_t width = 5;
    constexpr uint32_t height = 10;
    rendering::DeviceImage<color::Color3> d_img(width, height);
    d_img.free();
    ASSERT_TRUE(true);
}

TEST(RenderingImage, CreateHostImage)
{
    constexpr uint32_t width = 5;
    constexpr uint32_t height = 10;
    rendering::HostImage<color::Color3> h_img(width, height);
    h_img.free();
    ASSERT_TRUE(true);
}

TEST(RenderingImage, CreateImageHandler)
{
    constexpr uint32_t width = 5;
    constexpr uint32_t height = 10;
    rendering::ImageHandler<color::Color3> img_handler(width, height);
    ASSERT_TRUE(true);
}

__global__ void kernel_set(rendering::DeviceImage<color::Color3> d_img)
{
    color::Color3* const data = d_img.data_get();
    float counter = 1.f;
    for (uint32_t y = 0; y < 10; y++)
    {
        for (uint32_t x = 0; x < 5; x++)
        {
            const uint32_t pixel = y * 5 + x;
            data[pixel] = color::Color3(counter, counter + 1, counter + 2);
            counter++;
        }
    }
}

TEST(RenderingImage, DeviceImageCopyHost)
{
    constexpr uint32_t width = 5;
    constexpr uint32_t height = 10;
    rendering::ImageHandler<color::Color3> img_handler(width, height);
    kernel_set<<<1, 1>>>(img_handler.device);
    cuda_safe_call(cudaDeviceSynchronize());
    check_error();
    img_handler.copy_device_to_host();
    const color::Color3* const res_data = img_handler.host.data_get();

    color::Color3 error = color::Color3(0.f, 0.f, 0.f);
    for (uint32_t y = 0; y < height; y++)
    {
        for (uint32_t x = 0; x < width; x++)
        {
            const uint32_t pixel = y * width + x;
            ASSERT_TRUE(res_data[pixel] != error);
        }
    }
}

// FIXME: copy device to host
// FIXME: copy host to device

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}