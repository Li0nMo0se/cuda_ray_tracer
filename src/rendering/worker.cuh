#pragma once

#include "cuda_tools/cuda_error.cuh"
#include <string>
#include <thread>

namespace rendering
{
class Worker
{
  public:
    Worker(const std::string& input_filename,
           const std::string& output_filename,
           const uint32_t width,
           const uint32_t height,
           const uint32_t aliasing_level,
           const uint32_t reflection_max_depth)
    {
        cuda_safe_call(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        thread_ = std::thread(&VideoEngine::render_frame,
                              input_filename,
                              output_filename,
                              width,
                              height,
                              aliasing_level,
                              reflection_max_depth,
                              stream_);
    }

    void stop()
    {
        if (thread_.joinable())
            thread_.join();
    }

  private:
    std::thread thread_;
    cudaStream_t stream_;
};
} // namespace rendering