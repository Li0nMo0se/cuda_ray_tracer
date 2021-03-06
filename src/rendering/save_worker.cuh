#pragma once

#include "rendering/video_engine.cuh"
#include <string>
#include <thread>

namespace rendering
{
class SaveWorker
{
  public:
    SaveWorker(const color::Color3* const frame,
               const int32_t width,
               const int32_t height,
               const std::string& output_filename,
               const cudaStream_t stream)
        : thread_(std::thread(&VideoEngine::save,
                              frame,
                              width,
                              height,
                              output_filename,
                              stream))
    {
    }

    void stop()
    {
        if (thread_.joinable())
            thread_.join();
    }

  private:
    std::thread thread_;
};
} // namespace rendering