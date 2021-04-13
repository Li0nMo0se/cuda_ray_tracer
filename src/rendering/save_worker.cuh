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
               const std::string& output_filename)
        : thread_(std::thread(&VideoEngine::save,
                              frame,
                              width,
                              height,
                              std::ref(output_filename)))
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