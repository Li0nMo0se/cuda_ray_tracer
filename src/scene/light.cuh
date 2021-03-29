#pragma once

namespace scene
{
class Light
{
  public:
    Light(const space::Point3& origin, const float intensity)
        : origin_(origin)
        , intensity_(intensity)
    {
    }

    virtual ~Light() = default;
    Light(const Light&) = default;
    virtual Light& operator=(const Light&) = default;

    float intensity_get() const { return intensity_; }
    const space::Point3& origin_get() const { return origin_; }

  protected:
    const space::Point3 origin_;
    const float intensity_;
};
} // namespace scene