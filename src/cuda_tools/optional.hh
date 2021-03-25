#pragma once

namespace cuda_tools
{
struct nullopt
{
    explicit constexpr nullopt() {}
};

template <typename T>
class Optional
{
  public:
  public:
    Optional(const T& value)
        : is_set_(true)
        , value_(value)
    {
    }

    constexpr Optional(nullopt) noexcept
        : is_set_(false)
    {
    }

    constexpr Optional(const Optional& other)
        : is_set_(other.is_set_)
        , value_(other.is_set_ ? other.value_ : T{})
    {
    }

    constexpr Optional& operator=(const Optional& other)
    {
        is_set_ = other.is_set_;
        value_ = is_set_ ? other.value_ : T{};
        return *this;
    }

    constexpr explicit operator bool() const noexcept { return is_set_; }
    constexpr bool has_value() const noexcept { return is_set_; };

    constexpr T& value()
    {
        assert(has_value());
        return value_;
    }

    constexpr const T& value() const
    {
        assert(has_value());
        return value_;
    }

    // Quicker as there's no assert
    constexpr const T& operator*() const { return value_; }
    constexpr T& operator*() { return value_; }

  private:
    bool is_set_;
    T value_;
};
} // namespace cuda_tools