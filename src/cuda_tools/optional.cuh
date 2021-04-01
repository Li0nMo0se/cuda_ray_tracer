#pragma once

namespace cuda_tools
{
// FIXME: Is this really working? (for host & device memory?)
extern struct _nullopt
{
    explicit constexpr _nullopt() {}
} nullopt;

template <typename T>
class Optional
{
  public:
    __host__ __device__ Optional(const T& value)
        : is_set_(true)
        , value_(value)
    {
    }

    __host__ __device__ Optional(_nullopt) noexcept
        : is_set_(false)
    {
    }

    __host__ __device__ Optional() noexcept
        : Optional(nullopt)
    {
    }

    __host__ __device__ Optional(const Optional& other)
        : is_set_(other.is_set_)
        , value_(other.is_set_ ? other.value_ : T{})
    {
    }

    __host__ __device__ Optional& operator=(const Optional& other)
    {
        is_set_ = other.is_set_;
        value_ = is_set_ ? other.value_ : T{};
        return *this;
    }

    __host__ __device__ explicit operator bool() const noexcept
    {
        return is_set_;
    }

    __host__ __device__ bool has_value() const noexcept { return is_set_; };

    __host__ __device__ T& value()
    {
        assert(has_value());
        return value_;
    }

    __host__ __device__ const T& value() const
    {
        assert(has_value());
        return value_;
    }

    // Quicker as there's no assert
    __host__ __device__ const T& operator*() const { return value_; }
    __host__ __device__ T& operator*() { return value_; }

  private:
    bool is_set_;
    T value_;
};
} // namespace cuda_tools