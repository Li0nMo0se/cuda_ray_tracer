#pragma once

#include <cassert>

namespace space
{
template <unsigned int size, typename T = float>
class Vector final
{
  protected:
    T vect_[size];

  public:
    template <typename... Ts>
    __host__ __device__ Vector(Ts... args)
        : vect_{args...}
    {
        static_assert(sizeof...(Ts) == size,
                      "Vector constructor: wrong number of arguments");
    }

    Vector() = default;

    __host__ __device__ Vector& operator=(const Vector& vector);
    __host__ __device__ Vector(const Vector& vector);

    __host__ __device__ inline Vector& operator*=(T rhs);
    __host__ __device__ inline Vector operator*(T rhs) const;

    // Perform bitwise multiplication
    __host__ __device__ inline Vector& operator*=(const Vector& rhs);
    // Perform bitwise multiplication
    __host__ __device__ inline Vector operator*(const Vector& rhs) const;

    __host__ __device__ inline Vector& operator/=(T rhs);
    __host__ __device__ inline Vector operator/(T rhs) const;

    // Perform bitwise division
    __host__ __device__ inline Vector& operator/=(const Vector& rhs);
    // Perform bitwise division
    __host__ __device__ inline Vector operator/(const Vector& rhs) const;

    __host__ __device__ inline Vector& operator+=(const Vector& rhs);
    __host__ __device__ inline Vector operator+(const Vector& rhs) const;

    __host__ __device__ inline Vector& operator+=(const T val);
    __host__ __device__ inline Vector operator+(const T val) const;

    __host__ __device__ inline Vector& operator-=(const Vector& rhs);
    __host__ __device__ inline Vector operator-(const Vector& rhs) const;

    __host__ __device__ inline Vector operator-() const;

    __host__ __device__ inline T dot(const Vector& rhs) const;

    __host__ __device__ inline T length() const;

    __host__ __device__ inline void normalize();
    __host__ __device__ inline Vector normalized() const;

    __host__ __device__ inline constexpr const T&
    operator[](const unsigned int pos) const;

    __host__ __device__ inline constexpr T& operator[](const unsigned int pos);

    __host__ __device__ inline bool operator==(const Vector& rhs) const;
    __host__ __device__ inline bool operator!=(const Vector& rhs) const;

    virtual ~Vector() = default;

    friend Vector<3, float> cross_product(const Vector<3, float>& lhs,
                                          const Vector<3, float>& rhs);
};

template <unsigned int size, typename T = float>
__host__ __device__ inline Vector<size, T>
operator*(const float scalar, const Vector<size, T>& vect)
{
    return vect * scalar;
}

using Vector3 = Vector<3, float>;
using Point3 = Vector3;

__host__ __device__ inline Vector3 cross_product(const Vector3& lhs,
                                                 const Vector3& rhs);

} // namespace space
#include "vector.cuhxx"