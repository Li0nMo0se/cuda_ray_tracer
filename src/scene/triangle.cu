#include "scene/triangle.cuh"

namespace scene
{
__device__ Triangle::Triangle(const space::Point3& A,
                              const space::Point3& B,
                              const space::Point3& C,
                              const color::TextureMaterial* const texture,
                              const space::Vector3& translation)
    : Object(texture, translation)
    , A_(A)
    , B_(B)
    , C_(C)
    , normal_(compute_normal())
    , opposite_normal_(-normal_)
{
}

__device__ void Triangle::translate()
{
    A_ += translation_;
    B_ += translation_;
    C_ += translation_;
    normal_ = compute_normal();
    opposite_normal_ = -normal_;
}

__device__ space::Vector3 Triangle::compute_normal() const
{
    const space::Vector3 AB = B_ - A_;
    const space::Vector3 AC = C_ - A_;
    return cross_product(AB, AC).normalized();
}

static __device__ inline float det_matrix(const float matrix[3][3])
{
    return matrix[0][0] *
               (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
           matrix[0][1] *
               (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] *
               (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

/* Use the Cramer’s Rule */
// Return a Vector3 as the solution of the equation is a matrix shape=(3,1)
// The equation might have infinite or zero solutions, thus return a
// cuda_tools::Optional
static __device__ cuda_tools::Optional<space::Vector3>
find_solution(const float coeff[3][4])
{
    // Build the 3 Cramer's matrices
    const float d[3][3] = {
        {coeff[0][0], coeff[0][1], coeff[0][2]},
        {coeff[1][0], coeff[1][1], coeff[1][2]},
        {coeff[2][0], coeff[2][1], coeff[2][2]},
    };
    const float d1[3][3] = {
        {coeff[0][3], coeff[0][1], coeff[0][2]},
        {coeff[1][3], coeff[1][1], coeff[1][2]},
        {coeff[2][3], coeff[2][1], coeff[2][2]},
    };
    const float d2[3][3] = {
        {coeff[0][0], coeff[0][3], coeff[0][2]},
        {coeff[1][0], coeff[1][3], coeff[1][2]},
        {coeff[2][0], coeff[2][3], coeff[2][2]},
    };
    const float d3[3][3] = {
        {coeff[0][0], coeff[0][1], coeff[0][3]},
        {coeff[1][0], coeff[1][1], coeff[1][3]},
        {coeff[2][0], coeff[2][1], coeff[2][3]},
    };

    // Calculate determinant of Cramer's matrix
    const float D = det_matrix(d);
    const float D1 = det_matrix(d1);
    const float D2 = det_matrix(d2);
    const float D3 = det_matrix(d3);

    if (D != 0)
        return space::Vector3(D1 / D, D2 / D, D3 / D);
    else // No solution or infinite solution
        return cuda_tools::nullopt;
}

__device__ cuda_tools::Optional<space::IntersectionInfo>
Triangle::intersect(const space::Ray& ray) const
{
    // Solve linera equation using Cramer's rule to find intersection
    float coeff[3][4];
    // Fill coeff
    const space::Vector3& ray_direction = ray.direction_get();
    coeff[0][3] = ray_direction[0];
    coeff[1][3] = ray_direction[1];
    coeff[2][3] = ray_direction[2];

    const space::Point3& O = ray.origin_get();
    const space::Vector3 OA_OB_OC[3] = {A_ - O, B_ - O, C_ - O};
    for (int i = 0; i < 3; ++i)
    {
        coeff[0][i] = OA_OB_OC[i][0];
        coeff[1][i] = OA_OB_OC[i][1];
        coeff[2][i] = OA_OB_OC[i][2];
    }

    const cuda_tools::Optional<space::Vector3> res = find_solution(coeff);
    if (!res) // No solution found
        return cuda_tools::nullopt;
    const space::Vector3 res_vect = res.value();
    const float alpha = res_vect[0];
    const float beta = res_vect[1];
    const float gamma = res_vect[2];

    // Check same sign
    const bool all_positive = alpha >= 0.f && beta >= 0.f && gamma >= 0.f;
    const bool all_negative = alpha <= 0.f && beta <= 0.f && gamma <= 0.f;

    // Intersection not inside triangle
    if (!all_negative && !all_positive)
        return cuda_tools::nullopt;

    // Compute OG
    const space::Vector3 OG =
        (alpha * OA_OB_OC[0] + beta * OA_OB_OC[1] + gamma * OA_OB_OC[2]) /
        (alpha + beta + gamma);

    // Find t
    // OG = tD i.e t = OG.x / D.x = OG.y / D.y = OG.z / D.z
    const float t = OG[0] / ray_direction[0];
    if (t < space::T_MIN)
        return cuda_tools::nullopt;
    return space::IntersectionInfo(t, *this);
}

__device__ space::Vector3
Triangle::normal_get(const space::Ray& ray,
                     const space::IntersectionInfo&) const
{
    // Consider a triangle as a plan (see plan.cc for more details)
    // Return normal or -normal according to the ray
    if (ray.direction_get().dot(normal_) > 0.f)
        return opposite_normal_;

    return normal_;
}
} // namespace scene