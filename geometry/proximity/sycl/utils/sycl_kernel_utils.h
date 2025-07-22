#pragma once

#include <sycl/sycl.hpp>

#include "drake/common/eigen_types.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Helper function to round up to nearest multiple of work group size
SYCL_EXTERNAL inline uint32_t RoundUpToWorkGroupSize(uint32_t n,
                                                     uint32_t work_group_size) {
  return ((n + work_group_size - 1) / work_group_size) * work_group_size;
}

// Returns the componentwise min and max of two Vector3<double>.
// The first element of the pair is the min, the second is the max.
SYCL_EXTERNAL inline Vector3<double> ComponentwiseMin(
    const Vector3<double>& a, const Vector3<double>& b) {
  Vector3<double> min_v;
  for (int i = 0; i < 3; ++i) {
    min_v[i] = sycl::min(a[i], b[i]);
  }
  return min_v;
}

// Returns the componentwise max of two Vector3<double>.
SYCL_EXTERNAL inline Vector3<double> ComponentwiseMax(
    const Vector3<double>& a, const Vector3<double>& b) {
  Vector3<double> max_v;
  for (int i = 0; i < 3; ++i) {
    max_v[i] = sycl::max(a[i], b[i]);
  }
  return max_v;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake