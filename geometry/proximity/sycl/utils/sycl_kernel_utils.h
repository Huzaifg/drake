#pragma once

#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Helper function to round up to nearest multiple of work group size
SYCL_EXTERNAL inline uint32_t RoundUpToWorkGroupSize(uint32_t n,
                                                     uint32_t work_group_size) {
  return ((n + work_group_size - 1) / work_group_size) * work_group_size;
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake