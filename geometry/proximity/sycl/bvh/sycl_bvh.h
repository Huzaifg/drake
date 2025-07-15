#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

using sycl_vec3f = sycl::vec<float, 3>;

#ifdef __SYCL_DEVICE_ONLY__
#define DRAKE_SYCL_DEVICE_INLINE [[sycl::device]]
#else
#define DRAKE_SYCL_DEVICE_INLINE
#endif

// Struct's for BVH broad phase implementation
// Reference: Warp (https://github.com/NVIDIA/warp/blob/main/warp/native/bvh.h)
struct BVHPackedNodeHalf {
  float x;
  float y;
  float z;
  // For non-leaf nodes:
  // - 'lower.i' represents the index of the left child node.
  // - 'upper.i' represents the index of the right child node.
  //
  // For leaf nodes:
  // - 'lower.i' indicates the start index of the primitives (AABB) in
  // 'primitive_indices'.
  // - 'upper.i' indicates the index just after the last primitive (AABB) in
  // 'primitive_indices'
  unsigned int i : 31;
  unsigned int b : 1;
};
struct BVH {
  BVHPackedNodeHalf* node_lowers;  // See BVHPackedNodeHalf for details
  BVHPackedNodeHalf* node_uppers;  // See BVHPackedNodeHalf for details

  // used for fast refits
  int* node_parents;
  int* node_counts;
  // reordered primitive indices corresponds to the ordering of leaf nodes
  int* primitive_indices;

  int max_depth;
  int max_nodes;
  int num_nodes;
  // since we use packed leaf nodes, the number of them is no longer the number
  // of items, but variable
  int num_leaf_nodes;

  // pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
  // representing the root of the tree, this is not always the first node
  // for bottom-up builders
  int* root;

  // item bounds are not owned by the BVH but by the caller
  sycl_vec3f* item_lowers;
  sycl_vec3f* item_uppers;
  int num_items;
};

DRAKE_SYCL_DEVICE_INLINE inline BVHPackedNodeHalf make_node(
    const sycl_vec3f* bound, int child, bool leaf) {
  BVHPackedNodeHalf n;
  n.x = bound->x();
  n.y = bound->y();
  n.z = bound->z();
  n.i = (unsigned int)child;
  n.b = (unsigned int)(leaf ? 1 : 0);

  return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
DRAKE_SYCL_DEVICE_INLINE inline void make_node(volatile BVHPackedNodeHalf* n,
                                               const sycl_vec3f* bound,
                                               int child, bool leaf) {
  n->x = bound->x();
  n->y = bound->y();
  n->z = bound->z();
  n->i = (unsigned int)child;
  n->b = (unsigned int)(leaf ? 1 : 0);
}

// TODO - Can apparently be done more efficiently by loading as float4 and then
// converting to BVHPackedNodeHalf (according to Warp). Try later
DRAKE_SYCL_DEVICE_INLINE inline BVHPackedNodeHalf bvh_load_node(
    const BVHPackedNodeHalf* nodes, int index) {
  return nodes[index];
}

// Interleaves with 2 zeroes between each bit
DRAKE_SYCL_DEVICE_INLINE inline uint32_t part1by2(uint32_t n) {
  n = (n ^ (n << 16)) & 0xff0000ff;
  n = (n ^ (n << 8)) & 0x0300f00f;
  n = (n ^ (n << 4)) & 0x030c30c3;
  n = (n ^ (n << 2)) & 0x09249249;

  return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of
// length 3*lwp2(dim) bits
template <int dim>
DRAKE_SYCL_DEVICE_INLINE inline uint32_t morton3(float x, float y, float z) {
  uint32_t ux = sycl::clamp(int(x * dim), 0, dim - 1);
  uint32_t uy = sycl::clamp(int(y * dim), 0, dim - 1);
  uint32_t uz = sycl::clamp(int(z * dim), 0, dim - 1);
  // If dim = 2014, then 10 bit + 10 bit + 10 bit = 30 bit Morton code
  return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake