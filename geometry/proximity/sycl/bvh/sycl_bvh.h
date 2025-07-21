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

#include "geometry/proximity/sycl/utils/sycl_kernel_utils.h"
#include "geometry/proximity/sycl/utils/sycl_memory_manager.h"
#include <sycl/sycl.hpp>

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Overload that takes Vector3<double> by value
SYCL_EXTERNAL inline BVHPackedNodeHalf make_node(const Vector3<double>& bound,
                                                 int child, bool leaf) {
  BVHPackedNodeHalf n;
  n.x = bound.x();
  n.y = bound.y();
  n.z = bound.z();
  n.i = static_cast<unsigned int>(child);
  n.b = static_cast<unsigned int>(leaf ? 1 : 0);

  return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
SYCL_EXTERNAL inline void make_node(volatile BVHPackedNodeHalf* n,
                                    const Vector3<double>& bound, int child,
                                    bool leaf) {
  n->x = bound.x();
  n->y = bound.y();
  n->z = bound.z();
  n->i = static_cast<unsigned int>(child);
  n->b = static_cast<unsigned int>(leaf ? 1 : 0);
}

// TODO - Can apparently be done more efficiently by loading as float4 and then
// converting to BVHPackedNodeHalf (according to Warp). Try later
SYCL_EXTERNAL inline BVHPackedNodeHalf bvh_load_node(
    const BVHPackedNodeHalf* nodes, int index) {
  return nodes[index];
}

// Interleaves with 2 zeroes between each bit
SYCL_EXTERNAL inline uint32_t part1by2(uint32_t n) {
  n = (n ^ (n << 16)) & 0xff0000ff;
  n = (n ^ (n << 8)) & 0x0300f00f;
  n = (n ^ (n << 4)) & 0x030c30c3;
  n = (n ^ (n << 2)) & 0x09249249;

  return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of
// length 3*lwp2(dim) bits
template <int dim>
SYCL_EXTERNAL inline uint32_t morton3(float x, float y, float z) {
  uint32_t ux = sycl::clamp(int(x * dim), 0, dim - 1);
  uint32_t uy = sycl::clamp(int(y * dim), 0, dim - 1);
  uint32_t uz = sycl::clamp(int(z * dim), 0, dim - 1);
  // If dim = 2014, then 10 bit + 10 bit + 10 bit = 30 bit Morton code
  return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

// Broad Phase
// Build the mesh BVH's if not already built otherwise just traverse it

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH
// construction
// this is a bottom-up clustering method that outputs one node per-leaf
// This class creates BVHs for all meshes in parallel
class BVHBroadPhase {
 public:
  BVHBroadPhase() = default;
  ~BVHBroadPhase() = default;

  sycl::event BroadPhase(const DeviceMeshData& mesh_data,
                         const std::vector<Vector3<double>>& sorted_total_lower,
                         const std::vector<Vector3<double>>& sorted_total_upper,
                         DeviceBVHData& bvh_data,
                         sycl::event& element_aabb_event,
                         SyclMemoryManager& memory_manager,
                         sycl::queue& q_device);
  // Construct and return BVH for all meshes in the scene
  // They will be indexed by same order of sorted_geometry ids
  // q is waited on becaue memory needs to be released
  void build(const DeviceMeshData& mesh_data,
             const std::vector<Vector3<double>>& sorted_total_lower,
             const std::vector<Vector3<double>>& sorted_total_upper,
             DeviceBVHData& bvh_data, sycl::event& element_aabb_event,
             SyclMemoryManager& memory_manager, sycl::queue& q_device);
  bool IsBVHBuilt() const { return bvh_built_; }

 private:
  // BVH construction parameters
  enum class BVHParams : int {
    kMinPrimitivesPerLeaf = 8,  // Minimum primitives before creating leaf
    kMaxDepth = 32              // Maximum tree depth before forcing leaf
  };

  bool bvh_built_ = false;
};

// Attorney class for accessing and inspecting SYCL BVH data in tests.
// Provides safe copying from device to host and computation of tree statistics.
// Assumes BVH is built using the linear BVH method (bottom-up clustering).
class SyclBvhAttorney {
 public:
  // Struct to hold a deep-copied host version of BVH data.
  struct HostBVH {
    std::vector<BVHPackedNodeHalf> node_lowers;
    std::vector<BVHPackedNodeHalf> node_uppers;
    std::vector<int> node_parents;
    int root_index;  // Dereferenced from device root pointer.
    int max_nodes;
    int num_nodes;
    int num_leaf_nodes;
    // Add other fields as needed (e.g., primitive_indices if required).
  };

  // Copies the BVH for a specific mesh_id from device to host.
  // Performs deep copies of all arrays. Waits on the queue if necessary.
  // Throws if mesh_id is invalid or data is inconsistent.
  static HostBVH GetHostBVH(const DeviceBVHData& device_data,
                            const int sorted_mesh_id,
                            SyclMemoryManager& mem_mgr, sycl::queue& q) {
    if (sorted_mesh_id < 0 ||
        static_cast<uint32_t>(sorted_mesh_id) >= device_data.num_meshes) {
      throw std::runtime_error("Invalid mesh_id: " +
                               std::to_string(sorted_mesh_id));
    }

    const BVH& device_bvh = device_data.bvhAll[sorted_mesh_id];
    HostBVH host_bvh;
    host_bvh.max_nodes = device_bvh.max_nodes;

    // TODO - Set these post build in GPU BVH
    // host_bvh.num_nodes =
    //     device_bvh.num_nodes;  // Assuming this is set post-build.
    // host_bvh.num_leaf_nodes =
    //     device_bvh.num_leaf_nodes;  // Assuming set post-build.

    // Copy arrays to host vectors.
    host_bvh.node_lowers.resize(device_bvh.max_nodes);
    mem_mgr.CopyToHost(host_bvh.node_lowers.data(), device_bvh.node_lowers,
                       device_bvh.max_nodes);

    host_bvh.node_uppers.resize(device_bvh.max_nodes);
    mem_mgr.CopyToHost(host_bvh.node_uppers.data(), device_bvh.node_uppers,
                       device_bvh.max_nodes);

    host_bvh.node_parents.resize(device_bvh.max_nodes);
    mem_mgr.CopyToHost(host_bvh.node_parents.data(), device_bvh.node_parents,
                       device_bvh.max_nodes);

    // Copy root index (single int).
    int device_root;
    mem_mgr.CopyToHost(&device_root, device_bvh.root, 1);
    host_bvh.root_index = device_root;

    // Wait for all copies to complete.
    q.wait_and_throw();

    return host_bvh;
  }

  // Computes the height (max depth) of the tree starting from root.
  // Height is the longest path from root to a leaf (edges).
  // Returns -1 if tree is invalid (e.g., cycles or bad structure).
  static int ComputeHeight(const HostBVH& host_bvh) {
    std::vector<bool> visited(host_bvh.max_nodes, false);
    return ComputeSubtreeHeight(host_bvh, host_bvh.root_index, visited);
  }

  // Counts the number of leaf nodes in the tree.
  static int CountLeaves(const HostBVH& host_bvh) {
    int count = 0;
    std::vector<bool> visited(host_bvh.max_nodes, false);
    CountLeavesRecursive(host_bvh, host_bvh.root_index, visited, &count);
    return count;
  }

  // Computes a simple balance factor: max(|left_height - right_height|) over
  // all nodes. Returns 0 for perfectly balanced; higher values indicate
  // imbalance. Returns -1 if tree is invalid.
  static int ComputeBalanceFactor(const HostBVH& host_bvh) {
    int max_imbalance = 0;
    std::vector<bool> visited(host_bvh.max_nodes, false);
    ComputeBalanceRecursive(host_bvh, host_bvh.root_index, visited,
                            &max_imbalance);
    return max_imbalance;
  }

  // Computes average depth across all leaves
  static double ComputeAverageLeafDepth(const HostBVH& host_bvh) {
    int total_depth = 0;
    int leaf_count = 0;
    std::vector<bool> visited(host_bvh.max_nodes, false);
    ComputeDepthsRecursive(host_bvh, host_bvh.root_index, visited, 0,
                           &total_depth, &leaf_count);
    return leaf_count > 0 ? static_cast<double>(total_depth) / leaf_count : 0.0;
  }

  // Verifies if node bounds are correct (e.g., union of children).
  // Returns true if valid for the whole tree.
  static bool VerifyBounds(const HostBVH& host_bvh) {
    std::vector<bool> visited(host_bvh.max_nodes, false);
    return VerifyBoundsRecursive(host_bvh, host_bvh.root_index, visited);
  }

  // Computes a histogram of imbalance factors (|left_height - right_height|)
  // for all internal nodes. Returns a vector where histogram[i] = number of
  // internal nodes with imbalance i. The size is max_imbalance + 1. Returns
  // empty vector if tree is invalid.
  static void ComputeAndPrintImbalanceHistogram(const HostBVH& host_bvh,
                                                const std::string& filepath) {
    auto heights = ComputeAllHeights(host_bvh);

    int max_diff = 0;
    std::vector<int> diffs;
    for (int i = 0; i < host_bvh.max_nodes; ++i) {
      if (heights[i] == -1) continue;
      const auto& lower = host_bvh.node_lowers[i];
      if (lower.b == 1) continue;  // Leaf
      int left_h = heights[lower.i];
      int right_h = heights[host_bvh.node_uppers[i].i];
      int diff = std::abs(left_h - right_h);
      diffs.push_back(diff);
      max_diff = std::max(max_diff, diff);
    }

    std::vector<int> histogram(max_diff + 1, 0);
    for (int d : diffs) {
      ++histogram[d];
    }
    PrintHistogramJSON(histogram, filepath);
  }

 private:
  // Helper to compute subtree height with cycle detection.
  static int ComputeSubtreeHeight(const HostBVH& host_bvh, int node_index,
                                  std::vector<bool>& visited) {
    if (node_index < 0 || node_index >= host_bvh.max_nodes ||
        visited[node_index]) {
      return -1;  // Invalid or cycle.
    }
    visited[node_index] = true;

    const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
    if (lower.b == 1) {  // Leaf (based on your BVH: b=1 for leaves).
      return 0;
    }

    // Assume binary tree: left child in lower.i, right in upper.i.
    const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
    int left_height = ComputeSubtreeHeight(host_bvh, lower.i, visited);
    int right_height = ComputeSubtreeHeight(host_bvh, upper.i, visited);
    if (left_height == -1 || right_height == -1) return -1;
    return 1 + std::max(left_height, right_height);
  }

  // Recursive helper for counting leaves with visited set.
  static void CountLeavesRecursive(const HostBVH& host_bvh, int node_index,
                                   std::vector<bool>& visited, int* count) {
    if (node_index < 0 || node_index >= host_bvh.max_nodes ||
        visited[node_index]) {
      return;
    }
    visited[node_index] = true;
    const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
    if (lower.b == 1) {  // Leaf.
      ++(*count);
      return;
    }
    const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
    CountLeavesRecursive(host_bvh, lower.i, visited, count);
    CountLeavesRecursive(host_bvh, upper.i, visited, count);
  }

  // Recursive helper for balance factor.
  static void ComputeBalanceRecursive(const HostBVH& host_bvh, int node_index,
                                      std::vector<bool>& visited,
                                      int* max_imbalance) {
    if (node_index < 0 || node_index >= host_bvh.max_nodes ||
        visited[node_index]) {
      return;
    }
    visited[node_index] = true;
    const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
    if (lower.b == 1) return;  // Leaf.

    const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
    std::vector<bool> left_visited = visited;  // Copy to avoid interference.
    std::vector<bool> right_visited = visited;
    int left_height = ComputeSubtreeHeight(host_bvh, lower.i, left_visited);
    int right_height = ComputeSubtreeHeight(host_bvh, upper.i, right_visited);
    if (left_height != -1 && right_height != -1) {
      *max_imbalance =
          std::max(*max_imbalance, std::abs(left_height - right_height));
    }
    ComputeBalanceRecursive(host_bvh, lower.i, visited, max_imbalance);
    ComputeBalanceRecursive(host_bvh, upper.i, visited, max_imbalance);
  }

  // Recursive helper for average leaf depth.
  static void ComputeDepthsRecursive(const HostBVH& host_bvh, int node_index,
                                     std::vector<bool>& visited,
                                     int current_depth, int* total_depth,
                                     int* leaf_count) {
    if (node_index < 0 || node_index >= host_bvh.max_nodes ||
        visited[node_index]) {
      return;
    }
    visited[node_index] = true;
    const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
    if (lower.b == 1) {  // Leaf.
      *total_depth += current_depth;
      ++(*leaf_count);
      return;
    }
    const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];
    ComputeDepthsRecursive(host_bvh, lower.i, visited, current_depth + 1,
                           total_depth, leaf_count);
    ComputeDepthsRecursive(host_bvh, upper.i, visited, current_depth + 1,
                           total_depth, leaf_count);
  }

  // Recursive helper to verify bounds (e.g., parent bounds == union of
  // children). Assumes Vector3<double> has cwiseMin/cwiseMax.
  static bool VerifyBoundsRecursive(const HostBVH& host_bvh, int node_index,
                                    std::vector<bool>& visited) {
    if (node_index < 0 || node_index >= host_bvh.max_nodes ||
        visited[node_index]) {
      return false;
    }
    visited[node_index] = true;
    const BVHPackedNodeHalf& lower = host_bvh.node_lowers[node_index];
    const BVHPackedNodeHalf& upper = host_bvh.node_uppers[node_index];

    Vector3<double> parent_lower(lower.x, lower.y, lower.z);
    Vector3<double> parent_upper(upper.x, upper.y, upper.z);

    if (lower.b == 1) {
      // leafs : Check the maximum primitves per leaf
      // get their AABBs and test if the node is the union of all the primitive
      // AABBs
      const unsigned int start_index = lower.i;
      const unsigned int end_index = upper.i;
      const unsigned int num_primitives = end_index - start_index;
      Vector3<double> obtained_lower(lower.x, lower.y, lower.z);
      Vector3<double> obtained_upper(upper.x, upper.y, upper.z);

      for (unsigned int i = start_index; i < end_index; ++i) {
        const BVHPackedNodeHalf& primitive_lower = host_bvh.node_lowers[i];
        const BVHPackedNodeHalf& primitive_upper = host_bvh.node_uppers[i];
        Vector3<double> current_lower(primitive_lower.x, primitive_lower.y,
                                      primitive_lower.z);
        Vector3<double> current_upper(primitive_upper.x, primitive_upper.y,
                                      primitive_upper.z);
        obtained_lower = obtained_lower.cwiseMin(current_lower);
        obtained_upper = obtained_upper.cwiseMax(current_upper);
      }
      // Check if parent bounds properly enclose the obtained bounds
      if (obtained_lower[0] < parent_lower[0] ||
          obtained_lower[1] < parent_lower[1] ||
          obtained_lower[2] < parent_lower[2] ||
          obtained_upper[0] > parent_upper[0] ||
          obtained_upper[1] > parent_upper[1] ||
          obtained_upper[2] > parent_upper[2]) {
        return false;
      }
      return true;
    }

    // Check left child.
    const BVHPackedNodeHalf& left_lower = host_bvh.node_lowers[lower.i];
    const BVHPackedNodeHalf& left_upper = host_bvh.node_uppers[lower.i];
    Vector3<double> left_min(left_lower.x, left_lower.y, left_lower.z);
    Vector3<double> left_max(left_upper.x, left_upper.y, left_upper.z);

    // Check right child.
    const BVHPackedNodeHalf& right_lower = host_bvh.node_lowers[upper.i];
    const BVHPackedNodeHalf& right_upper = host_bvh.node_uppers[upper.i];
    Vector3<double> right_min(right_lower.x, right_lower.y, right_lower.z);
    Vector3<double> right_max(right_upper.x, right_upper.y, right_upper.z);

    // Verify parent is union.
    Vector3<double> expected_min = left_min.cwiseMin(right_min);
    Vector3<double> expected_max = left_max.cwiseMax(right_max);
    bool bounds_valid =
        (parent_lower == expected_min) && (parent_upper == expected_max);

    return bounds_valid && VerifyBoundsRecursive(host_bvh, lower.i, visited) &&
           VerifyBoundsRecursive(host_bvh, upper.i, visited);
  }

  // Computes heights for all nodes using memoization.
  // Returns vector of heights, or empty if invalid.
  static std::vector<int> ComputeAllHeights(const HostBVH& host_bvh) {
    std::vector<int> heights(host_bvh.max_nodes, -1);
    if (!ComputeHeightMemo(host_bvh, host_bvh.root_index, heights)) {
      return {};
    }
    return heights;
  }

  // Recursive memoized height computation. Returns true if valid.
  static bool ComputeHeightMemo(const HostBVH& host_bvh, int node,
                                std::vector<int>& heights) {
    if (node < 0 || node >= host_bvh.max_nodes) return false;
    if (heights[node] != -1) return true;

    const auto& lower = host_bvh.node_lowers[node];
    if (lower.b == 1) {  // Leaf
      heights[node] = 0;
      return true;
    }

    if (!ComputeHeightMemo(host_bvh, lower.i, heights)) return false;
    if (!ComputeHeightMemo(host_bvh, host_bvh.node_uppers[node].i, heights))
      return false;

    int left_h = heights[lower.i];
    int right_h = heights[host_bvh.node_uppers[node].i];
    heights[node] = 1 + std::max(left_h, right_h);
    return true;
  }

  static void PrintHistogramJSON(const std::vector<int>& histogram,
                                 const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
      std::cerr << "Failed to open histogram.json" << std::endl;
      return;
    }
    file << "{" << std::endl;
    file << "  \"histogram\": [" << std::endl;
    for (size_t i = 0; i < histogram.size(); ++i) {
      file << "    " << histogram[i];
      if (i < histogram.size() - 1) {
        file << ",";
      }
      file << std::endl;
    }
    file << "  ]" << std::endl;
    file << "}" << std::endl;
    file.close();
  }
};
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake