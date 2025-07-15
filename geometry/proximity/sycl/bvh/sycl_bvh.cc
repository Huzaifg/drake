#include "geometry/proximity/sycl/bvh/sycl_bvh.h"

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

#include "geometry/proximity/sycl/sycl_memory_manager.h"
#include <oneapi/dpl/execution>  // For execution policies
#include <oneapi/dpl/numeric>    // For exclusive_scan
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH
// construction
// this is a bottom-up clustering method that outputs one node per-leaf
// This class creates BVHs for all meshes in parallel
class BVHBuilder {
 public:
  BVHBuilder();
  ~BVHBuilder();

  // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for
  // each triangle
  void build(const DeviceMeshData& mesh_data,
             std::vector<Vector3<double>> sorted_total_lower,
             std::vector<Vector3<double>> sorted_total_upper,
             DeviceBVHData& bvh_data, SyclMemoryManager& memory_manager,
             sycl::queue& q_device);
};
// Construct and return BVH for all meshes in the scene
// They will be indexed by same order of sorted_geometry ids
BVHBuilder::build(const DeviceMeshData& mesh_data,
                  std::vector<Vector3<double>> sorted_total_lower,
                  std::vector<Vector3<double>> sorted_total_upper,
                  DeviceBVHData& bvh_data, SyclMemoryManager& memory_manager,
                  sycl::queue& q_device) {
  int num_geometries = sorted_total_lower.size();
  auto policy = oneapi::dpl::execution::make_device_policy(q_device);

  // Allocate memory for total lower and upper bounds
  bvh_data.total_lowerAll =
      memory_manager_.AllocateDevice<Vector3<double>>(num_geometries);
  bvh_data.total_upperAll =
      memory_manager_.AllocateDevice<Vector3<double>>(num_geometries);
  bvh_data.total_inv_edgesAll =
      memory_manager_.AllocateDevice<Vector3<double>>(num_geometries);

  // Copy total lower and upper bounds from host to device
  memory_manager_.CopyToDevice(bvh_data.total_lowerAll, &sorted_total_lower[0],
                               num_geometries);
  memory_manager_.CopyToDevice(bvh_data.total_upperAll, &sorted_total_upper[0],
                               num_geometries);

  // Compute inverse edges of the total AABB
  auto inv_edges_event = q_device.submit([&](sycl::handler& h) {
    h.parallel_for(
        num_geometries,
        [=, total_upperAll = bvh_data.total_upperAll,
         total_lowerAll = bvh_data.total_lowerAll,
         total_inv_edgesAll = bvh_data.total_inv_edgesAll](sycl::item<1> item) {
          int index = item.get_id(0);
          total_inv_edgesAll[index] =
              1.0 / (total_upperAll[index] - total_lowerAll[index]);
        });
  });

  // Initialize memory for BVH for each geometry on the device
  bvh_data.bvhAll = memory_manager.AllocateDevice<BVH>(num_geometries);

  // Compute the max nodes for each BVH
  auto max_nodes_event = q_device.submit([&](sycl::handler& h) {
    h.parallel_for(num_geometries,
                   [=, vertex_counts = mesh_data.vertex_counts,
                    bvhAll = bvh_data.bvhAll](sycl::item<1> item) {
                     int index = item.get_id(0);
                     int num_vertices = vertex_counts[index];
                     bvhAll[index].max_nodes = 2 * num_vertices - 1;
                   });
  });

  // Allocate memory for indices, keys, deltas, range_lefts, range_rights,
  // num_children
  bvh_data.indicesAll = memory_manager_.AllocateDevice<int>(num_tets);
  bvh_data.keysAll = memory_manager_.AllocateDevice<int>(num_tets);
  bvh_data.deltasAll = memory_manager_.AllocateDevice<int>(num_tets);
  bvh_data.range_leftsAll =
      memory_manager_.AllocateDevice<int>(bvh_data.max_nodes);
  bvh_data.range_rightsAll =
      memory_manager_.AllocateDevice<int>(bvh_data.max_nodes);
  bvh_data.num_childrenAll =
      memory_manager_.AllocateDevice<int>(bvh_data.max_nodes);

  sycl::event compute_morton_codes_event =
      q_device.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(num_tets), [=](sycl::item<1> item) {
          int index = item.get_id(0);
          int key = keys[index];
          int delta = deltas[index];
        });
      });

  // Build the BVH
  build_bvh(item_lowers, item_uppers, num_tets);
}
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake