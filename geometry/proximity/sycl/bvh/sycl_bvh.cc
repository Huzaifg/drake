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

#include <oneapi/dpl/algorithm>  // For sort_by_key
#include <oneapi/dpl/execution>  // For execution policies
#include <oneapi/dpl/numeric>    // For exclusive_scan
#include <sycl/sycl.hpp>

#include "drake/geometry/geometry_ids.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {
// Forward declarations for kernel names
class ComputeInvEdgesKernel;
class ComputeMortonCodesKernel;
class ComputeKeyDeltasKernel;
class BuildLeavesKernel;
class BuildTreeKernel;
class PackLeavesKernel;
class RefitKernel;

void BVHBroadPhase::build(
    const DeviceMeshData& mesh_data,
    const std::vector<Vector3<double>>& sorted_total_lower,
    const std::vector<Vector3<double>>& sorted_total_upper,
    DeviceBVHData& bvh_data, sycl::event& element_aabb_event,
    SyclMemoryManager& memory_manager, sycl::queue& q_device) {
  // Run all OneAPI algos on device
  auto policy = oneapi::dpl::execution::make_device_policy(q_device);
  int num_geometries = sorted_total_lower.size();

  // Copy the mesh wise AABB lower and upper to the device
  auto copy_mesh_aabb_event1 = memory_manager.CopyToDevice(
      bvh_data.total_lowerAll, sorted_total_lower.data(), num_geometries);
  auto copy_mesh_aabb_event2 = memory_manager.CopyToDevice(
      bvh_data.total_upperAll, sorted_total_upper.data(), num_geometries);

  std::vector<int> mesh_ids(num_geometries);
  std::iota(mesh_ids.begin(), mesh_ids.end(), 0);

  // // Compute the total bounds of the meshes
  // std::vector<sycl::event> total_bound_events;
  // for (int mesh_id : mesh_ids) {
  //   uint32_t this_geom_start = mesh_data.element_offsets[mesh_id];
  //   uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
  //   auto element_aabb_min_W = mesh_data.element_aabb_min_W + this_geom_start;
  //   auto element_aabb_max_W = mesh_data.element_aabb_max_W + this_geom_start;
  //   auto total_lowerAll = bvh_data.total_lowerAll + mesh_id;
  //   auto total_upperAll = bvh_data.total_upperAll + mesh_id;
  // }

  // Compute inverse edges of all mesh level AABBs
  // This is used to compute the Morton Codes
  auto inv_edges_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({copy_mesh_aabb_event1, copy_mesh_aabb_event2});
    h.parallel_for<ComputeInvEdgesKernel>(
        num_geometries,
        [=, total_upperAll = bvh_data.total_upperAll,
         total_lowerAll = bvh_data.total_lowerAll,
         total_inv_edgesAll = bvh_data.total_inv_edgesAll](sycl::item<1> item)
            [[intel::kernel_args_restrict]] {
              int index = item.get_id(0);
              Vector3<double> edge =
                  (total_upperAll[index] - total_lowerAll[index]);
              edge += Vector3<double>(1e-6, 1e-6, 1e-6);
              total_inv_edgesAll[index] =
                  Vector3<double>(1.0 / edge[0], 1.0 / edge[1], 1.0 / edge[2]);
            });
  });

  // Now we compute the morton codes for all the elements in all the meshes
  // simultaneously
  // For this we need our AABBs
  auto compute_morton_codes_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({element_aabb_event, inv_edges_event});
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.parallel_for<ComputeMortonCodesKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, total_elements = mesh_data.total_elements,
         total_lowerAll = bvh_data.total_lowerAll,
         total_inv_edgesAll = bvh_data.total_inv_edgesAll,
         element_mesh_ids = mesh_data.element_mesh_ids,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         keysAll = bvh_data.keysAll] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);  // Global Element Index
          if (global_eI < total_elements) {
            Vector3<double> min_W = element_aabb_min_W[global_eI];
            Vector3<double> max_W = element_aabb_max_W[global_eI];
            Vector3<double> center_W = (min_W + max_W) / 2.0;

            uint32_t geom_index =
                element_mesh_ids[global_eI];  // Geometry this Index
                                              // belongs to
            Vector3<double> geom_inv_edges = total_inv_edgesAll[geom_index];
            Vector3<double> geom_lower_W = total_lowerAll[geom_index];

            // Normalize
            float local_x = (center_W[0] - geom_lower_W[0]) * geom_inv_edges[0];
            float local_y = (center_W[1] - geom_lower_W[1]) * geom_inv_edges[1];
            float local_z = (center_W[2] - geom_lower_W[2]) * geom_inv_edges[2];

            // 10-bit Morton codes stored in lower 30bits (1024^3 effective
            // resolution)
            uint32_t key = morton3<1024>(local_x, local_y, local_z);

            indicesAll[global_eI] = global_eI;
            keysAll[global_eI] = key;
          }
        });
  });

  // Wait for this event since we need to sort based on it
  compute_morton_codes_event.wait_and_throw();

  // oneapi::dpl::for_each does not work
  // ================================
  // seg sort
  // ================================
  for (int mesh_id : mesh_ids) {
    uint32_t this_geom_start = mesh_data.element_offsets[mesh_id];
    uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
    auto keys_begin = bvh_data.keysAll + this_geom_start;
    auto keys_end = keys_begin + this_geom_count;
    auto indices_begin = bvh_data.indicesAll + this_geom_start;
    oneapi::dpl::sort_by_key(policy, keys_begin, keys_end, indices_begin);
    // BVH.primitive_indices stores the index of where to find the AABB of that
    // primitive. We just make it point to the right place in indicesAll
    bvh_data.bvhAll[mesh_id].primitive_indices =
        bvh_data.indicesAll + this_geom_start;
  }
  q_device.wait();

  // Launch the compute key deltas between nearby keys
  auto compute_key_deltas_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.parallel_for<ComputeKeyDeltasKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, keysAll = bvh_data.keysAll,
         total_elements = mesh_data.total_elements,
         element_mesh_ids = mesh_data.element_mesh_ids,
         deltasAll = bvh_data.deltasAll] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < total_elements - 1) {
            uint32_t key = keysAll[global_eI];
            uint32_t next_key = keysAll[global_eI + 1];
            // We need to ignore next key if it is different mesh from key
            if (element_mesh_ids[global_eI] ==
                element_mesh_ids[global_eI + 1]) {
              uint32_t delta = key ^ next_key;
              // No clz because we are always comparing
              // left and right keys and never require the
              // absolute key value
              deltasAll[global_eI] = delta;  //__clz(delta)
            }
          }
        });
  });

  // For this we need to go back to mesh local indices and we do this
  //  by using the offsets array to subtract the sorted index values We also
  //  need the per mesh element counts to assign the right amount of memory
  for (int mesh_id : mesh_ids) {
    uint32_t this_geom_start = mesh_data.element_offsets[mesh_id];
    uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
    auto primitive_indices_begin = bvh_data.bvhAll[mesh_id].primitive_indices;
    auto primitive_indices_end = primitive_indices_begin + this_geom_count;
    oneapi::dpl::transform(
        policy, primitive_indices_begin, primitive_indices_end,
        primitive_indices_begin,
        [=, element_offsets = mesh_data.element_offsets](uint32_t val) {
          return val - element_offsets[mesh_id];
        });
  }

  for (int mesh_id : mesh_ids) {
    // Also allocate memory for the nodes of each BVH
    // Another loop for this so that its non blocking on the host with the above
    // transform
    uint32_t this_geom_count = mesh_data.element_counts[mesh_id];
    uint32_t max_nodes = 2 * this_geom_count - 1;
    bvh_data.bvhAll[mesh_id].max_nodes = max_nodes;
    SyclMemoryHelper::AllocateBVHSingleMeshMemory(
        memory_manager, bvh_data.bvhAll[mesh_id], max_nodes);

    // Initialize parent arrays to -1 (no parent initially)
    q_device.fill(bvh_data.bvhAll[mesh_id].node_parents, -1, max_nodes);
  }
  q_device.wait();

  // Initialize the global num_childrenAll array to 0 for atomic operations
  auto init_children_event =
      memory_manager.Memset(bvh_data.num_childrenAll, bvh_data.total_nodes);
  init_children_event.wait();

  // Build the leaves of the BVH
  // Most of the complication is making the mesh local index to the global index
  // and vice versa General idea is to use local indexing for anything that is
  // used by or goes into bvh_data.bvhAll
  //
  // IMPORTANT: INDEXING SCHEME for range_leftsAll and range_rightsAll
  // ================================================================
  // These arrays store ranges for both leaf and internal nodes using a
  // consistent global node indexing scheme:
  //
  // Layout: [mesh0_nodes(leaves+internal), mesh1_nodes(leaves+internal), ...]
  //
  // For each mesh:
  // - Leaves are stored at: global_node_offset + [0, mesh_element_count-1]
  // - Internal nodes at: global_node_offset + [mesh_element_count,
  // 2*mesh_element_count-2]
  //
  // This ensures both build_leaves and build_tree kernels use the same indexing
  // pattern: global_node_offset + local_index
  // ================================================================
  auto build_leaves_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.parallel_for<BuildLeavesKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, indicesAll = bvh_data.indicesAll, bvhAll = bvh_data.bvhAll,
         total_elements = mesh_data.total_elements,
         element_mesh_ids = mesh_data.element_mesh_ids,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         element_offsets = mesh_data.element_offsets,
         range_leftsAll = bvh_data.range_leftsAll,
         range_rightsAll = bvh_data.range_rightsAll,
         node_offsets = bvh_data.node_offsets] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < total_elements) {
            const uint32_t mesh_id = element_mesh_ids[global_eI];
            const uint32_t geom_local_primitive_index = indicesAll[global_eI];
            const uint32_t ele_offset = element_offsets[mesh_id];
            const uint32_t local_array_indexer = global_eI - ele_offset;
            const uint32_t global_node_offset = node_offsets[mesh_id];

            Vector3<double> lower_W =
                element_aabb_min_W[geom_local_primitive_index + ele_offset];
            Vector3<double> upper_W =
                element_aabb_max_W[geom_local_primitive_index + ele_offset];

            // Create the nodes
            // node_lowers and node_uppers store the nodes in sorted order of
            // morton code
            bvhAll[mesh_id].node_lowers[local_array_indexer] =
                make_node(lower_W, geom_local_primitive_index, true);
            bvhAll[mesh_id].node_uppers[local_array_indexer] =
                make_node(upper_W, geom_local_primitive_index, false);

            // Write leaf key ranges
            // Store ranges using global node indexing (mesh-local leaf index +
            // global node offset)
            uint32_t global_leaf_range_index =
                global_node_offset + local_array_indexer;
            range_leftsAll[global_leaf_range_index] = local_array_indexer;
            range_rightsAll[global_leaf_range_index] = local_array_indexer;
          }
        });
  });

  // Build the entire tree hierarchicy and update the internal node bounds
  auto build_tree_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 512;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
    h.depends_on({build_leaves_event});
    h.parallel_for<BuildTreeKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, bvhAll = bvh_data.bvhAll, total_elements = mesh_data.total_elements,
         element_mesh_ids = mesh_data.element_mesh_ids,
         indicesAll = bvh_data.indicesAll, deltasAll = bvh_data.deltasAll,
         range_leftsAll = bvh_data.range_leftsAll,
         range_rightsAll = bvh_data.range_rightsAll,
         num_childrenAll = bvh_data.num_childrenAll,
         element_offsets = mesh_data.element_offsets,
         element_counts = mesh_data.element_counts,
         node_offsets = bvh_data.node_offsets] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < total_elements) {
            const uint32_t mesh_id = element_mesh_ids[global_eI];
            const uint32_t ele_offset = element_offsets[mesh_id];
            const uint32_t mesh_element_count = element_counts[mesh_id];
            const uint32_t local_leaf_index = global_eI - ele_offset;
            const uint32_t internal_offset =
                mesh_element_count;  // Internal nodes start after leaves
            const uint32_t global_node_offset =
                node_offsets[mesh_id];  // Global offset for this mesh's nodes

            // Current node index (starts as leaf, local to this mesh) and acts
            // on sorted arrays
            uint32_t current_index = local_leaf_index;

            for (;;) {
              // Get range for current node using consistent global node
              // indexing Both leaves and internal nodes use: global_node_offset
              // + current_index
              uint32_t global_range_index = global_node_offset + current_index;

              // (left, right) is the range of prrimitives contained within the
              // current node left and right are mesh local primitive indices
              // They can only be used to point to morton code sorted arrays
              // (see build_leaves and how they are stored for the leaves)
              int left = range_leftsAll[global_range_index];
              int right = range_rightsAll[global_range_index];

              // Check if we are the root node for this mesh
              if (left == 0 && right == mesh_element_count - 1) {
                // Set root for this mesh - root is a pointer to int, so we
                // dereference it
                *bvhAll[mesh_id].root = current_index;
                bvhAll[mesh_id].node_parents[current_index] =
                    -1;  // Root has no parent
                break;
              }

              // Determine parents
              int child_count = 0;
              uint32_t parent_local_index;
              bool parent_right = false;

              // If right delta is smaller merge to right
              // If both deltas are equal, do a coin flip ( decision is made
              // using the XOR result of whether the keys before and after the
              // internal node are divisible by 2) to promote balanced tree
              // Otherwise merge to left For extremums (right most or left most
              // node of mesh) merge the other direction
              if (left == 0) {
                parent_right = true;  // No left neighbor, must group with right
              } else if (right != mesh_element_count - 1) {
                // Need to map local indices back to global for deltasAll access
                // deltasAll is indexed by global element indices
                uint32_t right_global = ele_offset + right;
                uint32_t left_minus_1_global = ele_offset + left - 1;

                if (deltasAll[right_global] <= deltasAll[left_minus_1_global]) {
                  if (deltasAll[right_global] ==
                      deltasAll[left_minus_1_global]) {
                    // Tie breaking using primitive indices (global indices)
                    parent_right = (indicesAll[left_minus_1_global] % 2) ^
                                   (indicesAll[right_global] % 2);
                  } else {
                    parent_right = true;
                  }
                }
              }

              // Assign to parent and update parent's range
              if (parent_right) {
                parent_local_index = right + internal_offset;

                // Set parent's left child
                bvhAll[mesh_id].node_parents[current_index] =
                    parent_local_index;
                bvhAll[mesh_id].node_lowers[parent_local_index].i =
                    current_index;

                // Update parent's left range in global arrays
                uint32_t parent_global_index =
                    global_node_offset + parent_local_index;
                range_leftsAll[parent_global_index] = left;

                // Memory fence to ensure writes are visible before atomic
                // increment
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::device);

                // Atomic increment of child count for this parent
                sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device>
                    atomic_children(num_childrenAll[parent_global_index]);
                child_count =
                    atomic_children.fetch_add(1, sycl::memory_order::acq_rel);
              } else {
                parent_local_index = left + internal_offset - 1;

                // Set parent's right child
                bvhAll[mesh_id].node_parents[current_index] =
                    parent_local_index;
                bvhAll[mesh_id].node_uppers[parent_local_index].i =
                    current_index;

                // Update parent's right range in global arrays
                uint32_t parent_global_index =
                    global_node_offset + parent_local_index;
                range_rightsAll[parent_global_index] = right;

                // Memory fence to ensure writes are visible before atomic
                // increment
                sycl::atomic_fence(sycl::memory_order::acq_rel,
                                   sycl::memory_scope::device);

                // Atomic increment of child count for this parent
                sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device>
                    atomic_children(num_childrenAll[parent_global_index]);
                child_count =
                    atomic_children.fetch_add(1, sycl::memory_order::acq_rel);
              }

              // If we're the second child (completing the parent), update
              // bounds and continue
              if (child_count == 1) {
                // Get child indices
                const uint32_t left_child =
                    bvhAll[mesh_id].node_lowers[parent_local_index].i;
                const uint32_t right_child =
                    bvhAll[mesh_id].node_uppers[parent_local_index].i;

                // Get child bounds
                Vector3<double> left_lower(
                    bvhAll[mesh_id].node_lowers[left_child].x,
                    bvhAll[mesh_id].node_lowers[left_child].y,
                    bvhAll[mesh_id].node_lowers[left_child].z);
                Vector3<double> left_upper(
                    bvhAll[mesh_id].node_uppers[left_child].x,
                    bvhAll[mesh_id].node_uppers[left_child].y,
                    bvhAll[mesh_id].node_uppers[left_child].z);
                Vector3<double> right_lower(
                    bvhAll[mesh_id].node_lowers[right_child].x,
                    bvhAll[mesh_id].node_lowers[right_child].y,
                    bvhAll[mesh_id].node_lowers[right_child].z);
                Vector3<double> right_upper(
                    bvhAll[mesh_id].node_uppers[right_child].x,
                    bvhAll[mesh_id].node_uppers[right_child].y,
                    bvhAll[mesh_id].node_uppers[right_child].z);

                // Compute union bounds
                Vector3<double> lower(sycl::min(left_lower[0], right_lower[0]),
                                      sycl::min(left_lower[1], right_lower[1]),
                                      sycl::min(left_lower[2], right_lower[2]));
                Vector3<double> upper(sycl::max(left_upper[0], right_upper[0]),
                                      sycl::max(left_upper[1], right_upper[1]),
                                      sycl::max(left_upper[2], right_upper[2]));

                // Update parent nodes using the volatile version of make_node
                // for synchronization
                make_node((volatile BVHPackedNodeHalf*)&bvhAll[mesh_id]
                              .node_lowers[parent_local_index],
                          lower, left_child, false);
                make_node((volatile BVHPackedNodeHalf*)&bvhAll[mesh_id]
                              .node_uppers[parent_local_index],
                          upper, right_child, false);

                // Move up to process the parent
                current_index = parent_local_index;
              } else {
                // We're the first child, parent not ready yet - terminate this
                // thread
                break;
              }
            }
          }
        });
  });

  // Finally, lets pack the leaf nodes based on threshold number of minimuim
  // primitives it must contain This is done to reduce the height of the tree
  // which will help speed up tree traversal
  auto pack_leaves_event = q_device.submit([&](sycl::handler& h) {
    const uint32_t work_group_size = 1024;
    const uint32_t global_elements =
        RoundUpToWorkGroupSize(bvh_data.total_nodes, work_group_size);
    h.depends_on({build_tree_event});
    h.parallel_for<PackLeavesKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, bvhAll = bvh_data.bvhAll, range_leftsAll = bvh_data.range_leftsAll,
         range_rightsAll = bvh_data.range_rightsAll,
         node_mesh_ids =
             bvh_data.node_mesh_ids] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_node_index = item.get_global_id(0);
          if (global_node_index < bvh_data.total_nodes) {
            uint32_t mesh_id = node_mesh_ids[global_node_index];
            uint32_t local_node_index =
                global_node_index - bvh_data.node_offsets[mesh_id];
            int depth = 1;
            int parent = bvhAll[mesh_id].node_parents[local_node_index];

            while (parent != -1) {
              int old_parent = parent;
              parent = bvhAll[mesh_id].node_parents[parent];
              depth++;
            }

            // converts LBB range left <= i <= right
            // to convention: left <= i < right (copied from Warp for now)
            int left = range_leftsAll[global_node_index];
            int right = range_rightsAll[global_node_index] + 1;

            if (right - left <=
                    static_cast<int>(BVHParams::kMinPrimitivesPerLeaf) ||
                depth >= static_cast<int>(BVHParams::kMaxDepth)) {
              bvh_data.bvhAll[mesh_id].node_lowers[local_node_index].b =
                  1;  // Make leaf
              // Set leaf ranges
              bvh_data.bvhAll[mesh_id].node_lowers[local_node_index].i = left;
              bvh_data.bvhAll[mesh_id].node_uppers[local_node_index].i = right;
            }
          }
        });
  });

  q_device.wait();

  SyclMemoryHelper::FreeBVHAllMeshTempMemory(memory_manager, bvh_data);
  bvh_built_ = true;
}

sycl::event BVHBroadPhase::refit(const DeviceMeshData& mesh_data,
                                 DeviceBVHData& bvh_data,
                                 sycl::event& element_aabb_event,
                                 SyclMemoryManager& memory_manager,
                                 sycl::queue& q_device) {
  // Reinitialize the num_childrenAll array to 0 for atomic operations
  auto init_children_event =
      memory_manager.Memset(bvh_data.num_childrenAll, bvh_data.total_nodes);

  // Reinitialize the indicesAll array to 0 for atomic operations
  const uint32_t work_group_size = 1024;
  const uint32_t global_elements =
      RoundUpToWorkGroupSize(mesh_data.total_elements, work_group_size);
  auto refit_event = q_device.submit([&](sycl::handler& h) {
    h.depends_on({init_children_event, element_aabb_event});
    h.parallel_for<RefitKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_elements),
                          sycl::range<1>(work_group_size)),
        [=, element_offsets = mesh_data.element_offsets,
         element_aabb_min_W = mesh_data.element_aabb_min_W,
         element_aabb_max_W = mesh_data.element_aabb_max_W,
         indicesAll = bvh_data.indicesAll,
         element_mesh_ids = mesh_data.element_mesh_ids,
         bvhAll = bvh_data.bvhAll, num_childrenAll = bvh_data.num_childrenAll,
         node_offsets = bvh_data.node_offsets] [[intel::kernel_args_restrict]] (
            sycl::nd_item<1> item) {
          uint32_t global_eI = item.get_global_id(0);
          if (global_eI < mesh_data.total_elements) {
            uint32_t mesh_id = element_mesh_ids[global_eI];
            uint32_t global_element_offset = element_offsets[mesh_id];
            uint32_t global_node_offset = node_offsets[mesh_id];
            uint32_t local_element_index = global_eI - global_element_offset;
            BVH& bvh = bvhAll[mesh_id];
            bool is_leaf = bvh.node_lowers[local_element_index].b;
            int parent = bvh.node_parents[local_element_index];

            if (!is_leaf) {
              return;
            }
            // Set new bounding boxes for the leaf
            BVHPackedNodeHalf& lower = bvh.node_lowers[local_element_index];
            BVHPackedNodeHalf& upper = bvh.node_uppers[local_element_index];

            // Only set these new bounding boxes if the leaf is not a muted leaf
            // (parent of leaf has not been made leaf in packing)
            if (!bvh.node_lowers[parent].b) {
              const uint32_t start = lower.i;
              const uint32_t end = upper.i;
              // Compute new AABB
              Vector3<double> lower_W(std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max());
              Vector3<double> upper_W(std::numeric_limits<double>::min(),
                                      std::numeric_limits<double>::min(),
                                      std::numeric_limits<double>::min());
              for (uint32_t local_primitive_index = start;
                   local_primitive_index < end; local_primitive_index++) {
                uint32_t unsorted_local_primitive_index =
                    indicesAll[local_primitive_index + global_element_offset];
                Vector3<double> lower_W_i =
                    element_aabb_min_W[unsorted_local_primitive_index +
                                       global_element_offset];
                Vector3<double> upper_W_i =
                    element_aabb_max_W[unsorted_local_primitive_index +
                                       global_element_offset];
                lower_W = ComponentwiseMin(lower_W, lower_W_i);
                upper_W = ComponentwiseMax(upper_W, upper_W_i);
              }
              // Set the new bounds for the leaf
              lower.x = lower_W[0];
              lower.y = lower_W[1];
              lower.z = lower_W[2];
              upper.x = upper_W[0];
              upper.y = upper_W[1];
              upper.z = upper_W[2];
            }

            // Now update hierarchy by moving upwards
            while (parent != -1) {
              uint32_t parent_global_index = global_node_offset + parent;
              sycl::atomic_fence(sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device);
              sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                               sycl::memory_scope::device>
                  atomic_children(num_childrenAll[parent_global_index]);
              int finished =
                  atomic_children.fetch_add(1, sycl::memory_order::acq_rel);

              if (finished == 1) {
                BVHPackedNodeHalf& parent_lower = bvh.node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = bvh.node_uppers[parent];
                if (parent_lower.b) {
                  // a packed leaf node can still be a parent in LBVH, we need
                  // to recompute its bounds since we've lost its left and right
                  // child node index in the muting process
                  int parent_parent = bvh.node_parents[parent];

                  // Parent should also not a leaf, otherwise we are in muted
                  // section
                  if (parent_parent != -1 &&
                      !bvh.node_lowers[parent_parent].b) {
                    const uint32_t start = parent_lower.i;
                    const uint32_t end = parent_upper.i;
                    Vector3<double> lower_W(std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::max());
                    Vector3<double> upper_W(std::numeric_limits<double>::min(),
                                            std::numeric_limits<double>::min(),
                                            std::numeric_limits<double>::min());
                    for (uint32_t local_primitive_index = start;
                         local_primitive_index < end; local_primitive_index++) {
                      uint32_t unsorted_local_primitive_index =
                          indicesAll[local_primitive_index +
                                     global_element_offset];
                      Vector3<double> lower_W_i =
                          element_aabb_min_W[unsorted_local_primitive_index +
                                             global_element_offset];
                      Vector3<double> upper_W_i =
                          element_aabb_max_W[unsorted_local_primitive_index +
                                             global_element_offset];
                      lower_W = ComponentwiseMin(lower_W, lower_W_i);
                      upper_W = ComponentwiseMax(upper_W, upper_W_i);
                    }
                    parent_lower.x = lower_W[0];
                    parent_lower.y = lower_W[1];
                    parent_lower.z = lower_W[2];
                    parent_upper.x = upper_W[0];
                    parent_upper.y = upper_W[1];
                    parent_upper.z = upper_W[2];
                  }
                } else {
                  // Parent is not a leaf so we recompute its bounds from its
                  // left and right children
                  const uint32_t left = parent_lower.i;
                  const uint32_t right = parent_upper.i;

                  Vector3<double> left_lower(bvh.node_lowers[left].x,
                                             bvh.node_lowers[left].y,
                                             bvh.node_lowers[left].z);
                  Vector3<double> left_upper(bvh.node_uppers[left].x,
                                             bvh.node_uppers[left].y,
                                             bvh.node_uppers[left].z);
                  Vector3<double> right_lower(bvh.node_lowers[right].x,
                                              bvh.node_lowers[right].y,
                                              bvh.node_lowers[right].z);
                  Vector3<double> right_upper(bvh.node_uppers[right].x,
                                              bvh.node_uppers[right].y,
                                              bvh.node_uppers[right].z);
                  Vector3<double> lower_W =
                      ComponentwiseMin(left_lower, right_lower);
                  Vector3<double> upper_W =
                      ComponentwiseMax(left_upper, right_upper);

                  // Set the new bounds for the parent
                  parent_lower.x = lower_W[0];
                  parent_lower.y = lower_W[1];
                  parent_lower.z = lower_W[2];
                  parent_upper.x = upper_W[0];
                  parent_upper.y = upper_W[1];
                  parent_upper.z = upper_W[2];
                }
                // Move up to process the parent
                parent = bvh.node_parents[parent];
              } else {
                break;
              }
            }
          }
        });
  });
  return refit_event;
}

sycl::event BVHBroadPhase::BroadPhase(
    const DeviceMeshData& mesh_data,
    const std::vector<Vector3<double>>& sorted_total_lower,
    const std::vector<Vector3<double>>& sorted_total_upper,
    DeviceBVHData& bvh_data, sycl::event& element_aabb_event,
    SyclMemoryManager& memory_manager, sycl::queue& q_device) {
  // Run a refit with the new AABBs - If the BVH is just built, then we don't
  // need to refit on the very first time step
  // By default, we refit every single mesh every time step because otherwise
  // the number of nodes will be too little for the GPU
  // TODO(Huzaifa): Refit only colliding meshes and compare the performance
  if (IsBVHRefitted()) {
    auto refit_event = refit(mesh_data, bvh_data, element_aabb_event,
                             memory_manager, q_device);
  }
  bvh_refitted_ = false;

  return sycl::event();
}
}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake