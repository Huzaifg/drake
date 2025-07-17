#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <sycl/sycl.hpp>

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Helper class to manage SYCL device memory allocations and transfers
class SyclMemoryManager {
 public:
  explicit SyclMemoryManager(sycl::queue& queue) : queue_(queue) {}

  // Allocate device memory for basic types
  template <typename T>
  inline T* AllocateDevice(uint32_t count) {
    return sycl::malloc_device<T>(count, queue_);
  }

  // Allocate host-accessible memory for basic types
  template <typename T>
  inline T* AllocateHost(uint32_t count) {
    return sycl::malloc_host<T>(count, queue_);
  }

  // Free device/host memory
  template <typename T>
  inline void Free(T* ptr) {
    if (ptr != nullptr) {
      sycl::free(ptr, queue_);
    }
  }

  // Copy data from host to device
  template <typename T>
  inline sycl::event CopyToDevice(T* device_ptr, const T* host_ptr,
                                  uint32_t count) {
    return queue_.memcpy(device_ptr, host_ptr, count * sizeof(T));
  }

  // Copy data from device to host
  template <typename T>
  inline sycl::event CopyToHost(T* host_ptr, const T* device_ptr,
                                uint32_t count) {
    return queue_.memcpy(host_ptr, device_ptr, count * sizeof(T));
  }

  // Fill device memory with a value
  template <typename T>
  inline sycl::event Fill(T* device_ptr, const T& value, uint32_t count) {
    return queue_.fill(device_ptr, value, count);
  }

  // Memset device memory to zero
  template <typename T>
  inline sycl::event Memset(T* device_ptr, uint32_t count) {
    return queue_.memset(device_ptr, 0, count * sizeof(T));
  }

 private:
  sycl::queue& queue_;
};

// Structure to hold all device memory pointers for mesh data
struct DeviceMeshData {
  // Element data
  std::array<int, 4>* elements = nullptr;
  uint32_t* element_mesh_ids = nullptr;
  std::array<Vector3<double>, 4>* inward_normals_M = nullptr;
  std::array<Vector3<double>, 4>* inward_normals_W = nullptr;
  double* min_pressures = nullptr;
  double* max_pressures = nullptr;
  Vector4<double>* gradient_M_pressure_at_Mo = nullptr;
  Vector4<double>* gradient_W_pressure_at_Wo = nullptr;
  Vector3<double>* element_aabb_min_W = nullptr;
  Vector3<double>* element_aabb_max_W = nullptr;

  // Vertex data
  Vector3<double>* vertices_M = nullptr;
  Vector3<double>* vertices_W = nullptr;
  double* pressures = nullptr;
  uint32_t* vertex_mesh_ids = nullptr;

  // Lookup arrays (host accessible)
  uint32_t* element_offsets = nullptr;
  uint32_t* vertex_offsets = nullptr;
  uint32_t* element_counts = nullptr;
  uint32_t* vertex_counts = nullptr;
  GeometryId* geometry_ids = nullptr;
  double* transforms = nullptr;
  uint32_t total_elements;
  uint32_t total_vertices;
};

// Structure to hold collision detection memory
struct DeviceCollisionData {
  // Broad phase data
  uint8_t* collision_filter = nullptr;
  uint32_t* collision_filter_host_body_index = nullptr;
  uint32_t* total_checks_per_geometry = nullptr;
  uint32_t* geom_collision_filter_num_cols = nullptr;
  uint32_t* geom_collision_filter_check_offsets = nullptr;
  uint32_t* prefix_sum_total_checks = nullptr;

  // Narrow phase data
  uint32_t* narrow_phase_check_indices = nullptr;
  uint8_t* narrow_phase_check_validity = nullptr;
  uint32_t* prefix_sum_narrow_phase_checks = nullptr;
};

// Structure to hold polygon data memory
struct DevicePolygonData {
  // Raw polygon data
  double* polygon_areas = nullptr;
  Vector3<double>* polygon_centroids = nullptr;
  Vector3<double>* polygon_normals = nullptr;
  double* polygon_g_M = nullptr;
  double* polygon_g_N = nullptr;
  double* polygon_pressure_W = nullptr;
  GeometryId* polygon_geom_index_A = nullptr;
  GeometryId* polygon_geom_index_B = nullptr;

  // Compacted polygon data
  double* compacted_polygon_areas = nullptr;
  Vector3<double>* compacted_polygon_centroids = nullptr;
  Vector3<double>* compacted_polygon_normals = nullptr;
  double* compacted_polygon_g_M = nullptr;
  double* compacted_polygon_g_N = nullptr;
  double* compacted_polygon_pressure_W = nullptr;
  GeometryId* compacted_polygon_geom_index_A = nullptr;
  GeometryId* compacted_polygon_geom_index_B = nullptr;

  uint32_t* valid_polygon_indices = nullptr;

  // Debug data
  double* debug_polygon_vertices = nullptr;
};

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
  // Not owned by the BVH, just points to indicesAll in DeviceBVHData
  uint32_t* primitive_indices;

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
  Vector3<double>* item_lowers;
  Vector3<double>* item_uppers;
  int num_items;
};

struct DeviceBVHData {
  // Permenant data only deleted with the SYCL proximity engine
  BVH* bvhAll = nullptr;
  uint32_t* node_counts_per_mesh = nullptr;
  uint32_t* node_offsets = nullptr;
  uint32_t* node_mesh_ids = nullptr;
  Vector3<double>* total_lowerAll = nullptr;
  Vector3<double>* total_upperAll = nullptr;
  Vector3<double>* total_inv_edgesAll = nullptr;
  // This is modified in place to point to mesh local primitive index
  // If this is used again to get primitive AABBs from mesh_data, it needs the
  // mesh wise element offset added to it
  uint32_t* indicesAll = nullptr;

  // Temp data deleted after tree construction
  uint32_t* keysAll = nullptr;  // Morton keys of all elements
  uint32_t* deltasAll =
      nullptr;  // deltasAll[index] is the delta of key index and index+1
  uint32_t* range_leftsAll =
      nullptr;  // Each node stores the range of primitives it covers. This is
                // the left limit of the range
  uint32_t* range_rightsAll = nullptr;  // This is the right limit of the range
  uint32_t* num_childrenAll =
      nullptr;  // This is the number of children of the node
  uint32_t num_meshes;
  uint32_t total_nodes;
};

class SyclMemoryHelper {
 public:
  static void AllocateMeshMemory(SyclMemoryManager& mem_mgr,
                                 DeviceMeshData& mesh_data,
                                 uint32_t num_geometries);
  static void AllocateBVHPerMeshMemory(SyclMemoryManager& mem_mgr,
                                       DeviceBVHData& bvh_data,
                                       uint32_t num_geometries);
  static void AllocateBVHTempMemory(SyclMemoryManager& mem_mgr,
                                    DeviceBVHData& bvh_data,
                                    uint32_t total_elements);
  static void AllocateMeshElementVerticesMemory(SyclMemoryManager& mem_mgr,
                                                DeviceMeshData& mesh_data,
                                                uint32_t total_elements,
                                                uint32_t total_vertices);
  static void AllocateGeometryCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      uint32_t num_geometries);
  static void AllocateTotalChecksCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      uint32_t num_geometries);
  static void AllocateNarrowPhaseChecksCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
      uint32_t num_geometries);
  static void AllocateFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                        DevicePolygonData& polygon_data,
                                        uint32_t num_geometries);
  static void AllocateCompactPolygonMemory(SyclMemoryManager& mem_mgr,
                                           DevicePolygonData& polygon_data,
                                           uint32_t num_geometries);
  static void FreeMeshMemory(SyclMemoryManager& mem_mgr,
                             DeviceMeshData& mesh_data);
  static void FreeBVHMeshMemory(SyclMemoryManager& mem_mgr, BVH& bvh_mesh);
  static void FreeBVHNonTempMemory(SyclMemoryManager& mem_mgr,
                                   DeviceBVHData& bvh_data);
  static void FreeBVHTempMemory(SyclMemoryManager& mem_mgr,
                                DeviceBVHData& bvh_data);
  static void FreeCollisionMemory(SyclMemoryManager& mem_mgr,
                                  DeviceCollisionData& collision_data);
  static void FreeNarrowPhaseChecksCollisionMemory(
      SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data);
  static void FreeFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                    DevicePolygonData& polygon_data);
  static void FreeCompactPolygonMemory(SyclMemoryManager& mem_mgr,
                                       DevicePolygonData& polygon_data);
  static void FreePolygonMemory(SyclMemoryManager& mem_mgr,
                                DevicePolygonData& polygon_data);
};

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake