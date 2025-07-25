
#include "geometry/proximity/sycl/utils/sycl_memory_manager.h"

#include "drake/common/eigen_types.h"
#include "drake/geometry/geometry_ids.h"

namespace drake {
namespace geometry {
namespace internal {
namespace sycl_impl {

// Allocate all mesh-related device memory
void SyclMemoryHelper::AllocateMeshMemory(SyclMemoryManager& mem_mgr,
                                          DeviceMeshData& mesh_data,
                                          uint32_t num_geometries) {
  // Allocate lookup arrays (host accessible)
  mesh_data.element_offsets = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.vertex_offsets = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.element_counts = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.vertex_counts = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  mesh_data.geometry_ids = mem_mgr.AllocateHost<GeometryId>(num_geometries);
  mesh_data.transforms = mem_mgr.AllocateHost<double>(num_geometries * 12);
}

void SyclMemoryHelper::AllocateBVHSingleMeshMemory(SyclMemoryManager& mem_mgr,
                                                   BVH& bvh_mesh,
                                                   uint32_t max_nodes) {
  bvh_mesh.node_lowers = mem_mgr.AllocateDevice<BVHPackedNodeHalf>(max_nodes);
  bvh_mesh.node_uppers = mem_mgr.AllocateDevice<BVHPackedNodeHalf>(max_nodes);
  bvh_mesh.node_parents = mem_mgr.AllocateDevice<int>(max_nodes);
  bvh_mesh.root = mem_mgr.AllocateDevice<int>(1);
}

void SyclMemoryHelper::AllocateBVHAllMeshMemory(SyclMemoryManager& mem_mgr,
                                                DeviceBVHData& bvh_data,
                                                uint32_t num_geometries) {
  // The BVH counter data and pointers on Host but the data pointed to by BVH
  // will be on device
  bvh_data.bvhAll = mem_mgr.AllocateHost<BVH>(num_geometries);
  bvh_data.node_counts_per_mesh =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
  bvh_data.node_offsets = mem_mgr.AllocateHost<uint32_t>(num_geometries);
  bvh_data.total_lowerAll =
      mem_mgr.AllocateDevice<Vector3<double>>(num_geometries);
  bvh_data.total_upperAll =
      mem_mgr.AllocateDevice<Vector3<double>>(num_geometries);
  bvh_data.total_inv_edgesAll =
      mem_mgr.AllocateDevice<Vector3<double>>(num_geometries);
}

void SyclMemoryHelper::AllocateBVHAllMeshNodeCountsMemory(
    SyclMemoryManager& mem_mgr, DeviceBVHData& bvh_data) {
  bvh_data.node_mesh_ids =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
  bvh_data.num_childrenAll =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
}

void SyclMemoryHelper::AllocateBVHAllMeshTempMemory(SyclMemoryManager& mem_mgr,
                                                    DeviceBVHData& bvh_data,
                                                    uint32_t total_elements) {
  bvh_data.indicesAll = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  bvh_data.keysAll = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  bvh_data.deltasAll = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  bvh_data.range_leftsAll =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
  bvh_data.range_rightsAll =
      mem_mgr.AllocateDevice<uint32_t>(bvh_data.total_nodes);
}

void SyclMemoryHelper::AllocateDeviceMeshPairCollidingIndicesMemory(
    SyclMemoryManager& mem_mgr,
    DeviceMeshPairCollidingIndices& mesh_pair_colliding_indices,
    uint32_t new_size) {
  mesh_pair_colliding_indices.collision_indices_A =
      mem_mgr.AllocateDevice<uint32_t>(new_size);
  mesh_pair_colliding_indices.collision_indices_B =
      mem_mgr.AllocateDevice<uint32_t>(new_size);
  mesh_pair_colliding_indices.capacity_ = new_size;
  mesh_pair_colliding_indices.size_ = 0;
}

void SyclMemoryHelper::ResizeDeviceMeshPairCollidingIndicesMemory(
    SyclMemoryManager& mem_mgr,
    DeviceMeshPairCollidingIndices& mesh_pair_colliding_indices,
    uint32_t new_size) {
  if (new_size > mesh_pair_colliding_indices.capacity_) {
    // No need to copy any of the data since this is rewritten anyways
    mem_mgr.Free(mesh_pair_colliding_indices.collision_indices_A);
    mem_mgr.Free(mesh_pair_colliding_indices.collision_indices_B);
    mesh_pair_colliding_indices.capacity_ =
        std::max(static_cast<uint32_t>(
                     std::ceil(mesh_pair_colliding_indices.capacity_ * 1.2)),
                 new_size);
    mesh_pair_colliding_indices.collision_indices_A =
        mem_mgr.AllocateDevice<uint32_t>(mesh_pair_colliding_indices.capacity_);
    mesh_pair_colliding_indices.collision_indices_B =
        mem_mgr.AllocateDevice<uint32_t>(mesh_pair_colliding_indices.capacity_);
  }
  mesh_pair_colliding_indices.size_ = new_size;
}

void SyclMemoryHelper::AllocateMeshElementVerticesMemory(
    SyclMemoryManager& mem_mgr, DeviceMeshData& mesh_data,
    uint32_t total_elements, uint32_t total_vertices) {
  // Allocate element data
  mesh_data.elements =
      mem_mgr.AllocateDevice<std::array<int, 4>>(total_elements);
  mesh_data.element_mesh_ids = mem_mgr.AllocateDevice<uint32_t>(total_elements);
  mesh_data.inward_normals_M =
      mem_mgr.AllocateDevice<std::array<Vector3<double>, 4>>(total_elements);
  mesh_data.inward_normals_W =
      mem_mgr.AllocateDevice<std::array<Vector3<double>, 4>>(total_elements);
  mesh_data.min_pressures = mem_mgr.AllocateDevice<double>(total_elements);
  mesh_data.max_pressures = mem_mgr.AllocateDevice<double>(total_elements);
  mesh_data.gradient_M_pressure_at_Mo =
      mem_mgr.AllocateDevice<Vector4<double>>(total_elements);
  mesh_data.gradient_W_pressure_at_Wo =
      mem_mgr.AllocateDevice<Vector4<double>>(total_elements);
  mesh_data.element_aabb_min_W =
      mem_mgr.AllocateDevice<Vector3<double>>(total_elements);
  mesh_data.element_aabb_max_W =
      mem_mgr.AllocateDevice<Vector3<double>>(total_elements);

  // Allocate vertex data
  mesh_data.vertices_M =
      mem_mgr.AllocateDevice<Vector3<double>>(total_vertices);
  mesh_data.vertices_W =
      mem_mgr.AllocateDevice<Vector3<double>>(total_vertices);
  mesh_data.pressures = mem_mgr.AllocateDevice<double>(total_vertices);
  mesh_data.vertex_mesh_ids = mem_mgr.AllocateDevice<uint32_t>(total_vertices);
}

// Allocate collision detection memory of arrays based on number of geometries
void SyclMemoryHelper::AllocateGeometryCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
    uint32_t num_geometries) {
  collision_data.total_checks_per_geometry =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
  // geom_collision_filternum_cols[i] is the number of elements that need to
  // be checked with each of the elements of the ith geometry
  // Will be highest for 1st geometry and lowest for the last geometry (due to
  // symmetric nature of collision_filter - we are only consider upper
  // triangle)
  collision_data.geom_collision_filter_num_cols =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
  // Stores the exclusive scan of total checks per geometry
  collision_data.geom_collision_filter_check_offsets =
      mem_mgr.AllocateHost<uint32_t>(num_geometries);
}

// Allocate collision detection memory of arrays based on total checks
void SyclMemoryHelper::AllocateTotalChecksCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
    uint32_t total_checks) {
  // Broad phase data
  collision_data.collision_filter =
      mem_mgr.AllocateDevice<uint8_t>(total_checks);
  collision_data.collision_filter_host_body_index =
      mem_mgr.AllocateHost<uint32_t>(total_checks);
  collision_data.prefix_sum_total_checks =
      mem_mgr.AllocateDevice<uint32_t>(total_checks);
}

// Allocate collision detection memory of arrays based on estimated narrow
// phase checks
void SyclMemoryHelper::AllocateNarrowPhaseChecksCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data,
    uint32_t estimated_narrow_phase_checks) {
  // Narrow phase data
  collision_data.narrow_phase_check_indices =
      mem_mgr.AllocateDevice<uint32_t>(estimated_narrow_phase_checks);
  collision_data.narrow_phase_check_validity =
      mem_mgr.AllocateDevice<uint8_t>(estimated_narrow_phase_checks);
  collision_data.prefix_sum_narrow_phase_checks =
      mem_mgr.AllocateDevice<uint32_t>(estimated_narrow_phase_checks);
}

// Allocate polygon memory
void SyclMemoryHelper::AllocateFullPolygonMemory(
    SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data,
    uint32_t estimated_narrow_phase_checks) {
  // Raw polygon data
  polygon_data.polygon_areas =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_centroids =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_narrow_phase_checks);
  polygon_data.polygon_normals =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_narrow_phase_checks);
  polygon_data.polygon_g_M =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_g_N =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_pressure_W =
      mem_mgr.AllocateDevice<double>(estimated_narrow_phase_checks);
  polygon_data.polygon_geom_index_A =
      mem_mgr.AllocateDevice<GeometryId>(estimated_narrow_phase_checks);
  polygon_data.polygon_geom_index_B =
      mem_mgr.AllocateDevice<GeometryId>(estimated_narrow_phase_checks);
}

void SyclMemoryHelper::AllocateCompactPolygonMemory(
    SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data,
    uint32_t estimated_polygons) {
  // Compacted polygon data
  polygon_data.compacted_polygon_areas =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_centroids =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_polygons);
  polygon_data.compacted_polygon_normals =
      mem_mgr.AllocateDevice<Vector3<double>>(estimated_polygons);
  polygon_data.compacted_polygon_g_M =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_g_N =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_pressure_W =
      mem_mgr.AllocateDevice<double>(estimated_polygons);
  polygon_data.compacted_polygon_geom_index_A =
      mem_mgr.AllocateDevice<GeometryId>(estimated_polygons);
  polygon_data.compacted_polygon_geom_index_B =
      mem_mgr.AllocateDevice<GeometryId>(estimated_polygons);

  polygon_data.valid_polygon_indices =
      mem_mgr.AllocateDevice<uint32_t>(estimated_polygons);
}

// Free all mesh memory
void SyclMemoryHelper::FreeMeshMemory(SyclMemoryManager& mem_mgr,
                                      DeviceMeshData& mesh_data) {
  // Element data
  mem_mgr.Free(mesh_data.elements);
  mem_mgr.Free(mesh_data.element_mesh_ids);
  mem_mgr.Free(mesh_data.inward_normals_M);
  mem_mgr.Free(mesh_data.inward_normals_W);
  mem_mgr.Free(mesh_data.min_pressures);
  mem_mgr.Free(mesh_data.max_pressures);
  mem_mgr.Free(mesh_data.gradient_M_pressure_at_Mo);
  mem_mgr.Free(mesh_data.gradient_W_pressure_at_Wo);
  mem_mgr.Free(mesh_data.element_aabb_min_W);
  mem_mgr.Free(mesh_data.element_aabb_max_W);

  // Vertex data
  mem_mgr.Free(mesh_data.vertices_M);
  mem_mgr.Free(mesh_data.vertices_W);
  mem_mgr.Free(mesh_data.pressures);
  mem_mgr.Free(mesh_data.vertex_mesh_ids);

  // Lookup arrays
  mem_mgr.Free(mesh_data.element_offsets);
  mem_mgr.Free(mesh_data.vertex_offsets);
  mem_mgr.Free(mesh_data.element_counts);
  mem_mgr.Free(mesh_data.vertex_counts);
  mem_mgr.Free(mesh_data.geometry_ids);
  mem_mgr.Free(mesh_data.transforms);
}

// Free individual BVH mesh memory
void SyclMemoryHelper::FreeBVHSingleMeshMemory(SyclMemoryManager& mem_mgr,
                                               BVH& bvh_mesh) {
  if (bvh_mesh.node_lowers != nullptr) {
    mem_mgr.Free(bvh_mesh.node_lowers);
  }
  if (bvh_mesh.node_uppers != nullptr) {
    mem_mgr.Free(bvh_mesh.node_uppers);
  }
  if (bvh_mesh.node_parents != nullptr) {
    mem_mgr.Free(bvh_mesh.node_parents);
  }
  if (bvh_mesh.root != nullptr) {
    mem_mgr.Free(bvh_mesh.root);
  }
  bvh_mesh.node_lowers = nullptr;
  bvh_mesh.node_uppers = nullptr;
  bvh_mesh.node_parents = nullptr;
  bvh_mesh.root = nullptr;
}
// Free all BVH memory
void SyclMemoryHelper::FreeBVHSingleMeshAndAllMeshMemory(
    SyclMemoryManager& mem_mgr, DeviceBVHData& bvh_data) {
  // Free per-mesh BVH node data first
  if (bvh_data.bvhAll != nullptr) {
    for (uint32_t i = 0; i < bvh_data.num_meshes; ++i) {
      FreeBVHSingleMeshMemory(mem_mgr, bvh_data.bvhAll[i]);
    }
  }

  mem_mgr.Free(bvh_data.bvhAll);
  mem_mgr.Free(bvh_data.node_counts_per_mesh);
  mem_mgr.Free(bvh_data.node_offsets);
  mem_mgr.Free(bvh_data.total_lowerAll);
  mem_mgr.Free(bvh_data.total_upperAll);
  mem_mgr.Free(bvh_data.total_inv_edgesAll);
  mem_mgr.Free(bvh_data.indicesAll);
  mem_mgr.Free(bvh_data.node_mesh_ids);
  mem_mgr.Free(bvh_data.num_childrenAll);
}

void SyclMemoryHelper::FreeBVHAllMeshTempMemory(SyclMemoryManager& mem_mgr,
                                                DeviceBVHData& bvh_data) {
  mem_mgr.Free(bvh_data.keysAll);
  mem_mgr.Free(bvh_data.deltasAll);
  mem_mgr.Free(bvh_data.range_leftsAll);
  mem_mgr.Free(bvh_data.range_rightsAll);
}

// Free collision memory
void SyclMemoryHelper::FreeCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data) {
  mem_mgr.Free(collision_data.collision_filter);
  mem_mgr.Free(collision_data.collision_filter_host_body_index);
  mem_mgr.Free(collision_data.total_checks_per_geometry);
  mem_mgr.Free(collision_data.geom_collision_filter_num_cols);
  mem_mgr.Free(collision_data.geom_collision_filter_check_offsets);
  mem_mgr.Free(collision_data.prefix_sum_total_checks);
  mem_mgr.Free(collision_data.narrow_phase_check_indices);
  mem_mgr.Free(collision_data.narrow_phase_check_validity);
  mem_mgr.Free(collision_data.prefix_sum_narrow_phase_checks);
}

// Free only the collision detection memory of arrays based on narrow phase
// checks
void SyclMemoryHelper::FreeNarrowPhaseChecksCollisionMemory(
    SyclMemoryManager& mem_mgr, DeviceCollisionData& collision_data) {
  mem_mgr.Free(collision_data.narrow_phase_check_indices);
  mem_mgr.Free(collision_data.narrow_phase_check_validity);
  mem_mgr.Free(collision_data.prefix_sum_narrow_phase_checks);
}

// Free polygon memory
void SyclMemoryHelper::FreeFullPolygonMemory(SyclMemoryManager& mem_mgr,
                                             DevicePolygonData& polygon_data) {
  // Raw polygon data
  mem_mgr.Free(polygon_data.polygon_areas);
  mem_mgr.Free(polygon_data.polygon_centroids);
  mem_mgr.Free(polygon_data.polygon_normals);
  mem_mgr.Free(polygon_data.polygon_g_M);
  mem_mgr.Free(polygon_data.polygon_g_N);
  mem_mgr.Free(polygon_data.polygon_pressure_W);
  mem_mgr.Free(polygon_data.polygon_geom_index_A);
  mem_mgr.Free(polygon_data.polygon_geom_index_B);

  // Debug data
  mem_mgr.Free(polygon_data.debug_polygon_vertices);
}

void SyclMemoryHelper::FreeCompactPolygonMemory(
    SyclMemoryManager& mem_mgr, DevicePolygonData& polygon_data) {
  // Compacted polygon data
  mem_mgr.Free(polygon_data.compacted_polygon_areas);
  mem_mgr.Free(polygon_data.compacted_polygon_centroids);
  mem_mgr.Free(polygon_data.compacted_polygon_normals);
  mem_mgr.Free(polygon_data.compacted_polygon_g_M);
  mem_mgr.Free(polygon_data.compacted_polygon_g_N);
  mem_mgr.Free(polygon_data.compacted_polygon_pressure_W);
  mem_mgr.Free(polygon_data.compacted_polygon_geom_index_A);
  mem_mgr.Free(polygon_data.compacted_polygon_geom_index_B);
  mem_mgr.Free(polygon_data.valid_polygon_indices);
}

void SyclMemoryHelper::FreePolygonMemory(SyclMemoryManager& mem_mgr,
                                         DevicePolygonData& polygon_data) {
  FreeFullPolygonMemory(mem_mgr, polygon_data);
  FreeCompactPolygonMemory(mem_mgr, polygon_data);
}

}  // namespace sycl_impl
}  // namespace internal
}  // namespace geometry
}  // namespace drake