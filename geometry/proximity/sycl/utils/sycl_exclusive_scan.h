#include <cmath>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

class CustomExclusiveScan {
 public:
  CustomExclusiveScan(sycl::queue& queue) : queue_(queue) {}

  void operator()(const uint8_t* input, size_t* output, size_t size) {
    const size_t local_size = 256;
    size_t num_groups = (size + local_size - 1) / local_size;

    // Allocate device memory for block sums
    size_t* block_sums = sycl::malloc_device<size_t>(num_groups, queue_);

    // Step 1: Local scan and write block sums
    auto event = queue_.submit([&](sycl::handler& h) {
      sycl::local_accessor<size_t, 1> temp(sycl::range<1>(local_size), h);
      h.parallel_for(sycl::nd_range<1>(num_groups * local_size, local_size),
                     [=](sycl::nd_item<1> item) {
                       size_t gid = item.get_global_id(0);
                       size_t lid = item.get_local_id(0);
                       size_t group = item.get_group(0);

                       // Load to shared memory
                       size_t val =
                           (gid < size) ? static_cast<size_t>(input[gid]) : 0;
                       temp[lid] = val;
                       item.barrier(sycl::access::fence_space::local_space);

                       // Up-sweep phase (build sum tree)
                       for (int d = 1; d < local_size; d <<= 1) {
                         if (lid % (2 * d) == 0) {
                           temp[lid + 2 * d - 1] += temp[lid + d - 1];
                         }
                         item.barrier(sycl::access::fence_space::local_space);
                       }

                       // Save total sum and clear last element
                       if (lid == 0) {
                         block_sums[group] = temp[local_size - 1];
                         temp[local_size - 1] = 0;
                       }
                       item.barrier(sycl::access::fence_space::local_space);

                       // Down-sweep phase (traverse down tree)
                       for (int d = local_size >> 1; d > 0; d >>= 1) {
                         if (lid % (2 * d) == 0) {
                           size_t t = temp[lid + d - 1];
                           temp[lid + d - 1] = temp[lid + 2 * d - 1];
                           temp[lid + 2 * d - 1] += t;
                         }
                         item.barrier(sycl::access::fence_space::local_space);
                       }

                       // Write result
                       if (gid < size) output[gid] = temp[lid];
                     });
    });

    // Step 2: Scan the block sums (recursively if needed)
    if (num_groups > 1) {
      size_t* scanned_block_sums =
          sycl::malloc_device<size_t>(num_groups, queue_);
      (*this)(block_sums, scanned_block_sums, num_groups);

      // Step 3: Add scanned block sums to each local result
      auto event2 = queue_.submit([&](sycl::handler& h) {
        h.depends_on(event);
        h.parallel_for(sycl::nd_range<1>(num_groups * local_size, local_size),
                       [=](sycl::nd_item<1> item) {
                         size_t gid = item.get_global_id(0);
                         size_t group = item.get_group(0);

                         if (gid < size) {
                           output[gid] += scanned_block_sums[group];
                         }
                       });
      });
      event2.wait_and_throw();
      sycl::free(scanned_block_sums, queue_);
    }
    sycl::free(block_sums, queue_);
  }

  // Overload for size_t input (for block sums)
  void operator()(const size_t* input, size_t* output, size_t size) {
    const size_t local_size = 256;
    size_t num_groups = (size + local_size - 1) / local_size;

    size_t* block_sums = sycl::malloc_device<size_t>(num_groups, queue_);

    auto event = queue_.submit([&](sycl::handler& h) {
      sycl::local_accessor<size_t, 1> temp(sycl::range<1>(local_size), h);
      h.parallel_for(sycl::nd_range<1>(num_groups * local_size, local_size),
                     [=](sycl::nd_item<1> item) {
                       size_t gid = item.get_global_id(0);
                       size_t lid = item.get_local_id(0);
                       size_t group = item.get_group(0);

                       size_t val = (gid < size) ? input[gid] : 0;
                       temp[lid] = val;
                       item.barrier(sycl::access::fence_space::local_space);

                       for (int d = 1; d < local_size; d <<= 1) {
                         if (lid % (2 * d) == 0) {
                           temp[lid + 2 * d - 1] += temp[lid + d - 1];
                         }
                         item.barrier(sycl::access::fence_space::local_space);
                       }

                       if (lid == 0) {
                         block_sums[group] = temp[local_size - 1];
                         temp[local_size - 1] = 0;
                       }
                       item.barrier(sycl::access::fence_space::local_space);

                       for (int d = local_size >> 1; d > 0; d >>= 1) {
                         if (lid % (2 * d) == 0) {
                           size_t t = temp[lid + d - 1];
                           temp[lid + d - 1] = temp[lid + 2 * d - 1];
                           temp[lid + 2 * d - 1] += t;
                         }
                         item.barrier(sycl::access::fence_space::local_space);
                       }

                       if (gid < size) output[gid] = temp[lid];
                     });
    });

    if (num_groups > 1) {
      size_t* scanned_block_sums =
          sycl::malloc_device<size_t>(num_groups, queue_);
      (*this)(block_sums, scanned_block_sums, num_groups);
      auto event2 = queue_.submit([&](sycl::handler& h) {
        h.depends_on(event);
        h.parallel_for(sycl::nd_range<1>(num_groups * local_size, local_size),
                       [=](sycl::nd_item<1> item) {
                         size_t gid = item.get_global_id(0);
                         size_t group = item.get_group(0);
                         if (gid < size) {
                           output[gid] += scanned_block_sums[group];
                         }
                       });
      });
      event2.wait_and_throw();
      sycl::free(scanned_block_sums, queue_);
    }
    sycl::free(block_sums, queue_);
  }

 private:
  sycl::queue& queue_;
};