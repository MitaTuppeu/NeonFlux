#ifndef NEONFLUX_VECTOR_H
#define NEONFLUX_VECTOR_H

#include "allocator.h"
#include <algorithm>
#include <cstring>

namespace neonflux {

struct FloatVector {
  float *data;
  size_t size;

  FloatVector(size_t s) : size(s) {
    data = AlignedAllocator::allocate(size);
    std::memset(data, 0, size * sizeof(float));
  }

  // Copy constructor (deep copy)
  FloatVector(const FloatVector &other) : size(other.size) {
    data = AlignedAllocator::allocate(size);
    std::memcpy(data, other.data, size * sizeof(float));
  }

  // Move constructor
  FloatVector(FloatVector &&other) noexcept
      : data(other.data), size(other.size) {
    other.data = nullptr;
    other.size = 0;
  }

  ~FloatVector() {
    if (data) {
      AlignedAllocator::deallocate(data);
    }
  }

  void print(size_t n = 10) const;
};

} 

#endif 
