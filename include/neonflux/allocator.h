#ifndef NEONFLUX_ALLOCATOR_H
#define NEONFLUX_ALLOCATOR_H

#include <cstdlib>
#include <stdexcept>

namespace neonflux {

class AlignedAllocator {
public:
  static float *allocate(size_t size) {
    void *ptr = nullptr;
    // 16-byte alignment for NEON (128-bit)
    if (posix_memalign(&ptr, 16, size * sizeof(float)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<float *>(ptr);
  }

  static void deallocate(float *ptr) { free(ptr); }
};

} 

#endif 
