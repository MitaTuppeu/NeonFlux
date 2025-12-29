#include "../include/neonflux/activations.h"
#include <algorithm>
#include <arm_neon.h>
#include <omp.h>

using namespace std;

namespace neonflux {

void relu_optimized(const float *src, float *dst, int n) {
  float32x4_t vzero = vdupq_n_f32(0.0f);

// Parallelize: Divide the array chunks across threads
#pragma omp parallel for schedule(static)
  for (int i = 0; i <= n - 4; i += 4) {
    float32x4_t vdata = vld1q_f32(&src[i]);
    float32x4_t vres = vmaxq_f32(vdata, vzero);
    vst1q_f32(&dst[i], vres);
  }

  // Tail Handling (Scalar fallback)
  int remainder_start = (n / 4) * 4;
  for (int i = remainder_start; i < n; i++) {
    dst[i] = max(src[i], 0.0f);
  }
}
} 
