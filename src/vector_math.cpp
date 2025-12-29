#include "../include/neonflux/vector_math.h"
#include <arm_neon.h>
#include <stdexcept>

using namespace std;

namespace neonflux {

namespace {
void check_sizes(const FloatVector &a, const FloatVector &b,
                 const FloatVector &res) {
  if (a.size != b.size || a.size != res.size) {
    throw invalid_argument("Vector sizes must match");
  }
}
} 
void add(const FloatVector &a, const FloatVector &b, FloatVector &result) {
  check_sizes(a, b, result);
  size_t n = a.size;
  size_t i = 0;
  // SIMD Loop (4 floats at a time)
  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vb = vld1q_f32(&b.data[i]);
    float32x4_t vres = vaddq_f32(va, vb);
    vst1q_f32(&result.data[i], vres);
  }
  for (; i < n; ++i) {
    result.data[i] = a.data[i] + b.data[i];
  }
}

void sub(const FloatVector &a, const FloatVector &b, FloatVector &result) {
  check_sizes(a, b, result);
  size_t n = a.size;
  size_t i = 0;

  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vb = vld1q_f32(&b.data[i]);
    float32x4_t vres = vsubq_f32(va, vb);
    vst1q_f32(&result.data[i], vres);
  }

  for (; i < n; ++i) {
    result.data[i] = a.data[i] - b.data[i];
  }
}

void mul(const FloatVector &a, const FloatVector &b, FloatVector &result) {
  check_sizes(a, b, result);
  size_t n = a.size;
  size_t i = 0;

  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vb = vld1q_f32(&b.data[i]);
    float32x4_t vres = vmulq_f32(va, vb);
    vst1q_f32(&result.data[i], vres);
  }

  for (; i < n; ++i) {
    result.data[i] = a.data[i] * b.data[i];
  }
}

void scalar_mul(const FloatVector &a, float scalar, FloatVector &result) {
  if (a.size != result.size) {
    throw invalid_argument("Vector sizes must match for scalar output");
  }
  size_t n = a.size;
  size_t i = 0;

  float32x4_t vscalar = vdupq_n_f32(scalar);

  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vres = vmulq_f32(va, vscalar);
    vst1q_f32(&result.data[i], vres);
  }

  for (; i < n; ++i) {
    result.data[i] = a.data[i] * scalar;
  }
}

} 
