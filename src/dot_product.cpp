#include "../include/neonflux/dot_product.h"
#include <arm_neon.h>
#include <stdexcept>

using namespace std;

namespace neonflux {

static void check_sizes(const FloatVector &a, const FloatVector &b) {
  if (a.size != b.size) {
    throw invalid_argument("Vector sizes must match for dot product");
  }
}

float dot_naive(const FloatVector &a, const FloatVector &b) {
  check_sizes(a, b);
  float sum = 0.0f;
  for (size_t i = 0; i < a.size; ++i) {
    sum += a.data[i] * b.data[i];
  }
  return sum;
}

float dot_unrolled4(const FloatVector &a, const FloatVector &b) {
  check_sizes(a, b);

  size_t n = a.size;
  size_t i = 0;

  // 4 Accumulators for ILP
  float32x4_t vsum0 = vdupq_n_f32(0.0f);
  float32x4_t vsum1 = vdupq_n_f32(0.0f);
  float32x4_t vsum2 = vdupq_n_f32(0.0f);
  float32x4_t vsum3 = vdupq_n_f32(0.0f);

  for (; i + 15 < n; i += 16) {
    float32x4_t va0 = vld1q_f32(&a.data[i]);
    float32x4_t vb0 = vld1q_f32(&b.data[i]);
    vsum0 = vmlaq_f32(vsum0, va0, vb0);

    float32x4_t va1 = vld1q_f32(&a.data[i + 4]);
    float32x4_t vb1 = vld1q_f32(&b.data[i + 4]);
    vsum1 = vmlaq_f32(vsum1, va1, vb1);

    float32x4_t va2 = vld1q_f32(&a.data[i + 8]);
    float32x4_t vb2 = vld1q_f32(&b.data[i + 8]);
    vsum2 = vmlaq_f32(vsum2, va2, vb2);

    float32x4_t va3 = vld1q_f32(&a.data[i + 12]);
    float32x4_t vb3 = vld1q_f32(&b.data[i + 12]);
    vsum3 = vmlaq_f32(vsum3, va3, vb3);
  }

  // Step 1: Reduce 4 vectors to 1 vector
  float32x4_t vtotal = vaddq_f32(vsum0, vsum1);
  vtotal = vaddq_f32(vtotal, vsum2);
  vtotal = vaddq_f32(vtotal, vsum3);

  for (; i + 3 < n; i += 4) {
    float32x4_t va = vld1q_f32(&a.data[i]);
    float32x4_t vb = vld1q_f32(&b.data[i]);
    vtotal = vmlaq_f32(vtotal, va, vb);
  }
  // Step 2: Horizontal Reduction (Vector -> Scalar)
  float scalar_sum = vaddvq_f32(vtotal);

  for (; i < n; ++i) {
    scalar_sum += a.data[i] * b.data[i];
  }

  return scalar_sum;
}
} 
