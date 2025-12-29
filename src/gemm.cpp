#include "../include/neonflux/gemm.h"
#include <algorithm>
#include <arm_neon.h>
#include <cstring>
#include <omp.h>
#include <vector>

using namespace std;

namespace neonflux {

// Naive implementation for verification
void gemm_ref(int M, int N, int K, const float *A, const float *B, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < K; ++p) {
        sum += A[i * K + p] * B[p * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// Optimized Implementation

#define MC 256
#define KC 128
#define NC 128

#define MR 4
#define NR 4

void pack_matrix_b(int k, const float *B, int ldb, float *buffer) {
  for (int p = 0; p < k; ++p) {
    for (int j = 0; j < NR; ++j) {
      *buffer++ = B[p * ldb + j];
    }
  }
}

void pack_matrix_a_interleaved(int k, const float *A, int lda, float *buffer) {
  for (int p = 0; p < k; ++p) {
    for (int i = 0; i < MR; ++i) {
      *buffer++ = A[i * lda + p];
    }
  }
}

void kernel_4x4(int k, const float *A_packed, const float *B_packed, float *C,
                int ldc) {
  float32x4_t c_0 = vdupq_n_f32(0.0f);
  float32x4_t c_1 = vdupq_n_f32(0.0f);
  float32x4_t c_2 = vdupq_n_f32(0.0f);
  float32x4_t c_3 = vdupq_n_f32(0.0f);

  const float *a_ptr = A_packed;
  const float *b_ptr = B_packed;

  for (int p = 0; p < k; ++p) {
    float32x4_t b_reg = vld1q_f32(b_ptr);
    b_ptr += 4;

    float32x4_t a0 = vdupq_n_f32(*a_ptr++);
    float32x4_t a1 = vdupq_n_f32(*a_ptr++);
    float32x4_t a2 = vdupq_n_f32(*a_ptr++);
    float32x4_t a3 = vdupq_n_f32(*a_ptr++);

    c_0 = vmlaq_f32(c_0, a0, b_reg);
    c_1 = vmlaq_f32(c_1, a1, b_reg);
    c_2 = vmlaq_f32(c_2, a2, b_reg);
    c_3 = vmlaq_f32(c_3, a3, b_reg);
  }

  float32x4_t c_old_0 = vld1q_f32(C);
  vst1q_f32(C, vaddq_f32(c_old_0, c_0));

  C += ldc;
  float32x4_t c_old_1 = vld1q_f32(C);
  vst1q_f32(C, vaddq_f32(c_old_1, c_1));

  C += ldc;
  float32x4_t c_old_2 = vld1q_f32(C);
  vst1q_f32(C, vaddq_f32(c_old_2, c_2));

  C += ldc;
  float32x4_t c_old_3 = vld1q_f32(C);
  vst1q_f32(C, vaddq_f32(c_old_3, c_3));
}

void gemm_optimized(int M, int N, int K, const float *A, const float *B,
                    float *C) {

#pragma omp parallel for schedule(static)
  for (int j = 0; j < N; j += NC) {
    vector<float> _packA(MC * KC);
    vector<float> _packB(KC * NC);
    float *packA = _packA.data();
    float *packB = _packB.data();

    int nc = min(NC, N - j);

    for (int k = 0; k < K; k += KC) {
      int kc = min(KC, K - k);

      for (int i = 0; i < M; i += MC) {
        int mc = min(MC, M - i);
        for (int ii = 0; ii < mc; ii += MR) {
          pack_matrix_a_interleaved(kc, &A[(i + ii) * K + k], K,
                                    packA + ii * kc);
        }

        for (int jj = 0; jj < nc; jj += NR) {
          pack_matrix_b(kc, &B[k * N + (j + jj)], N, packB);
          bool col_edge = (jj + NR > nc);

          if (!col_edge) {
            for (int ii = 0; ii < mc; ii += MR) {
              // We might still have row boundary issues if M % MR != 0
              // But for now assuming M is multiple of 4 (batch size 64)
              bool row_edge = (i + ii + MR > M);

              if (!row_edge) {
                kernel_4x4(kc, packA + ii * kc, packB,
                           &C[(i + ii) * N + (j + jj)], N);
              } else {
                float tempC[MR * NR];
                memset(tempC, 0, sizeof(tempC));
                kernel_4x4(kc, packA + ii * kc, packB, tempC, NR);
                int valid_rows = M - (i + ii);
                for (int r = 0; r < valid_rows; ++r) {
                  for (int c = 0; c < NR; ++c) {
                    C[(i + ii + r) * N + (j + jj + c)] += tempC[r * NR + c];
                  }
                }
              }
            }
          } else {
            int valid_cols = nc - jj;

            for (int ii = 0; ii < mc; ii += MR) {
              float tempC[MR * NR];
              memset(tempC, 0, sizeof(tempC));
              kernel_4x4(kc, packA + ii * kc, packB, tempC, NR);
              int valid_rows = min((int)MR, M - (i + ii));
              for (int r = 0; r < valid_rows; ++r) {
                for (int c = 0; c < valid_cols; ++c) {
                  C[(i + ii + r) * N + (j + jj + c)] += tempC[r * NR + c];
                }
              }
            }
          }
        }
      }
    }
  }
}

}
