#ifndef NEONFLUX_GEMM_H
#define NEONFLUX_GEMM_H

namespace neonflux {
void gemm_ref(int M, int N, int K, const float *A, const float *B, float *C);
void gemm_optimized(int M, int N, int K, const float *A, const float *B, float *C);
} 

#endif
