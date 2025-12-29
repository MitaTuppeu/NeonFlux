#include "../include/neonflux/gemm.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace neonflux;
using namespace std;

bool is_close(float a, float b, float epsilon = 1e-3f) {
  return fabs(a - b) < epsilon;
}

void test_gemm(int M, int N, int K) {
  cout << "Testing GEMM " << M << "x" << N << "x" << K << "..." << endl;

  vector<float> A(M * K);
  vector<float> B(K * N);
  vector<float> C_ref(M * N, 0.0f);
  vector<float> C_opt(M * N, 0.0f);

  // Init A, B
  for (size_t i = 0; i < A.size(); ++i)
    A[i] = (float)(rand() % 10) * 0.1f;
  for (size_t i = 0; i < B.size(); ++i)
    B[i] = (float)(rand() % 10) * 0.1f;

  gemm_ref(M, N, K, A.data(), B.data(), C_ref.data());
  gemm_optimized(M, N, K, A.data(), B.data(), C_opt.data());

  // Compare
  for (size_t i = 0; i < C_ref.size(); ++i) {
    if (!is_close(C_ref[i], C_opt[i])) {
      cout << "[FAIL] Mismatch at index " << i << ". Ref: " << C_ref[i]
           << " Opt: " << C_opt[i] << endl;
      exit(1);
    }
  }
  cout << "[PASS]" << endl;
}

int main() {
  // Phase 3 Proof of Concept assumes multiples of 4 for now
  test_gemm(4, 4, 4);
  test_gemm(16, 16, 16);
  test_gemm(64, 64, 64);
  test_gemm(128, 128, 128);
  // test_gemm(129, 129, 129); // Would fail without padding/edge handling

  cout << "===========================" << endl;
  cout << "PHASE 3 VERIFICATION PASSED" << endl;
  cout << "===========================" << endl;
  return 0;
}
