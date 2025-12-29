#include "../include/neonflux/gemm.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace neonflux;
using namespace std;

template <typename Func> double measure_ms(Func f, int iterations) {
  auto start = chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    f();
  }
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> elapsed = end - start;
  return elapsed.count() / iterations;
}

int main() {
  int N = 256; // N x N
  int iterations = 10;

  cout << "Benchmarking GEMM (N=" << N << ", " << iterations << " iters)..."
       << endl;

  vector<float> A(N * N, 1.0f);
  vector<float> B(N * N, 1.0f);
  vector<float> C(N * N, 0.0f);

  // Ref
  double ms_ref = measure_ms(
      [&]() { gemm_ref(N, N, N, A.data(), B.data(), C.data()); }, iterations);

  // Opt
  double ms_opt = measure_ms(
      [&]() { gemm_optimized(N, N, N, A.data(), B.data(), C.data()); },
      iterations * 5);

  double gflops_ref = (2.0 * N * N * N * 1e-9) / (ms_ref * 1e-3);
  double gflops_opt = (2.0 * N * N * N * 1e-9) / (ms_opt * 1e-3);

  cout << fixed << setprecision(4);
  cout << "Ref: " << ms_ref << " ms | " << gflops_ref << " GFLOPS" << endl;
  cout << "Opt: " << ms_opt << " ms | " << gflops_opt << " GFLOPS" << endl;
  cout << "Speedup: " << (ms_ref / ms_opt) << "x" << endl;

  return 0;
}
