#include "../include/neonflux/dot_product.h"
#include "../include/neonflux/vector.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace neonflux;
using namespace std;

template <typename Func> double measure_ms(Func f, int iterations) {
  auto start = chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    f();
  }
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double, milli> elapsed = end - start;
  return elapsed.count();
}

int main() {
  size_t N = 1000000; // 1 Million elements (4MB)
  int iterations = 1000;

  cout << "Benchmarking Dot Product (N = " << N
       << ", Iterations = " << iterations << ")" << endl;

  FloatVector a(N);
  FloatVector b(N);
  // Random data:
  for (size_t i = 0; i < N; ++i) {
    a.data[i] = 1.0f;
    b.data[i] = 1.0f;
  }

  dot_unrolled4(a, b);
  dot_naive(a, b);

  double naive_ms = measure_ms(
      [&]() {
        volatile float res = dot_naive(a, b);
        (void)res;
      },
      iterations);

  double optim_ms = measure_ms(
      [&]() {
        volatile float res = dot_unrolled4(a, b);
        (void)res;
      },
      iterations);

  double naive_avg_ms = naive_ms / iterations;
  double optim_avg_ms = optim_ms / iterations;

  double gflops_naive = (2.0 * N * 1e-9) / (naive_avg_ms * 1e-3);
  double gflops_optim = (2.0 * N * 1e-9) / (optim_avg_ms * 1e-3);

  cout << fixed << setprecision(4);
  cout << "Naive:    " << naive_avg_ms << " ms/iter | " << gflops_naive
       << " GFLOPS" << endl;
  cout << "Unrolled: " << optim_avg_ms << " ms/iter | " << gflops_optim
       << " GFLOPS" << endl;
  cout << "Speedup:  " << (naive_ms / optim_ms) << "x" << endl;
  return 0;
}
