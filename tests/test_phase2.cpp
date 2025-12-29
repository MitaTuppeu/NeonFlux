#include "../include/neonflux/dot_product.h"
#include "../include/neonflux/vector.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace neonflux;
using namespace std;

bool is_close(float a, float b, float epsilon = 1e-4f) {
  return fabs(a - b) < epsilon;
}

void test_dot_correctness(size_t size) {
  FloatVector a(size);
  FloatVector b(size);

  for (size_t i = 0; i < size; ++i) {
    a.data[i] = 1.0f;
    b.data[i] = (float)(i % 5); // 0, 1, 2, 3, 4, 0...
  }

  float naive = dot_naive(a, b);
  float optimized = dot_unrolled4(a, b);

  if (!is_close(naive, optimized)) {
    cout << "[FAIL] Dot product mismatch at size " << size
         << ". Naive: " << naive << ", Optimized: " << optimized << endl;
    exit(1);
  }
}

int main() {
  cout << "Testing Dot Product Correctness..." << endl;

  // Test various sizes for edge cases
  vector<size_t> sizes = {1, 3, 4, 8, 15, 16, 17, 32, 63, 64, 65, 1024, 100003};

  for (size_t s : sizes) {
    test_dot_correctness(s);
  }

  cout << "[PASS] All sizes verified against naive implementation." << endl;
  cout << "===========================" << endl;
  cout << "PHASE 2 VERIFICATION PASSED" << endl;
  cout << "===========================" << endl;
  return 0;
}
