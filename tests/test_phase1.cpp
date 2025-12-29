#include "../include/neonflux/vector.h"
#include "../include/neonflux/vector_math.h"
#include <cassert>
#include <iostream>

using namespace neonflux;
using namespace std;

bool check_alignment(float *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) % 16 == 0);
}

void test_alignment() {
  cout << "Testing Memory Alignment..." << endl;
  FloatVector v1(10);
  FloatVector v2(17);
  FloatVector v3(1024);

  if (check_alignment(v1.data) && check_alignment(v2.data) &&
      check_alignment(v3.data)) {
    cout << "[PASS] All vectors aligned to 16 bytes." << endl;
  } else {
    cout << "[FAIL] Memory alignment check failed!" << endl;
    exit(1);
  }
}

void test_arithmetic() {
  cout << "Testing Basic Arithmetic (Size 8 - multiple of 4)..." << endl;
  size_t size = 8;
  FloatVector a(size);
  FloatVector b(size);
  FloatVector res(size);

  for (size_t i = 0; i < size; ++i) {
    a.data[i] = i * 1.0f;
    b.data[i] = i * 2.0f;
  }

  add(a, b, res);
  for (size_t i = 0; i < size; ++i) {
    if (res.data[i] != (a.data[i] + b.data[i])) {
      cout << "[FAIL] Add failed at index " << i << ". Got " << res.data[i]
           << " expected " << (a.data[i] + b.data[i]) << endl;
      exit(1);
    }
  }
  cout << "[PASS] Add operation." << endl;

  sub(b, a, res);
  for (size_t i = 0; i < size; ++i) {
    if (res.data[i] != (b.data[i] - a.data[i])) {
      cout << "[FAIL] Sub failed at index " << i << endl;
      exit(1);
    }
  }
  cout << "[PASS] Sub operation." << endl;

  mul(a, b, res);
  for (size_t i = 0; i < size; ++i) {
    if (res.data[i] != (a.data[i] * b.data[i])) {
      cout << "[FAIL] Mul failed at index " << i << endl;
      exit(1);
    }
  }
  cout << "[PASS] Mul operation." << endl;
}

void test_scalar() {
  cout << "Testing Scalar Multiplication..." << endl;
  FloatVector a(8);
  FloatVector res(8);
  for (size_t i = 0; i < 8; ++i)
    a.data[i] = (float)i;

  scalar_mul(a, 2.5f, res);
  for (size_t i = 0; i < 8; ++i) {
    if (res.data[i] != (a.data[i] * 2.5f)) {
      cout << "[FAIL] Scalar mul failed at index " << i << endl;
      exit(1);
    }
  }
  cout << "[PASS] Scalar multiplication." << endl;
}

void test_tail_handling() {
  cout << "Testing Tail Handling (Size 7)..." << endl;
  size_t size = 7;
  FloatVector a(size);
  FloatVector b(size);
  FloatVector res(size);

  for (size_t i = 0; i < size; ++i) {
    a.data[i] = 10.0f;
    b.data[i] = (float)i;
  }

  add(a, b, res);
  for (size_t i = 0; i < size; ++i) {
    if (res.data[i] != (10.0f + i)) {
      cout << "[FAIL] Tail add failed at index " << i << endl;
      exit(1);
    }
  }
  cout << "[PASS] Tail handling correct for Add." << endl;
}

int main() {
  test_alignment();
  test_arithmetic();
  test_scalar();
  test_tail_handling();

  cout << "===========================" << endl;
  cout << "PHASE 1 VERIFICATION PASSED" << endl;
  cout << "===========================" << endl;
  return 0;
}
