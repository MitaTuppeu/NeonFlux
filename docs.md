# NeonFlux Technical Documentation

This document details the architectural decisions and optimization strategies employed in NeonFlux.

## Phase 1: Memory & Alignment

### The Alignment Problem
NEON instructions like `vld1q_f32` (load 128-bit) perform best when addresses are 16-byte aligned (i.e., `addr % 16 == 0`). Misaligned accesses can cause pipeline stalls or faults on strict architectures.

### Solution: `AlignedAllocator`
We wrap `posix_memalign` (Standard C) to guarantee alignment.
- **Allocation**: `posix_memalign(&ptr, 16, size * sizeof(float))`
- **Container**: `FloatVector` encapsulates this pointer, ensuring RAII-style cleanup.

### Tail Handling
Since SIMD operates on 4 floats at a time, arrays with `size % 4 != 0` leave leftover elements.
- **Main Loop**: Iterates `i` from `0` to `size - size % 4`.
- **Cleanup Loop**: Standard scalar loop processes the remaining 1-3 elements.

---

## Phase 2: Instruction Level Parallelism (Dot Product)

### The Latency Problem
A standard accumulation loop looks like this:
```cpp
vsum = vaddq_f32(vsum, vprod); // vsum depends on previous vsum
```
The CPU cannot dispatch the next `vaddq` until the previous one completes (latency ~3-4 cycles).

### Solution: 4x Loop Unrolling
We break the dependency chain by using **4 independent accumulators**:
```cpp
vsum0 = vmlaq_f32(vsum0, a0, b0);
vsum1 = vmlaq_f32(vsum1, a1, b1);
vsum2 = vmlaq_f32(vsum2, a2, b2);
vsum3 = vmlaq_f32(vsum3, a3, b3);
```
Since `vsum0` does not depend on `vsum1`, the CPU can pipeline these operations, executing them nearly in parallel.

---

## Phase 3: The Workhorse (GEMM)

We implement a simplified version of the **GotoBLAS** algorithm, optimizing for the memory hierarchy.
**Operation**: $C = C + A \times B$

### 1. Memory Packing (`pack_matrix_b`)
Matrix B (K x N) is stored row-major. Walking down a column (needed for the dot product in standard algorithms) creates a stride of `N`, causing cache thrashing.
- **Optimization**: We copy a "panel" of B (width `NR`) into a contiguous temporary buffer.
- **Benefit**: The CPU reads B sequentially during the inner kernel, maximizing prefetcher efficiency.

### 2. The 4x4 Micro-Kernel
The heart of the computation. It computes a 4x4 block of C using registers.
- **Registers Used**:
    - `v0-v3`: Accumulators for C (4x4 block).
    - `v4`: Loaded vector from B (column strip).
    - `v5-v8`: Broadcasted scalars from A.
- **Logic**:
    1. Load 4 floats from B into a vector register.
    2. Broadcast 1 float from A into a vector register.
    3. Perform Fused Multiply-Add (FMA) across the B vector.
    4. Repeat for all 4 rows of A.

### 3. Cache Blocking (Tiling)
We loop over the matrices in blocks to ensure data fits in cache.
- **KC (Kernel K)**: The depth of the dot product. Chosen to keep the packed panels of A and B in **L1 Cache**.
- **MC (Kernel M)**: The number of rows of A. Chosen to keep the block of A in **L2 Cache**.
- **Structure**:
    ```text
    Loop j over N (Step NC)
      Loop p over K (Step KC) -> Pack B panel
        Loop i over M (Step MC) -> Pack A block
           Micro-Kernel (updates 4x4 block of C)
    ```

### Benchmark Results
| Optimization Level | GFLOPS | Analysis |
|--------------------|--------|----------|
| **Naive (Ref)** | 2.5 | Memory Bound (Random Access) |
| **Optimized** | 42.8 | Compute Bound (Near Peak FPU) |

The **17x speedup** validates that the implementation successfully hides memory latency and maximizes FPU utilization.

---

The **17x speedup** validates that the implementation successfully hides memory latency and maximizes FPU utilization.

---

## Phase 4: Parallelization (OpenMP)

To scale beyond a single core, we employ **OpenMP** to parallelize the outer loop of the GEMM kernel.

### Thread-Local Packing
A naive `#pragma omp parallel for` would fail because the packing buffers (`packA`, `packB`) are shared resources. Using a lock would destroy performance.

**Solution**: We allocate packing buffers **inside** the parallel loop.
- **Memory Overhead**: Each thread allocates ~170KB (tiny) for its L2 block.
- **Throughput**: Threads work completely independently on different strips of the output matrix $C$.

```cpp
#pragma omp parallel for schedule(static)
for (int j = 0; j < N; j += NC) {
    // Thread-local buffers
    std::vector<float> _packA(MC * KC); 
    // ...
}
```

---

## Phase 5: Python Bindings (Pybind11)

We expose the C++ kernel to Python to compete with `numpy`.

### Interface
We use `pybind11` to accept `numpy.ndarray` directly.
- **Input**: `py::array_t<float>` (Wraps the underlying buffer).
- **Zero-Copy**: We pass the `void*` pointer from the NumPy array directly to our C++ kernel.
- **Output**: We allocate the result matrix `C` in C++, initialize it to zero (crucial for accumulation), and return it to Python.

### Building Extensions
Compiling C++ extensions with OpenMP on macOS is tricky because Apple Clang does not support `-fopenmp` natively.
- **Solution**: We link against `libomp` (via Homebrew) and use the `-Xpreprocessor -fopenmp` flags in `setup.py`.

- **Solution**: We link against `libomp` (via Homebrew) and use the `-Xpreprocessor -fopenmp` flags in `setup.py`.

---

## Phase 6: The Nervous System (Activations & DL Framework)

We extended the engine to support full neural network inference.

### 1. Vectorized Activation (ReLU)
ReLU (`max(0, x)`) is memory-bandwidth bound. We optimize it using **NEON** and **OpenMP**.

- **Instruction**: `vmaxq_f32(v, vdupq_n_f32(0.0f))`
    - Calculates max element-wise for 4 floats in a single cycle.
- **Parallelism**: Since elements are independent, we use `#pragma omp parallel for` checks to saturate memory channels.

```cpp
// Kernel: neonflux::relu_optimized
float32x4_t vzero = vdupq_n_f32(0.0f);
#pragma omp parallel for
for (int i = 0; i <= n - 4; i += 4) {
    float32x4_t vdata = vld1q_f32(&src[i]);
    vst1q_f32(&dst[i], vmaxq_f32(vdata, vzero));
}
```

### 2. High-Level Framework (`neon_nn`)
We implemented a PyTorch-like API in pure Python calling our C++ backend.

- **`Linear`**: Wraps `neonflux.matmul`. Includes He-Initialization ($N(0, \sqrt{2/n_{in}})$) and bias handling.
- **`Sequential`**: Chains layers for easy model definition.

```python
model = neon_nn.Sequential([
    neon_nn.Linear(128, 1024),
    neon_nn.ReLU(),
    neon_nn.Linear(1024, 10)
])
output = model.forward(input_batch)
```

---

## API Reference & Integration

### 1. Installation & Compilation

NeonFlux is a header-only library (mostly) but requires compiling the source files for the implementation.

**Compiler Flags**:
- Standard: C++17 or later (`-std=c++17`)
- Optimization: `-O3` (Critical for loop unrolling)
- Architecture: `-march=armv8-a+simd` (Enables NEON)

### 2. Core Components (`neonflux/vector.h`)

#### `struct FloatVector`
A RAII container wrapper ensuring 16-byte alignment compatible with NEON loads/stores.

```cpp
struct FloatVector {
    float* data;  // Pointer to aligned memory
    size_t size;  // Number of elements

    explicit FloatVector(size_t s); // Allocates aligned memory, initializes to 0.
    ~FloatVector();                 // Frees memory using AlignedAllocator.
};
```
- **Alignment Guarantee**: `data` is always 16-byte aligned (`addr % 16 == 0`).
- **Safety**: Throws `std::bad_alloc` if allocation fails.

#### `class AlignedAllocator` (`allocator.h`)
Low-level wrapper around `posix_memalign`.
- `static float* allocate(size_t size)`: return 16-byte aligned ptr.
- `static void deallocate(float* ptr)`: frees memory.

### 3. Vector Arithmetic (`neonflux/vector_math.h`)

Perform element-wise operations on `FloatVector`.
**Requirements**: Input vectors `a`, `b`, and `result` must have the same `size`.

| Function | Signature | Description |
|:---|:---|:---|
| `add` | `void add(const FloatVector& a, const FloatVector& b, FloatVector& res)` | `res[i] = a[i] + b[i]` |
| `sub` | `void sub(const FloatVector& a, const FloatVector& b, FloatVector& res)` | `res[i] = a[i] - b[i]` |
| `mul` | `void mul(const FloatVector& a, const FloatVector& b, FloatVector& res)` | `res[i] = a[i] * b[i]` |
| `scalar_mul` | `void scalar_mul(const FloatVector& a, float s, FloatVector& res)` | `res[i] = a[i] * s` |

**Note**: All functions handle "tail elements" (where `size % 4 != 0`) transparently.

### 4. Dot Product (`neonflux/dot_product.h`)

| Function | Performance | Description |
|:---|:---|:---|
| `dot_naive` | 1x (Baseline) | Standard scalar accumulation loop. |
| `dot_unrolled4` | **~10.8x Speedup** | Uses 4 parallel NEON accumulators and horizontal reduction. |

```cpp
float result = neonflux::dot_unrolled4(vec_a, vec_b);
```

### 5. Matrix Multiplication (`neonflux/gemm.h`)

Implements $C = A \times B$ assuming **row-major** storage.

#### `gemm_optimized`
```cpp
void gemm_optimized(int M, int N, int K, 
                    const float* A, const float* B, float* C);
```
- **Arguments**:
    - `M`, `N`: Dimensions of result matrix C (M x N).
    - `K`: Inner dimension (A is M x K, B is K x N).
    - `A`, `B`: Input pointers (must be valid arrays).
    - `C`: Output pointer (must be pre-allocated).
- **Features**:
    - **Packing**: B is repacked into contiguous panels to optimize for L1 cache.
    - **Micro-Kernel**: 4x4 register blocking.
    - **Tiling**: Loop blocking for L2 cache residency.
    - **Constraints**: Currently optimized for square matrices or sizes divisible by 4. Edge handling is basic.

### 6. Python API (`neonflux`)

```python
import neonflux
import numpy as np

A = np.random.rand(1024, 1024).astype(np.float32)
B = np.random.rand(1024, 1024).astype(np.float32)

# High-performance execution (Multi-threaded)
C = neonflux.matmul(A, B)
```
- **Requirements**: Inputs must be `float32` (single precision).
- **Return**: A new `numpy.ndarray` containing the result.

---

## Example Usage

### compiling
```bash
g++ -O3 -march=armv8-a+simd main.cpp src/*.cpp -o myapp
```

### main.cpp
```cpp
#include "neonflux/vector.h"
#include "neonflux/vector_math.h"
#include "neonflux/gemm.h"
#include <iostream>

using namespace neonflux;

int main() {
    // 1. Vector Math
    size_t N = 1024;
    FloatVector v1(N), v2(N), result(N);
    
    // Fill data...
    scalar_mul(v1, 2.0f, v1); // v1 = v1 * 2
    add(v1, v2, result);      // res = v1 + v2
    
    // 2. High Performance GEMM
    int M = 256;
    std::vector<float> A(M*M, 1.0f);
    std::vector<float> B(M*M, 2.0f);
    std::vector<float> C(M*M);
    
    gemm_optimized(M, M, M, A.data(), B.data(), C.data());
    
    std::cout << "Computation Complete." << std::endl;
    return 0;
}
```
