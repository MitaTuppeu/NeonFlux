#include "../include/neonflux/activations.h"
#include "../include/neonflux/gemm.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

namespace py = pybind11;

// The Python-facing wrapper
py::array_t<float> gemm_py(py::array_t<float> A, py::array_t<float> B) {
  py::buffer_info bufA = A.request();
  py::buffer_info bufB = B.request();

  if (bufA.ndim != 2 || bufB.ndim != 2)
    throw runtime_error("Inputs must be 2D matrices");

  int M = bufA.shape[0];
  int K = bufA.shape[1];
  int N = bufB.shape[1];

  if (bufB.shape[0] != K)
    throw runtime_error("Dimension mismatch: A.cols != B.rows");
  auto result = py::array_t<float>({M, N});
  py::buffer_info bufC = result.request();
  memset(bufC.ptr, 0, M * N * sizeof(float));
  neonflux::gemm_optimized(M, N, K, static_cast<float *>(bufA.ptr), static_cast<float *>(bufB.ptr), static_cast<float *>(bufC.ptr));

  return result;
}

// ReLU Wrapper
py::array_t<float> relu_py(py::array_t<float> input) {
  py::buffer_info buf = input.request();
  auto result = py::array_t<float>(buf.shape);

  py::buffer_info res_buf = result.request();
  neonflux::relu_optimized(static_cast<float *>(buf.ptr), static_cast<float *>(res_buf.ptr), static_cast<int>(buf.size));

  return result;
}

PYBIND11_MODULE(neonflux, m) {
  m.doc() = "NeonFlux: High-Performance ARM64 Linear Algebra";
  m.def("matmul", &gemm_py, "Compute C = A * B using NEON + OpenMP", py::arg("A"), py::arg("B"));
  m.def("relu", &relu_py, "Apply ReLU activation (Vectorized + OpenMP)", py::arg("input"));
}
