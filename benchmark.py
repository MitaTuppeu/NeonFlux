import neonflux
import numpy as np
import time

def benchmark(N):
    print(f"\nBenchmarking Matrix Multiplication (N={N}x{N})...")
    
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # 1. NumPy (Reference - uses Accelerate/BLAS backend on macOS)
    start = time.time()
    C_ref = np.dot(A, B)
    end = time.time()
    numpy_time = end - start
    print(f"NumPy (BLAS): {numpy_time:.4f} sec | {(2*N**3*1e-9)/numpy_time:.2f} GFLOPS")

    # 2. NeonFlux 
    start = time.time()
    C_opt = neonflux.matmul(A, B)
    end = time.time()
    neon_time = end - start
    print(f"NeonFlux: {neon_time:.4f} sec | {(2*N**3*1e-9)/neon_time:.2f} GFLOPS")

    if np.allclose(C_ref, C_opt, atol=1e-3, rtol=1e-3):
         print("Verification Passed!")
    else:
         diff = np.abs(C_ref - C_opt)
         print(f"Verification FAILED! Max diff: {np.max(diff)}")

    return numpy_time, neon_time

if __name__ == "__main__":
    # Warmup
    benchmark(256)
    
    # Real test
    benchmark(1024)
    benchmark(2048)
