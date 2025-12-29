from setuptools import setup, Extension
import pybind11
import sys
import subprocess
import os

# Helper to find header/lib paths
def get_brew_prefix(package):
    try:
        res = subprocess.run(['brew', '--prefix', package], check=True, capture_output=True, text=True)
        return res.stdout.strip()
    except:
        return f"/opt/homebrew/opt/{package}"

# Compiler flags for ARM64 + OpenMP optimization
cxx_args = ['-O3', '-march=armv8-a+simd', '-std=c++17']
link_args = []

# macOS specific handling
if sys.platform == 'darwin':
    omp_prefix = get_brew_prefix('libomp')
    cxx_args += ['-Xpreprocessor', '-fopenmp', f'-I{omp_prefix}/include']
    link_args = ['-Xpreprocessor', '-fopenmp', f'-L{omp_prefix}/lib', '-lomp']
else:
    cxx_args += ['-fopenmp']
    link_args += ['-fopenmp']

start_dir = os.getcwd()
src_dir = os.path.join(start_dir, "src")

# Need to include all source files that gemm depends on
sources = [
    'src/bindings.cpp',
    'src/gemm.cpp', 
    'src/activations.cpp',
    'src/dot_product.cpp',  
    'src/vector_math.cpp'
]

ext_modules = [
    Extension(
        'neonflux',
        sources,
        include_dirs=[pybind11.get_include(), 'include'],
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
        language='c++'
    ),
]

setup(
    name='neonflux',
    version='0.1',
    description='High Performance ARM64 GEMM Library',
    ext_modules=ext_modules,
)
