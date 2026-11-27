"""
mathlib -- OpenBLAS / LAPACK / Armadillo semantics as a pure-Python package.

The three C/C++ math libraries in this workspace
  - D:\\1\\OpenBLAS-develop   (BLAS)
  - D:\\1\\lapack-master       (LAPACK)
  - D:\\1\\armadillo-code      (Armadillo C++ templates)
are exposed here through the compiled OpenBLAS + LAPACK reference routines
shipped inside the official NumPy wheel, wrapped with 1:1 calling semantics
(openblas_kernels) and Armadillo-style containers (armadillo_types).  The
reward/penalty math of FoxSay.gan is rewritten on top of this package so it
inherits BLAS-level speed without an MSVC/pybind11 build step.

Submodules:
  openblas_kernels : gemm/gemv/axpy/dot/nrm2, getrf/potrf/geqrf/svd/eigh ...
  armadillo_types  : Mat / Col / Row containers with .t()/.i()/submat
  reward_math      : vectorized rewrite of FoxSay.gan.reward
  penalty_math     : vectorized rewrite of the penalty terms (truncation,
                     efficiency overuse, verifier rank loss)
"""

from . import armadillo_types, openblas_kernels
from .armadillo_types import Col, Mat, Row, col, mat, row

__all__ = [
    "openblas_kernels", "armadillo_types",
    "Mat", "Col", "Row", "mat", "col", "row",
]
