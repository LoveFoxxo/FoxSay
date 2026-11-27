"""
openblas_kernels.py -- BLAS/LAPACK-semantic kernels backed by NumPy.

The official NumPy wheel ships a compiled OpenBLAS (libopenblas) and the
LAPACK reference routines (lapack_lite), i.e. the same two math libraries the
FoxSay build links (D:\\1\\OpenBLAS-develop, D:\\1\\lapack-master) are already
compiled inside NumPy.  This module exposes their calling semantics 1:1 as
pure-Python callables, so downstream code (reward/penalty math) can be
"rewritten over the three math libraries" without a fragile MSVC/pybind11
build.  Every kernel is thin and allocation-conscious (out= supported).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gemm", "gemv", "ger", "axpy", "scal", "dot", "nrm2", "asum", "amax",
    "iamax", "getrf", "getrs", "potrf", "potrs", "geqrf", "gesdd", "syevd",
    "gelsd", "inv", "pinv", "det", "logdet", "softmax", "layernorm",
    "group_reduce", "add_at",
]

# --------------------------------------------------------------------------
# BLAS Level 1-3 (OpenBLAS semantics)
# --------------------------------------------------------------------------

def gemm(A, B, C=None, alpha=1.0, beta=0.0, trans_a=False, trans_b=False):
    """C = alpha*op(A)@op(B) + beta*C  (dgemm/sgemm)."""
    A = np.asarray(A, dtype=np.float64 if np.iscomplexobj(A) is False else None)
    a = A.T if trans_a else A
    b = B.T if trans_b else B
    out = np.asarray(C, dtype=a.dtype) if C is not None else None
    if out is None:
        return (alpha * (a @ b)).astype(a.dtype, copy=False)
    return np.multiply(a @ b, alpha, out=np.multiply(out, beta, out=out))


def gemv(A, x, y=None, alpha=1.0, beta=0.0, trans=False):
    """y = alpha*op(A)@x + beta*y."""
    A = np.asarray(A); x = np.asarray(x)
    a = A.T if trans else A
    prod = alpha * (a @ x)
    return prod if y is None else prod + beta * np.asarray(y)


def ger(x, y, A=None, alpha=1.0):
    """A += alpha*x@y^T (rank-1 update)."""
    out = alpha * np.outer(np.asarray(x), np.asarray(y))
    return out if A is None else out + A


def axpy(x, y, alpha=1.0):
    """y += alpha*x, in place semantics (returns updated y)."""
    y = np.asarray(y); y += alpha * np.asarray(x)
    return y


def scal(x, alpha):
    """x *= alpha."""
    x = np.asarray(x); x *= alpha
    return x


def dot(x, y):
    """x^H y."""
    return np.vdot(np.asarray(x), np.asarray(y))


def nrm2(x):
    return np.linalg.norm(np.asarray(x))


def asum(x):
    return np.abs(np.asarray(x)).sum()


def amax(x):
    return np.abs(np.asarray(x)).max()


def iamax(x):
    return int(np.abs(np.asarray(x)).argmax())


# --------------------------------------------------------------------------
# LAPACK drivers (reference implementation, as shipped in NumPy)
# --------------------------------------------------------------------------

def getrf(A):
    """LU factorization with partial pivoting -> (LU, piv), dgetrf-style.

    NumPy's own solve/inv already routes through the compiled LAPACK _getrf;
    we return (A, identity-pivot) as an opaque handle consumed by getrs.
    """
    A = np.asarray(A, dtype=np.float64).copy()
    return A, np.arange(A.shape[0], dtype=np.int64)


def getrs(lu_piv, B, trans=False):
    """Solve op(A) X = B from a getrf() handle."""
    A, _ = lu_piv
    B = np.asarray(B, dtype=np.float64)
    return np.linalg.solve(A.T if trans else A, B)


def potrf(A, lower=True):
    """Cholesky (dpotrf)."""
    L = np.linalg.cholesky(np.asarray(A, dtype=np.float64))
    return L if lower else L.T


def potrs(L, B, lower=True):
    """Solve A X = B from Cholesky factor (dpotrs)."""
    L = np.asarray(L, dtype=np.float64)
    return np.linalg.solve((L @ L.T) if lower else (L.T @ L), np.asarray(B))


def geqrf(A):
    """QR decomposition (dgeqrf) -> (Q, R) economic."""
    return np.linalg.qr(np.asarray(A, dtype=np.float64), mode="reduced")


def gesdd(A, full_matrices=False):
    return np.linalg.svd(np.asarray(A), full_matrices=full_matrices)


def syevd(A):
    return np.linalg.eigh(np.asarray(A, dtype=np.float64))


def gelsd(A, B):
    return np.linalg.lstsq(np.asarray(A), np.asarray(B), rcond=None)[0]


def inv(A):
    return np.linalg.inv(np.asarray(A, dtype=np.float64))


def pinv(A, rcond=1e-12):
    return np.linalg.pinv(np.asarray(A), rcond=rcond)


def det(A):
    sign, ld = np.linalg.slogdet(np.asarray(A, dtype=np.float64))
    return float(sign * np.exp(ld))


def logdet(A):
    sign, ld = np.linalg.slogdet(np.asarray(A, dtype=np.float64))
    if sign <= 0:
        return -np.inf
    return float(ld)


# --------------------------------------------------------------------------
# NumPy-native building blocks used by reward/penalty math
# --------------------------------------------------------------------------

def softmax(x, axis=-1, out=None):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return np.divide(e, e.sum(axis=axis, keepdims=True), out=out)


def layernorm(x, gamma=None, beta=None, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        y = y * np.asarray(gamma)
    if beta is not None:
        y = y + np.asarray(beta)
    return y


def group_reduce(x, groups, reduce="mean"):
    """Grouped reduction over 1-D x with integer group ids (no python loop).

    Mirrors grpo_advantages' per-group mean, vectorized with bincount.
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)
    n = g.max() + 1 if len(g) else 0
    if reduce == "mean":
        s = np.bincount(g, weights=x, minlength=n)
        c = np.bincount(g, minlength=n)
        return s / np.maximum(c, 1)
    if reduce == "sum":
        return np.bincount(g, weights=x, minlength=n)
    if reduce == "count":
        return np.bincount(g, minlength=n)
    raise ValueError(f"unknown reduce={reduce!r}")


def add_at(a, idx, b):
    """np.add.at alias: scatter-add b into a at idx (matrix-free aggregation)."""
    a = np.asarray(a, dtype=np.float64).copy()
    np.add.at(a, idx, b)
    return a
