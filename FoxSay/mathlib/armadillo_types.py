"""
armadillo_types.py -- Armadillo-semantics matrix containers on NumPy.

Mirrors the subset of the Armadillo C++ template library API used by the
reward/penalty rewrite: Mat / Col / Row with element access, .t()/.i(),
submatrices, and the usual overloaded arithmetic.  Storage is a NumPy array
so every heavy operation lands in the compiled OpenBLAS/LAPACK kernels of
numpy (see openblas_kernels.py), giving Armadillo-level ergonomics at
BLAS-level speed without a C++ build.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Mat", "Col", "Row", "mat", "col", "row"]


class _Base:
    __slots__ = ("_a",)

    def __init__(self, data):
        if isinstance(data, _Base):
            self._a = np.asarray(data._a, dtype=np.float64)
        else:
            self._a = np.asarray(data, dtype=np.float64)

    # -- shape ---------------------------------------------------------
    @property
    def shape(self):
        return self._a.shape

    @property
    def size(self):
        return self._a.size

    @property
    def n_rows(self):
        return self._a.shape[0]

    @property
    def n_cols(self):
        return self._a.shape[1]

    # -- access --------------------------------------------------------
    def __getitem__(self, key):
        return self._a[key]

    def __setitem__(self, key, value):
        self._a[key] = value

    def at(self, i, j=0):
        return self._a[i, j]

    def set_at(self, i, j, v):
        self._a[i, j] = v

    def __array__(self, dtype=None):
        return np.asarray(self._a, dtype=dtype)

    # -- arithmetic (operator overloads, Armadillo-style) ---------------
    def __add__(self, o):
        return type(self)(self._a + _as_array(o))

    def __sub__(self, o):
        return type(self)(self._a - _as_array(o))

    def __mul__(self, o):
        return type(self)(self._a * _as_array(o))

    def __truediv__(self, o):
        return type(self)(self._a / _as_array(o))

    def __matmul__(self, o):
        return Mat(self._a @ _as_array(o))

    def __neg__(self):
        return type(self)(-self._a)

    def __radd__(self, o):
        return type(self)(_as_array(o) + self._a)

    def __rmul__(self, o):
        return type(self)(_as_array(o) * self._a)

    def __len__(self):
        return self._a.shape[0]


def _as_array(o):
    return o._a if isinstance(o, _Base) else np.asarray(o, dtype=np.float64)


class Mat(_Base):
    """Armadillo-style dense matrix (column-major spirit; NumPy storage)."""

    def __init__(self, data, n_rows=None, n_cols=None):
        super().__init__(data)
        if n_rows is not None and self._a.size == n_rows * n_cols:
            self._a = self._a.reshape(n_rows, n_cols)

    def t(self):
        return Mat(self._a.T)

    def i(self):
        return Mat(np.linalg.inv(self._a))

    def col(self, j):
        return Col(self._a[:, j].copy())

    def row(self, i):
        return Row(self._a[i, :].copy())

    def submat(self, r0, r1, c0, c1):
        return Mat(self._a[r0:r1, c0:c1].copy())

    def rows(self, r0, r1):
        return Mat(self._a[r0:r1, :].copy())

    def cols(self, c0, c1):
        return Mat(self._a[:, c0:c1].copy())

    def solve(self, B):
        """A \\ B via LAPACK gesv."""
        return Mat(np.linalg.solve(self._a, _as_array(B)))

    def norm(self, p="fro"):
        return float(np.linalg.norm(self._a, ord=p))

    def dot(self, o):
        return Mat(self._a @ _as_array(o))

    def __repr__(self):
        return f"Mat({self._a.shape})\n{self._a}"


class Col(_Base):
    """Armadillo-style column vector."""

    def __init__(self, data):
        super().__init__(data)
        self._a = self._a.reshape(-1, 1)

    def t(self):
        return Row(self._a.T.copy())

    def __repr__(self):
        return f"Col({self._a.shape[0]})\n{self._a}"


class Row(_Base):
    """Armadillo-style row vector."""

    def __init__(self, data):
        super().__init__(data)
        self._a = self._a.reshape(1, -1)

    def t(self):
        return Col(self._a.T.copy())

    def __repr__(self):
        return f"Row({self._a.shape[1]})\n{self._a}"


# convenience constructors (armadillo-like names)
def mat(data, n_rows=None, n_cols=None):
    return Mat(data, n_rows, n_cols)


def col(data):
    return Col(data)


def row(data):
    return Row(data)
