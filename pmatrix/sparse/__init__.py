"""
2D sparse arrays.
CSC: Compressed Sparse Column matrix.
CSR: Compressed Sparse row matrix.
DIA: Diagonal sparse matrix.

All these matrices allow matrix multiplaction with X @ Y, where X and Y are any sparse matrix, or DMatrix.
"""

from ._csc import CSC
from ._csr import CSR
from ._dia import DIA