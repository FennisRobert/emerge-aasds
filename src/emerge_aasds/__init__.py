"""
emerge-aasds: Apple Accelerate Sparse Direct Solver

Fast sparse linear system solver for macOS using Apple's native Accelerate framework.
Provides 2-4x speedup over UMFPACK on Apple Silicon, with no 32-bit limitations.

Example:
    >>> from emerge_aasds import AccelerateInterface
    >>> solver = AccelerateInterface(factorization='lu', verbose=1)
    >>> solver.analyse(A)
    >>> solver.factorize(A)
    >>> x, info = solver.solve(b)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("emerge-aasds")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"

from .utilities import Factorization, Symmetry, Ordering, Scaling
from .interface import AccelerateInterface

__all__ = ['AccelerateInterface', 'Ordering', 'Scaling', 'Factorization', 'Symmetry']
