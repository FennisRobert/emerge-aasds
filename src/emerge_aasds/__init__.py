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

__version__ = "0.1.0"

from ._interface import AccelerateInterface, create_accelerate_solver

__all__ = ['AccelerateInterface', 'create_accelerate_solver']