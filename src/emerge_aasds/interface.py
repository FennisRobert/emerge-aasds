from __future__ import annotations
from typing import Union, Optional
from ._interface import _AccelerateInterface
from .utilities import Factorization, Symmetry, Ordering, Scaling

class AccelerateInterface(_AccelerateInterface):
    """
    High-level Python interface to the Accelerate linear solver backend.

    Parameters
    ----------
    factorization
        Matrix factorization to use. May be a `Factorization` enum or
        a string such as ``"lu"``, ``"cholesky"``, ``"ldlt"``, or ``"qr"``.
    symmetry
        Matrix symmetry type. May be a `Symmetry` enum or a string such as
        ``"symmetric"``, ``"nonsymmetric"``, or ``"hermitian"``.
    ordering
        Fill-reducing ordering algorithm (default: "default").
        Options: "default", "amd", "metis", "colamd"
    scaling
        Matrix scaling strategy (default: "default").
        Options: "default", "none"
    pivot_tolerance
        Pivot threshold (default: auto = 0.01).
        Range: 0.0 to 1.0. Higher = more stable.
    zero_tolerance
        Zero detection threshold (default: auto).
    verbose
        Enable verbose backend output.
    """

    def __init__(
        self,
        factorization: Union[Factorization, str] = Factorization.LU,
        symmetry: Union[Symmetry, str] = Symmetry.NONSYMMETRIC,
        ordering: Union[Ordering, str] = Ordering.DEFAULT,
        scaling: Union[Scaling, str] = Scaling.DEFAULT,
        pivot_tolerance: Optional[float] = None,
        zero_tolerance: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Create an Accelerate solver interface.

        Parameters
        ----------
        factorization
            Matrix factorization to use. May be a `Factorization` enum or
            a string such as ``"lu"``, ``"cholesky"``, ``"ldlt"``, or ``"qr"``.
        symmetry
            Matrix symmetry type. May be a `Symmetry` enum or a string such as
            ``"symmetric"``, ``"nonsymmetric"``, or ``"hermitian"``.
        ordering
            Fill-reducing ordering algorithm.
        scaling
            Matrix scaling strategy.
        pivot_tolerance
            Pivot threshold. None = auto.
        zero_tolerance
            Zero detection threshold. None = auto.
        verbose
            Enable verbose backend output.
        """
        # Parse enums
        fac = Factorization._from_any(factorization)
        sym = Symmetry._from_any(symmetry)
        ord_ = Ordering._from_any(ordering)
        scl = Scaling._from_any(scaling)

        # Check for unsupported factorizations
        if fac is Factorization.LDLT or fac is Factorization.CHOLESKY:
            raise ValueError(
                'LDLT and Cholesky factorization are currently not supported.'
            )

        verbosity = 1 if verbose else 0

        # Call parent with all parameters
        super().__init__(
            fac._backend_string,
            sym._backend_string,
            ord_._backend_string,
            scl._backend_string,
            pivot_tolerance,
            zero_tolerance,
            verbosity
        )