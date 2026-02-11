from __future__ import annotations
from typing import Union, Optional
from enum import Enum
from ._interface import _AccelerateInterface


class Factorization(Enum):
    """Supported matrix factorizations."""
    LU = "lu"
    CHOLESKY = "cholesky"
    LDLT = "ldlt"
    QR = "qr"

    @property
    def _backend_string(self) -> str:
        """String identifier expected by the backend."""
        return self.value

    @classmethod
    def _from_any(cls, value: Union["Factorization", str]) -> "Factorization":
        """
        Parse a factorization from an enum or string.
        Accepted strings are case-insensitive and may be abbreviated.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "lu": cls.LU,
                "chol": cls.CHOLESKY,
                "cholesky": cls.CHOLESKY,
                "ldlt": cls.LDLT,
                "qr": cls.QR,
            }
            if key in aliases:
                return aliases[key]
        raise ValueError(
            f"Invalid factorization '{value}'. "
            f"Expected one of {[e._backend_string for e in cls]}"
        )


class Symmetry(Enum):
    """Supported matrix symmetry types."""
    SYMMETRIC = "symmetric"
    NONSYMMETRIC = "nonsymmetric"
    HERMITIAN = "hermitian"

    @property
    def _backend_string(self) -> str:
        """String identifier expected by the backend."""
        return self.value

    @classmethod
    def _from_any(cls, value: Union["Symmetry", str]) -> "Symmetry":
        """
        Parse a symmetry type from an enum or string.
        Accepted strings are case-insensitive.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "symmetric": cls.SYMMETRIC,
                "sym": cls.SYMMETRIC,
                "nonsymmetric": cls.NONSYMMETRIC,
                "non-symmetric": cls.NONSYMMETRIC,
                "ns": cls.NONSYMMETRIC,
                "hermitian": cls.HERMITIAN,
                "herm": cls.HERMITIAN,
            }
            if key in aliases:
                return aliases[key]
        raise ValueError(
            f"Invalid symmetry '{value}'. "
            f"Expected one of {[e._backend_string for e in cls]}"
        )


class Ordering(Enum):
    """Fill-reducing ordering methods."""
    DEFAULT = "default"
    AMD = "amd"
    METIS = "metis"
    COLAMD = "colamd"
    MTMETIS = "mtmetis"

    @property
    def _backend_string(self) -> str:
        return self.value

    @classmethod
    def _from_any(cls, value: Union["Ordering", str]) -> "Ordering":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "default": cls.DEFAULT,
                "amd": cls.AMD,
                "metis": cls.METIS,
                "colamd": cls.COLAMD,
                "mtmetis": cls.MTMETIS,
            }
            if key in aliases:
                return aliases[key]
        raise ValueError(f"Invalid ordering '{value}'")


class Scaling(Enum):
    """Matrix scaling strategies."""
    DEFAULT = "default"
    NONE = "none"

    @property
    def _backend_string(self) -> str:
        return self.value

    @classmethod
    def _from_any(cls, value: Union["Scaling", str]) -> "Scaling":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "default": cls.DEFAULT,
                "none": cls.NONE,
            }
            if key in aliases:
                return aliases[key]
        raise ValueError(f"Invalid scaling '{value}'")


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
                f'LDLT and Cholesky factorization are currently not supported.'
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