from __future__ import annotations

from typing import Union
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
            f"Expected one of {[e.backend_string for e in cls]}"
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
            f"Expected one of {[e.backend_string for e in cls]}"
        )



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
    """

    def __init__(
        self,
        factorization: Union[Factorization, str] = Factorization.LU,
        symmetry: Union[Symmetry, str] = Symmetry.NONSYMMETRIC,
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
        verbose
            Enable verbose backend output.
        """
        fac = Factorization._from_any(factorization)
        sym = Symmetry._from_any(symmetry)
        if fac is Factorization.LDLT or fac is Factorization.CHOLESKY:
            raise ValueError(f'LDLT and Cholesky factorization are currently not supported.')
        verbosity = 1 if verbose else 0

        super().__init__(fac._backend_string, sym._backend_string, verbosity)