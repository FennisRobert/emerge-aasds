
from enum import Enum
from typing import Union

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

