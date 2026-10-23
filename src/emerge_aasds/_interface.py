"""
Apple Accelerate Sparse Direct Solver Interface
Uses compiled C wrapper (libaccelerate_wrapper.dylib)
EXTENDED VERSION with tunable parameters + direct CSC path

Refactored to stay in CSC format throughout:
  - scipy.sparse matrices are converted to CSC (scipy's native efficient format)
  - CSC arrays are passed directly to Accelerate (which uses CSC internally)
  - For symmetric/hermitian, scipy.sparse.tril() extracts the lower triangle
  - No intermediate COO conversion
"""

import numpy as np
from scipy import sparse
from typing import Literal, Optional
import ctypes
import time
import platform
import os
import warnings

# Lowest macOS that provides the Sparse API used by the wrapper (see _build_wrapper.py)
MIN_MACOS = (15, 5)

# Check if we're on macOS
if platform.system() != 'Darwin':
    raise ImportError("AccelerateInterface only works on macOS")

_mac_ver = tuple(int(p) for p in platform.mac_ver()[0].split('.')[:2] if p.isdigit())
if _mac_ver and _mac_ver < MIN_MACOS:
    raise ImportError(
        f"emerge-aasds requires macOS {'.'.join(map(str, MIN_MACOS))} or newer "
        f"(running {platform.mac_ver()[0]})"
    )

# Load our compiled wrapper
wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libaccelerate_wrapper.dylib')
if not os.path.exists(wrapper_path):
    raise ImportError(
        f"Wrapper library not found: {wrapper_path}\n"
        "Reinstall the package, or for a source checkout run: "
        "python src/emerge_aasds/_build_wrapper.py"
    )

try:
    accel = ctypes.CDLL(wrapper_path)
except Exception as e:
    raise ImportError(f"Could not load wrapper library: {e}")

# Complex number type for C
class c_complex_double(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

# Sparse matrix attributes (bit-packed)
class SparseAttributes_t(ctypes.Structure):
    _fields_ = [("_backing", ctypes.c_uint32)]
    
    @classmethod
    def create(cls, kind=0):
        attrs = cls()
        attrs._backing = kind
        return attrs

class SparseAttributesComplex_t(ctypes.Structure):
    _fields_ = [("_backing", ctypes.c_uint32)]
    
    @classmethod
    def create(cls, kind=0):
        attrs = cls()
        attrs._backing = kind
        return attrs

# Dense matrix structures
class DenseMatrix_Double(ctypes.Structure):
    _fields_ = [
        ("rowCount", ctypes.c_int),
        ("columnCount", ctypes.c_int),
        ("columnStride", ctypes.c_int),
        ("attributes", SparseAttributes_t),
        ("data", ctypes.POINTER(ctypes.c_double))
    ]

class DenseMatrix_Complex_Double(ctypes.Structure):
    _fields_ = [
        ("rowCount", ctypes.c_int),
        ("columnCount", ctypes.c_int),
        ("columnStride", ctypes.c_int),
        ("attributes", SparseAttributesComplex_t),
        ("data", ctypes.POINTER(c_complex_double))
    ]

# Opaque types (just pointers)
class SparseMatrix_Double(ctypes.Structure):
    pass

class SparseMatrix_Complex_Double(ctypes.Structure):
    pass

class SparseOpaqueFactorization_Double(ctypes.Structure):
    pass

class SparseOpaqueFactorization_Complex_Double(ctypes.Structure):
    pass

# ============================================================================
# EXTENDED: AccelFactorOptions struct
# ============================================================================
class AccelFactorOptions(ctypes.Structure):
    _fields_ = [
        ("order_method", ctypes.c_int),      # AccelOrderMethod enum
        ("scaling_method", ctypes.c_int),    # AccelScalingMethod enum  
        ("pivot_tolerance", ctypes.c_double),
        ("zero_tolerance", ctypes.c_double)
    ]

# Ordering method enums
ACCEL_ORDER_DEFAULT = 0
ACCEL_ORDER_AMD = 1
ACCEL_ORDER_METIS = 2
ACCEL_ORDER_COLAMD = 3
ACCEL_ORDER_MTMETIS = 4

# Scaling method enums
ACCEL_SCALING_DEFAULT = 0
ACCEL_SCALING_NONE = 1

# Factorization types
SPARSE_FACTOR_CHOLESKY = 0          # SparseFactorizationCholesky
SPARSE_FACTOR_LDLT = 1              # SparseFactorizationLDLT (default LDLT, currently TPP)
SPARSE_FACTOR_LDLT_UNPIVOTED = 2    # SparseFactorizationLDLTUnpivoted
SPARSE_FACTOR_LDLT_SBK = 3          # SparseFactorizationLDLTSBK
SPARSE_FACTOR_LDLT_TPP = 4          # SparseFactorizationLDLTTPP
SPARSE_FACTOR_QR = 40               # SparseFactorizationQR
SPARSE_FACTOR_CHOLESKY_AT_A = 41    # SparseFactorizationCholeskyAtA
SPARSE_FACTOR_LU = 80               # SparseFactorizationLU (default LU, currently TPP)
SPARSE_FACTOR_LU_UNPIVOTED = 81     # SparseFactorizationLUUnpivoted
SPARSE_FACTOR_LU_SPP = 82           # SparseFactorizationLUSPP
SPARSE_FACTOR_LU_TPP = 83           # SparseFactorizationLUTPP

# Factorizations that work on one triangle of a symmetric/hermitian matrix.
# All others (LU, QR, CholeskyAtA) take the full matrix.
SYMMETRIC_FACTOR_TYPES = {
    SPARSE_FACTOR_CHOLESKY,
    SPARSE_FACTOR_LDLT,
    SPARSE_FACTOR_LDLT_UNPIVOTED,
    SPARSE_FACTOR_LDLT_SBK,
    SPARSE_FACTOR_LDLT_TPP,
}

# Kind types (matching _wrapper.h / Accelerate)
SPARSE_KIND_ORDINARY = 0
SPARSE_KIND_SYMMETRIC = 3
SPARSE_KIND_HERMITIAN = 4

# Triangle selection for symmetric/hermitian
SPARSE_TRIANGLE_UPPER = 0
SPARSE_TRIANGLE_LOWER = 1

# ============================================================================
# Setup function signatures - EXISTING API (kept for backward compatibility)
# ============================================================================
accel.accel_convert_from_coordinate_double.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_uint8,
    SparseAttributes_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double)
]
accel.accel_convert_from_coordinate_double.restype = ctypes.POINTER(SparseMatrix_Double)

accel.accel_convert_from_coordinate_complex_double.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_uint8,
    SparseAttributesComplex_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p
]
accel.accel_convert_from_coordinate_complex_double.restype = ctypes.POINTER(SparseMatrix_Complex_Double)

accel.accel_factor_double.argtypes = [ctypes.c_uint8, ctypes.POINTER(SparseMatrix_Double)]
accel.accel_factor_double.restype = ctypes.POINTER(SparseOpaqueFactorization_Double)

accel.accel_factor_complex_double.argtypes = [ctypes.c_uint8, ctypes.POINTER(SparseMatrix_Complex_Double)]
accel.accel_factor_complex_double.restype = ctypes.POINTER(SparseOpaqueFactorization_Complex_Double)

accel.accel_solve_double.argtypes = [
    ctypes.POINTER(SparseOpaqueFactorization_Double),
    DenseMatrix_Double,
    DenseMatrix_Double
]
accel.accel_solve_double.restype = None

accel.accel_solve_complex_double.argtypes = [
    ctypes.POINTER(SparseOpaqueFactorization_Complex_Double),
    DenseMatrix_Complex_Double,
    DenseMatrix_Complex_Double
]
accel.accel_solve_complex_double.restype = None

accel.accel_cleanup_matrix_double.argtypes = [ctypes.POINTER(SparseMatrix_Double)]
accel.accel_cleanup_matrix_double.restype = None

accel.accel_cleanup_matrix_complex_double.argtypes = [ctypes.POINTER(SparseMatrix_Complex_Double)]
accel.accel_cleanup_matrix_complex_double.restype = None

accel.accel_cleanup_factorization_double.argtypes = [ctypes.POINTER(SparseOpaqueFactorization_Double)]
accel.accel_cleanup_factorization_double.restype = None

accel.accel_cleanup_factorization_complex_double.argtypes = [ctypes.POINTER(SparseOpaqueFactorization_Complex_Double)]
accel.accel_cleanup_factorization_complex_double.restype = None

accel.accel_refactor_double.argtypes = [
    ctypes.POINTER(SparseMatrix_Double),
    ctypes.POINTER(SparseOpaqueFactorization_Double)
]
accel.accel_refactor_double.restype = None

accel.accel_refactor_complex_double.argtypes = [
    ctypes.POINTER(SparseMatrix_Complex_Double),
    ctypes.POINTER(SparseOpaqueFactorization_Complex_Double)
]
accel.accel_refactor_complex_double.restype = None

# ============================================================================
# EXTENDED API - Function signatures with parameters
# ============================================================================
accel.accel_get_default_options.argtypes = [ctypes.c_int]
accel.accel_get_default_options.restype = AccelFactorOptions

accel.accel_mtmetis_available.argtypes = []
accel.accel_mtmetis_available.restype = ctypes.c_int

accel.accel_complex_symmetric_available.argtypes = []
accel.accel_complex_symmetric_available.restype = ctypes.c_int

MTMETIS_AVAILABLE: bool = bool(accel.accel_mtmetis_available())
COMPLEX_SYMMETRIC_AVAILABLE: bool = bool(accel.accel_complex_symmetric_available())

# SparseStatus_t values
SPARSE_STATUS_MESSAGES = {
    -1: "factorization failed (matrix is not positive definite or pivoting failed)",
    -2: "matrix is singular",
    -3: "internal error in Accelerate",
    -4: "invalid parameter passed to Accelerate",
}

accel.accel_factor_double_ex.argtypes = [
    ctypes.c_uint8,
    ctypes.POINTER(SparseMatrix_Double),
    ctypes.POINTER(AccelFactorOptions)
]
accel.accel_factor_double_ex.restype = ctypes.POINTER(SparseOpaqueFactorization_Double)

accel.accel_factor_complex_double_ex.argtypes = [
    ctypes.c_uint8,
    ctypes.POINTER(SparseMatrix_Complex_Double),
    ctypes.POINTER(AccelFactorOptions)
]
accel.accel_factor_complex_double_ex.restype = ctypes.POINTER(SparseOpaqueFactorization_Complex_Double)

accel.accel_refactor_double_ex.argtypes = [
    ctypes.POINTER(SparseMatrix_Double),
    ctypes.POINTER(SparseOpaqueFactorization_Double),
    ctypes.POINTER(AccelFactorOptions)
]
accel.accel_refactor_double_ex.restype = None

accel.accel_refactor_complex_double_ex.argtypes = [
    ctypes.POINTER(SparseMatrix_Complex_Double),
    ctypes.POINTER(SparseOpaqueFactorization_Complex_Double),
    ctypes.POINTER(AccelFactorOptions)
]
accel.accel_refactor_complex_double_ex.restype = None

# ============================================================================
# CSC API - Direct construction from CSC arrays
# ============================================================================
accel.accel_create_from_csc_double.argtypes = [
    ctypes.c_int, ctypes.c_int,             # rowCount, columnCount
    ctypes.POINTER(ctypes.c_long),           # columnStarts
    ctypes.POINTER(ctypes.c_int),            # rowIndices
    ctypes.POINTER(ctypes.c_double),         # data
    ctypes.c_int, ctypes.c_int               # kind, triangle
]
accel.accel_create_from_csc_double.restype = ctypes.POINTER(SparseMatrix_Double)

accel.accel_create_from_csc_complex_double.argtypes = [
    ctypes.c_int, ctypes.c_int,             # rowCount, columnCount
    ctypes.POINTER(ctypes.c_long),           # columnStarts
    ctypes.POINTER(ctypes.c_int),            # rowIndices
    ctypes.c_void_p,                         # data (complex interleaved)
    ctypes.c_int, ctypes.c_int               # kind, triangle
]
accel.accel_create_from_csc_complex_double.restype = ctypes.POINTER(SparseMatrix_Complex_Double)

accel.accel_cleanup_csc_matrix_double.argtypes = [ctypes.POINTER(SparseMatrix_Double)]
accel.accel_cleanup_csc_matrix_double.restype = None

accel.accel_cleanup_csc_matrix_complex_double.argtypes = [ctypes.POINTER(SparseMatrix_Complex_Double)]
accel.accel_cleanup_csc_matrix_complex_double.restype = None

accel.accel_factorization_status_double.argtypes = [ctypes.POINTER(SparseOpaqueFactorization_Double)]
accel.accel_factorization_status_double.restype = ctypes.c_int

accel.accel_factorization_status_complex_double.argtypes = [ctypes.POINTER(SparseOpaqueFactorization_Complex_Double)]
accel.accel_factorization_status_complex_double.restype = ctypes.c_int


class _AccelerateInterface:
    """Apple Accelerate Sparse Direct Solver with tunable parameters.
    
    Uses direct CSC (Compressed Sparse Column) path to Accelerate.
    Scipy CSC arrays are passed straight through — no COO conversion.
    For symmetric/hermitian matrices, only the lower triangle is stored.
    """
    
    def __init__(self, 
                 factorization: Literal['LU','Cholesky','LDLT','QR','LUSBK']='LU', 
                 symmetry: Literal['symmetric','nonsymmetric','hermitian']='nonsymmetric',
                 ordering: str = 'default',
                 scaling: str = 'default',
                 pivot_tolerance: Optional[float] = None,
                 zero_tolerance: Optional[float] = None,
                 verbose=0):
        self._factorization: str = factorization
        self._symmetry: str = symmetry
        self._factored_obj = None
        self._sparse_matrix = None
        self._is_complex: bool = None
        self._n: int = None
        self._csc_created: bool = False  # Track which cleanup path to use
        
        # Persistent references to numpy arrays backing the SparseMatrix.
        # Accelerate does NOT copy CSC arrays, so we must prevent GC.
        self._kept_indptr = None
        self._kept_indices = None
        self._kept_data = None
        
        factor_map = {
            'cholesky': SPARSE_FACTOR_CHOLESKY,
            'qr': SPARSE_FACTOR_QR,
            'lu': SPARSE_FACTOR_LU,
            'luup': SPARSE_FACTOR_LU_UNPIVOTED,
            'luspp': SPARSE_FACTOR_LU_SPP,
            'lutpp': SPARSE_FACTOR_LU_TPP,
            'ldlt': SPARSE_FACTOR_LDLT,
            'ldltup': SPARSE_FACTOR_LDLT_UNPIVOTED,
            'ldlttpp': SPARSE_FACTOR_LDLT_TPP,
            'ldltsbk': SPARSE_FACTOR_LDLT_SBK,
        }
        self._factor_type = factor_map.get(factorization.lower())
        if self._factor_type is None:
            raise ValueError(f"Unknown factorization: {factorization}")
        if self._factor_type in SYMMETRIC_FACTOR_TYPES and symmetry == 'nonsymmetric':
            raise ValueError(
                f"Factorization '{factorization}' requires a symmetric or hermitian matrix; "
                "use LU or QR for nonsymmetric matrices."
            )
        # Factorization actually used; may fall back for complex symmetric on macOS < 26
        self._active_factor_type = self._factor_type

        # Map ordering string to enum
        ordering_map = {
            'default': ACCEL_ORDER_DEFAULT,
            'amd': ACCEL_ORDER_AMD,
            'metis': ACCEL_ORDER_METIS,
            'colamd': ACCEL_ORDER_COLAMD,
            'mtmetis': ACCEL_ORDER_MTMETIS,
        }
        self._ordering = ordering_map.get(ordering.lower(), ACCEL_ORDER_DEFAULT)
        if self._ordering == ACCEL_ORDER_MTMETIS and not MTMETIS_AVAILABLE:
            warnings.warn(
                "MTMetis ordering requires macOS 26 or newer; falling back to Metis.",
                RuntimeWarning,
                stacklevel=3,
            )

        # Map scaling string to enum
        scaling_map = {
            'default': ACCEL_SCALING_DEFAULT,
            'none': ACCEL_SCALING_NONE
        }
        self._scaling = scaling_map.get(scaling.lower(), ACCEL_SCALING_DEFAULT)
        
        self._pivot_tolerance = pivot_tolerance
        self._zero_tolerance = zero_tolerance
        self.verbose = verbose
    
    def _build_options(self) -> AccelFactorOptions:
        """Build options struct for C API"""
        opts = AccelFactorOptions()
        opts.order_method = self._ordering
        opts.scaling_method = self._scaling
        opts.pivot_tolerance = self._pivot_tolerance if self._pivot_tolerance is not None else -1.0
        opts.zero_tolerance = self._zero_tolerance if self._zero_tolerance is not None else -1.0
        return opts
    
    def _prepare_csc(self, A: sparse.spmatrix, symmetric_storage: bool) -> sparse.csc_matrix:
        """Convert to CSC and extract lower triangle for symmetric factorizations.

        Symmetric factorizations (Cholesky, LDLT) read only one triangle; the
        structure is built directly so Accelerate does not filter the other
        triangle out for us. LU and QR always take the full matrix.
        """
        if symmetric_storage:
            # Extract lower-triangular part (including diagonal)
            A = sparse.tril(A, format='csc')
        else:
            A = A.tocsc()
        
        # Ensure sorted indices (Accelerate requires sorted row indices per column)
        if not A.has_sorted_indices:
            A.sort_indices()
        
        return A
    
    def analyse(self, A: sparse.spmatrix):
        """Symbolic analysis — records matrix shape and type."""
        if not sparse.issparse(A):
            raise ValueError("A must be scipy sparse matrix")
        
        # Clean up any previous factorization
        self.destroy()
        
        self._n = A.shape[0]
        self._is_complex = np.iscomplexobj(A.data) if hasattr(A, 'data') else False
        self._analysed = True
        
        if self.verbose > 0:
            ctype = "complex" if self._is_complex else "real"
            print(f"Analyse ({self._factorization}, {ctype}): shape={A.shape}, nnz={A.nnz}")
    
    def factorize(self, A: sparse.spmatrix) -> None:
        """Numeric factorization via direct CSC path.
        
        Converts the input to CSC, extracts the lower triangle for
        symmetric/hermitian modes, and passes the arrays directly
        to Accelerate — no intermediate COO conversion.
        """
        if not hasattr(self, '_analysed') or not self._analysed:
            raise RuntimeError("Call analyse() first")
        
        t0 = time.time()
        
        # Determine complex type from input
        self._is_complex = np.iscomplexobj(A.data) if hasattr(A, 'data') else False
        
        # Complex symmetric (non-Hermitian) LDLT crashes inside Accelerate on macOS < 26
        if (self._factored_obj is None
                and self._is_complex
                and self._symmetry == 'symmetric'
                and self._factor_type in SYMMETRIC_FACTOR_TYPES
                and not COMPLEX_SYMMETRIC_AVAILABLE):
            warnings.warn(
                f"Complex symmetric '{self._factorization}' factorization requires macOS 26 "
                "or newer; falling back to LU.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._active_factor_type = SPARSE_FACTOR_LU_TPP
        elif self._factored_obj is None:
            self._active_factor_type = self._factor_type

        symmetric_storage = self._active_factor_type in SYMMETRIC_FACTOR_TYPES

        # Convert to CSC (with tril for symmetric factorizations)
        csc = self._prepare_csc(A, symmetric_storage)

        n = csc.shape[0]
        m = csc.shape[1]

        # Determine kind and triangle for Accelerate
        kind = SPARSE_KIND_ORDINARY
        triangle = SPARSE_TRIANGLE_LOWER  # default, ignored for ordinary
        if symmetric_storage:
            # A real hermitian matrix is just symmetric
            if self._symmetry == 'hermitian' and self._is_complex:
                kind = SPARSE_KIND_HERMITIAN
            else:
                kind = SPARSE_KIND_SYMMETRIC
        
        # Prepare CSC arrays with correct dtypes for Accelerate:
        #   columnStarts: int64 (c_long on macOS is 64-bit)
        #   rowIndices: int32
        #   data: float64 or complex128 (interleaved real/imag)
        indptr = csc.indptr.astype(np.int64, copy=False)
        indices = csc.indices.astype(np.int32, copy=False)
        
        # Build options struct
        opts = self._build_options()
        
        if self._is_complex:
            # Interleave real/imag as struct { double real; double imag; }
            data_c = np.empty(csc.nnz, dtype=[('real', np.float64), ('imag', np.float64)])
            data_c['real'] = csc.data.real
            data_c['imag'] = csc.data.imag
            
            # Keep references alive — Accelerate does NOT copy these
            self._kept_indptr = indptr
            self._kept_indices = indices
            self._kept_data = data_c
            
            new_sparse_matrix = accel.accel_create_from_csc_complex_double(
                n, m,
                indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                data_c.ctypes.data,
                kind, triangle
            )
            
            if self._factored_obj is None:
                self._factored_obj = accel.accel_factor_complex_double_ex(
                    self._active_factor_type, new_sparse_matrix, ctypes.byref(opts)
                )
            else:
                accel.accel_refactor_complex_double_ex(
                    new_sparse_matrix, self._factored_obj, ctypes.byref(opts)
                )
            status = accel.accel_factorization_status_complex_double(self._factored_obj)

            # Clean up previous sparse matrix wrapper (not the arrays)
            if self._sparse_matrix is not None:
                if self._csc_created:
                    accel.accel_cleanup_csc_matrix_complex_double(self._sparse_matrix)
                else:
                    accel.accel_cleanup_matrix_complex_double(self._sparse_matrix)
            
            self._sparse_matrix = new_sparse_matrix
            self._csc_created = True
            
        else:
            data_arr = csc.data.astype(np.float64, copy=False)
            
            # Keep references alive
            self._kept_indptr = indptr
            self._kept_indices = indices
            self._kept_data = data_arr
            
            new_sparse_matrix = accel.accel_create_from_csc_double(
                n, m,
                indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                data_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                kind, triangle
            )
            
            if self._factored_obj is None:
                self._factored_obj = accel.accel_factor_double_ex(
                    self._active_factor_type, new_sparse_matrix, ctypes.byref(opts)
                )
            else:
                accel.accel_refactor_double_ex(
                    new_sparse_matrix, self._factored_obj, ctypes.byref(opts)
                )
            status = accel.accel_factorization_status_double(self._factored_obj)

            # Clean up previous sparse matrix wrapper
            if self._sparse_matrix is not None:
                if self._csc_created:
                    accel.accel_cleanup_csc_matrix_double(self._sparse_matrix)
                else:
                    accel.accel_cleanup_matrix_double(self._sparse_matrix)
            
            self._sparse_matrix = new_sparse_matrix
            self._csc_created = True

        if status != 0:
            # Keep the analysis but drop the failed factorization so the next
            # factorize() starts fresh instead of refactoring a broken object.
            self.destroy()
            self._analysed = True
            reason = SPARSE_STATUS_MESSAGES.get(status, f"status {status}")
            raise RuntimeError(f"Accelerate {self._factorization} factorization failed: {reason}")

        if self.verbose > 0:
            print(f"Factorize ({self._factorization}): {time.time()-t0:.3f}s  "
                  f"[CSC direct, nnz={csc.nnz}]")
    
    def solve(self, b: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Solve using factorization - supports multiple RHS."""
        if self._factored_obj is None:
            raise RuntimeError("Call factorize() first")
        
        b = np.asarray(b)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False
        
        n, nrhs = b.shape
        t0 = time.time()
        
        if self._is_complex:
            b = np.asfortranarray(b)
            
            b_c = np.empty((n, nrhs), dtype=[('real', np.float64), ('imag', np.float64)], order='F')
            b_c['real'] = b.real
            b_c['imag'] = b.imag
            
            x_c = np.empty((n, nrhs), dtype=[('real', np.float64), ('imag', np.float64)], order='F')
            
            B_mat = DenseMatrix_Complex_Double()
            B_mat.rowCount = n
            B_mat.columnCount = nrhs
            B_mat.columnStride = n
            B_mat.attributes = SparseAttributesComplex_t.create()
            B_mat.data = b_c.ctypes.data_as(ctypes.POINTER(c_complex_double))
            
            X_mat = DenseMatrix_Complex_Double()
            X_mat.rowCount = n
            X_mat.columnCount = nrhs
            X_mat.columnStride = n
            X_mat.attributes = SparseAttributesComplex_t.create()
            X_mat.data = x_c.ctypes.data_as(ctypes.POINTER(c_complex_double))
            
            accel.accel_solve_complex_double(self._factored_obj, B_mat, X_mat)
            
            x = x_c['real'] + 1j * x_c['imag']
            
        else:
            b_f = np.asfortranarray(b, dtype=np.float64)
            x_f = np.zeros((n, nrhs), dtype=np.float64, order='F')
            
            B_mat = DenseMatrix_Double()
            B_mat.rowCount = n
            B_mat.columnCount = nrhs
            B_mat.columnStride = n
            B_mat.attributes = SparseAttributes_t.create()
            B_mat.data = b_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            X_mat = DenseMatrix_Double()
            X_mat.rowCount = n
            X_mat.columnCount = nrhs
            X_mat.columnStride = n
            X_mat.attributes = SparseAttributes_t.create()
            X_mat.data = x_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            accel.accel_solve_double(self._factored_obj, B_mat, X_mat)
            
            x = x_f
        
        if self.verbose > 0:
            print(f"Solve ({self._factorization}): {time.time()-t0:.3f}s")
        
        info = {'converged': True, 'residual': 0.0}
        if squeeze:
            x = x.squeeze()
        
        return x, info
    
    def destroy(self):
        """Cleanup solver resources"""
        if self._factored_obj is not None:
            if self._is_complex:
                accel.accel_cleanup_factorization_complex_double(self._factored_obj)
            else:
                accel.accel_cleanup_factorization_double(self._factored_obj)
            self._factored_obj = None
        
        if self._sparse_matrix is not None:
            if self._is_complex:
                if self._csc_created:
                    accel.accel_cleanup_csc_matrix_complex_double(self._sparse_matrix)
                else:
                    accel.accel_cleanup_matrix_complex_double(self._sparse_matrix)
            else:
                if self._csc_created:
                    accel.accel_cleanup_csc_matrix_double(self._sparse_matrix)
                else:
                    accel.accel_cleanup_matrix_double(self._sparse_matrix)
            self._sparse_matrix = None
        
        # Release numpy array references
        self._kept_indptr = None
        self._kept_indices = None
        self._kept_data = None
        self._csc_created = False
        self._analysed = False
    
    def __del__(self):
        self.destroy()