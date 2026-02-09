"""
Apple Accelerate Sparse Direct Solver Interface
Uses compiled C wrapper (libaccelerate_wrapper.dylib)
"""

import numpy as np
from scipy import sparse
import ctypes
import time
import platform
import os

# Check if we're on macOS
if platform.system() != 'Darwin':
    raise RuntimeError("AccelerateInterface only works on macOS")

# Load our compiled wrapper
wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libaccelerate_wrapper.dylib')
if not os.path.exists(wrapper_path):
    raise RuntimeError(f"Wrapper library not found: {wrapper_path}\nRun build_wrapper.sh first!")

try:
    accel = ctypes.CDLL(wrapper_path)
except Exception as e:
    raise RuntimeError(f"Could not load wrapper library: {e}")

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

# Factorization types
SPARSE_FACTOR_CHOLESKY = 0
SPARSE_FACTOR_LDLT = 1
SPARSE_FACTOR_QR = 40
SPARSE_FACTOR_LU = 80

# Kind types
SPARSE_KIND_ORDINARY = 0
SPARSE_KIND_SYMMETRIC = 3
SPARSE_KIND_HERMITIAN = 4

# Setup function signatures
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


class AccelerateInterface:
    """Apple Accelerate Sparse Direct Solver"""
    
    def __init__(self, factorization='lu', symmetry='nonsymmetric', verbose=0):
        self._factorization = factorization
        self._symmetry = symmetry
        self._factored_obj = None
        self._sparse_matrix = None
        self._is_complex = None
        self._n = None
        
        factor_map = {
            'cholesky': SPARSE_FACTOR_CHOLESKY,
            'ldlt': SPARSE_FACTOR_LDLT,
            'qr': SPARSE_FACTOR_QR,
            'lu': SPARSE_FACTOR_LU,
        }
        self._factor_type = factor_map.get(factorization.lower())
        if self._factor_type is None:
            raise ValueError(f"Unknown factorization: {factorization}")
        
        self.verbose = verbose
    
    def analyse(self, A):
        """Symbolic factorization"""
        if not sparse.issparse(A):
            raise ValueError("A must be scipy sparse matrix")
        
        # Clean up any previous factorization
        self.destroy()
        
        A_coo = A.tocoo()
        self._n = A_coo.shape[0]
        self._is_complex = np.iscomplexobj(A_coo.data)
        
        if self.verbose > 0:
            t0 = time.time()
            ctype = "complex" if self._is_complex else "real"
            print(f"Analyse ({self._factorization}, {ctype}): ", end='', flush=True)
        
        self._A_coo = A_coo
        
        if self.verbose > 0:
            print(f"{time.time()-t0:.3f}s")
    
    def factorize(self, A):
        """Numeric factorization"""
        if self._A_coo is None:
            raise RuntimeError("Call analyse() first")
        
        A_coo = A.tocoo()
        
        t0 = time.time()
        
        kind = SPARSE_KIND_ORDINARY
        if self._symmetry == 'symmetric':
            kind = SPARSE_KIND_SYMMETRIC
        elif self._symmetry == 'hermitian':
            kind = SPARSE_KIND_HERMITIAN
        
        n = A_coo.shape[0]
        nnz = A_coo.nnz
        
        row_arr = A_coo.row.astype(np.int32)
        col_arr = A_coo.col.astype(np.int32)
        
        if self._is_complex:
            data_c = np.empty(nnz, dtype=[('real', np.float64), ('imag', np.float64)])
            data_c['real'] = A_coo.data.real
            data_c['imag'] = A_coo.data.imag
            
            attrs = SparseAttributesComplex_t.create(kind=kind)
            
            new_sparse_matrix = accel.accel_convert_from_coordinate_complex_double(
                n, n, nnz, 1, attrs,
                row_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                col_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                data_c.ctypes.data
            )
            
            if self._factored_obj is None:
                self._factored_obj = accel.accel_factor_complex_double(
                    self._factor_type, new_sparse_matrix
                )
            else:
                accel.accel_refactor_complex_double(new_sparse_matrix, self._factored_obj)
            
            if self._sparse_matrix is not None:
                accel.accel_cleanup_matrix_complex_double(self._sparse_matrix)
            
            self._sparse_matrix = new_sparse_matrix
            
        else:
            data_arr = A_coo.data.astype(np.float64)
            attrs = SparseAttributes_t.create(kind=kind)
            
            new_sparse_matrix = accel.accel_convert_from_coordinate_double(
                n, n, nnz, 1, attrs,
                row_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                col_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                data_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            
            if self._factored_obj is None:
                self._factored_obj = accel.accel_factor_double(
                    self._factor_type, new_sparse_matrix
                )
            else:
                accel.accel_refactor_double(new_sparse_matrix, self._factored_obj)
            
            if self._sparse_matrix is not None:
                accel.accel_cleanup_matrix_double(self._sparse_matrix)
            
            self._sparse_matrix = new_sparse_matrix
        
        if self.verbose > 0:
            print(f"Factorize ({self._factorization}): {time.time()-t0:.3f}s")
    
    def solve(self, b):
        """Solve using factorization - FIXED for multiple RHS"""
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
            # Solve each RHS separately for now (safer)
            x = np.zeros((n, nrhs), dtype=np.complex128)
            
            for i in range(nrhs):
                b_vec = np.ascontiguousarray(b[:, i])
                x_vec = np.zeros(n, dtype=np.complex128)
                
                b_c = np.empty(n, dtype=[('real', np.float64), ('imag', np.float64)])
                b_c['real'] = b_vec.real
                b_c['imag'] = b_vec.imag
                
                x_c = np.empty(n, dtype=[('real', np.float64), ('imag', np.float64)])
                
                B_mat = DenseMatrix_Complex_Double()
                B_mat.rowCount = n
                B_mat.columnCount = 1
                B_mat.columnStride = n
                B_mat.attributes = SparseAttributesComplex_t.create()
                B_mat.data = b_c.ctypes.data_as(ctypes.POINTER(c_complex_double))
                
                X_mat = DenseMatrix_Complex_Double()
                X_mat.rowCount = n
                X_mat.columnCount = 1
                X_mat.columnStride = n
                X_mat.attributes = SparseAttributesComplex_t.create()
                X_mat.data = x_c.ctypes.data_as(ctypes.POINTER(c_complex_double))
                
                accel.accel_solve_complex_double(self._factored_obj, B_mat, X_mat)
                
                x[:, i] = x_c['real'] + 1j * x_c['imag']
            
        else:
            # Real solve - also one at a time
            x = np.zeros((n, nrhs), dtype=np.float64)
            
            for i in range(nrhs):
                b_vec = np.ascontiguousarray(b[:, i])
                x_vec = np.zeros(n, dtype=np.float64)
                
                B_mat = DenseMatrix_Double()
                B_mat.rowCount = n
                B_mat.columnCount = 1
                B_mat.columnStride = n
                B_mat.attributes = SparseAttributes_t.create()
                B_mat.data = b_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                
                X_mat = DenseMatrix_Double()
                X_mat.rowCount = n
                X_mat.columnCount = 1
                X_mat.columnStride = n
                X_mat.attributes = SparseAttributes_t.create()
                X_mat.data = x_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                
                accel.accel_solve_double(self._factored_obj, B_mat, X_mat)
                
                x[:, i] = x_vec
        
        if self.verbose > 0:
            print(f"Solve ({self._factorization}): {time.time()-t0:.3f}s")
        
        info = {'converged': True, 'residual': 0.0}
        return (x.squeeze() if squeeze else x), info
    
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
                accel.accel_cleanup_matrix_complex_double(self._sparse_matrix)
            else:
                accel.accel_cleanup_matrix_double(self._sparse_matrix)
            self._sparse_matrix = None
    
    def __del__(self):
        self.destroy()


def create_accelerate_solver(factorization='lu', symmetry='nonsymmetric', verbose=0):
    return AccelerateInterface(factorization=factorization, symmetry=symmetry, verbose=verbose)