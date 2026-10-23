"""
Correctness tests for every factorization / symmetry / dtype combination.

Run with: pytest tests/
"""
import numpy as np
import pytest
import scipy.sparse as sp

from emerge_aasds import AccelerateInterface
from emerge_aasds._interface import COMPLEX_SYMMETRIC_AVAILABLE

N = 300
RTOL = 1e-8

LU_FACTORS = ['lu', 'luup', 'luspp', 'lutpp', 'qr']
LDLT_FACTORS = ['ldlt', 'ldltup', 'ldltsbk', 'ldlttpp']
SYMMETRIC_FACTORS = ['cholesky'] + LDLT_FACTORS


def _random(n, complex_, seed):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.02, format='csr', random_state=rng)
    if complex_:
        A = A + 1j * sp.random(n, n, density=0.02, format='csr', random_state=rng)
    return A


def nonsymmetric(complex_=False, seed=0):
    """Diagonally dominant, so unpivoted LU works too."""
    A = _random(N, complex_, seed)
    return (A + sp.eye(N) * (abs(A).sum(axis=1).max() + 1.0)).tocsc()


def symmetric_pd(complex_=False, hermitian=False, seed=0):
    """Symmetric (A = A^T) or hermitian (A = A^H) positive definite."""
    B = _random(N, complex_, seed)
    A = B @ B.conj().T if hermitian else B + B.T
    return (A + sp.eye(N) * (abs(A).sum(axis=1).max() + 1.0)).tocsc()


def symmetric_indefinite(complex_=False, hermitian=False, seed=0):
    B = _random(N, complex_, seed)
    A = B + (B.conj().T if hermitian else B.T)
    d = np.where(np.arange(N) % 2 == 0, 1.0, -1.0) * (abs(A).sum(axis=1).A1.max() + 1.0)
    return (A + sp.diags(d)).tocsc()


def check_solve(A, factorization, symmetry, nrhs=3):
    solver = AccelerateInterface(factorization=factorization, symmetry=symmetry)
    solver.analyse(A)
    solver.factorize(A)

    rng = np.random.default_rng(1)
    dtype = np.complex128 if np.iscomplexobj(A.data) else np.float64
    b = rng.standard_normal((N, nrhs)).astype(dtype)
    if dtype == np.complex128:
        b = b + 1j * rng.standard_normal((N, nrhs))

    # Single RHS
    x, _ = solver.solve(b[:, 0])
    assert x.shape == (N,)
    assert np.linalg.norm(A @ x - b[:, 0]) <= RTOL * np.linalg.norm(b[:, 0])

    # Multiple RHS
    X, _ = solver.solve(b)
    assert X.shape == (N, nrhs)
    assert np.linalg.norm(A @ X - b) <= RTOL * np.linalg.norm(b)

    # Refactor with new values on the same pattern
    A2 = A.copy()
    A2.data = A2.data * 2.0
    solver.factorize(A2)
    x2, _ = solver.solve(b[:, 0])
    assert np.linalg.norm(A2 @ x2 - b[:, 0]) <= RTOL * np.linalg.norm(b[:, 0])

    solver.destroy()


@pytest.mark.parametrize('complex_', [False, True], ids=['real', 'complex'])
@pytest.mark.parametrize('factorization', LU_FACTORS)
def test_nonsymmetric(factorization, complex_):
    check_solve(nonsymmetric(complex_), factorization, 'nonsymmetric')


@pytest.mark.parametrize('factorization', SYMMETRIC_FACTORS + LU_FACTORS)
def test_real_symmetric_pd(factorization):
    check_solve(symmetric_pd(), factorization, 'symmetric')


@pytest.mark.parametrize('factorization', ['ldlt', 'ldltsbk', 'ldlttpp'])
def test_real_symmetric_indefinite(factorization):
    check_solve(symmetric_indefinite(), factorization, 'symmetric')


@pytest.mark.parametrize('factorization', SYMMETRIC_FACTORS + LU_FACTORS)
def test_complex_hermitian_pd(factorization):
    check_solve(symmetric_pd(complex_=True, hermitian=True), factorization, 'hermitian')


@pytest.mark.parametrize('factorization', ['ldlt', 'ldltsbk', 'ldlttpp'])
def test_complex_hermitian_indefinite(factorization):
    check_solve(symmetric_indefinite(complex_=True, hermitian=True), factorization, 'hermitian')


@pytest.mark.parametrize('factorization', LDLT_FACTORS + LU_FACTORS)
def test_complex_symmetric(factorization):
    """Complex symmetric (A = A^T, not hermitian) - the typical FEM case."""
    A = symmetric_indefinite(complex_=True)
    if factorization in LDLT_FACTORS and not COMPLEX_SYMMETRIC_AVAILABLE:
        with pytest.warns(RuntimeWarning, match='falling back to LU'):
            check_solve(A, factorization, 'symmetric')
    else:
        check_solve(A, factorization, 'symmetric')


def test_complex_symmetric_fallback_before_macos26(monkeypatch):
    """Simulate macOS < 26, where Accelerate would trap on complex symmetric LDLT."""
    import emerge_aasds._interface as backend
    monkeypatch.setattr(backend, 'COMPLEX_SYMMETRIC_AVAILABLE', False)
    with pytest.warns(RuntimeWarning, match='falling back to LU'):
        check_solve(symmetric_indefinite(complex_=True), 'ldltsbk', 'symmetric')


def test_cholesky_rejects_indefinite():
    A = symmetric_indefinite()
    solver = AccelerateInterface(factorization='cholesky', symmetry='symmetric')
    solver.analyse(A)
    with pytest.raises(RuntimeError, match='factorization failed'):
        solver.factorize(A)
    # Solver is still usable after a failed factorization
    solver.factorize(symmetric_pd())


def test_symmetric_factorization_requires_symmetric_matrix():
    with pytest.raises(ValueError, match='requires a symmetric or hermitian'):
        AccelerateInterface(factorization='cholesky', symmetry='nonsymmetric')


def test_default_symmetric_uses_upper_triangle_correctly():
    """Only the lower triangle is passed; the result must match the full matrix."""
    A = symmetric_indefinite(complex_=True, hermitian=True)
    check_solve(A, 'lu', 'hermitian')  # default override -> LDLT_SBK
