# emerge-aasds

**Apple Accelerate Sparse Direct Solver** - Written for MacOS to be used as solver for EMerge FEM Library.

The majority of this interface was written by Claude code. Mostly the C-interface, installer, build stripts and C-files. 
I will be doing refactors in the upcoming period. 

## Installation

**macOS only** (requires Xcode Command Line Tools):
There is currently no PyPI interface yet because I want to make sure there are no license restrictions I should be abiding by.

```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install emerge-aasds
pip install git+https://github.com/FennisRobert/emerge-aasds.git
```
---
## Shortcomings

The current C wrapper does not correctly support Cholesky and LDLT factorization.

---

## API

### `AccelerateInterface(factorization=Factorization.LU, symmetry=Symmetry.NONSYMMETRIC, verbose=False)`

High-level interface to the Accelerate sparse direct solver backend.

This class separates **symbolic analysis**, **numeric factorization**, and **solution**, allowing efficient reuse when solving multiple systems with identical sparsity patterns.

---

### Parameters

* **`factorization`** : `Factorization | str`, optional
  Matrix factorization method.

  Supported values:

  * `'lu'` *(default, recommended for general matrices)*
  * `'qr'`
  * `'cholesky'` *(requires symmetric positive definite matrices)*
  * `'ldlt'` *(requires symmetric or Hermitian matrices)*

* **`symmetry`** : `Symmetry | str`, optional
  Structural symmetry of the matrix.

  Supported values:

  * `'nonsymmetric'` *(default)*
  * `'symmetric'`
  * `'hermitian'`

  This is used to select optimized backend paths and **must match the actual matrix structure**.

* **`verbose`** : `bool`, optional
  Enable backend timing and diagnostic output.

  * `False` → silent
  * `True` → print timing and solver diagnostics

---

## Methods

### `analyse(A)`

Perform **symbolic factorization** of the matrix.

This step analyzes the **sparsity pattern only** and allocates internal data structures.
It must be called **once per sparsity pattern** and may be reused across multiple numeric factorizations.

#### Parameters

* **`A`** : sparse matrix
  System matrix. Must have the same sparsity pattern for reuse.

#### Notes

* Expensive relative to `factorize`
* Must be re-run if the sparsity pattern changes
* Does **not** depend on numerical values

---

### `factorize(A)`

Perform **numeric factorization** using the previously analyzed structure.

This step computes the actual matrix factors and may be repeated efficiently for new numerical values with the same sparsity pattern.

#### Parameters

* **`A`** : sparse matrix
  System matrix with identical sparsity pattern to the one passed to `analyse`.

#### Notes

* Requires a prior call to `analyse`
* Much faster than symbolic analysis
* Overwrites any existing numeric factorization

---

### `solve(b)`

Solve the linear system using the most recent factorization.

#### Parameters

* **`b`** : array-like
  Right-hand side vector or matrix.

#### Returns

* **`x`** : array
  Solution vector or matrix.
* **`info`** : dict
  Solver information and diagnostics. Typical entries includez:

  * `success` : bool
  * `iterations` : int (if applicable)
  * `residual_norm` : float
  * `message` : str

#### Notes

* Requires a prior successful `factorize`
* Supports multiple RHS vectors (backend-dependent)

---

### `destroy()`

Release all internal solver resources.

This method frees memory allocated by the backend and invalidates the solver instance.

#### Notes

* Called automatically on object destruction
* Safe to call multiple times
* Recommended when managing many solver instances manually

---

## Typical Usage Pattern

```python
from emerge_aasds import AccelerateInterface, Factorization, Symmetry
solver = AccelerateInterface(
    factorization=Factorization.LU,
    symmetry=Symmetry.NONSYMMETRIC,
    verbose=True,
)

solver.analyse(A)        # once per sparsity pattern
solver.factorize(A)     # per numeric update
x, info = solver.solve(b)

solver.destroy()        # optional, automatic otherwise
```

---

## Performance Notes

* **Reuse `analyse` whenever possible**
* Changing matrix values → call `factorize`
* Changing sparsity pattern → must call `analyse` again
* Incorrect symmetry declarations may lead to incorrect results or backend errors

---

## Error Handling

* Invalid factorization or symmetry values raise `ValueError`
* Backend failures are reported via the returned `info` dictionary
* Structural mismatches may raise runtime errors from the backend

---

## Requirements

- macOS 10.13+ (High Sierra or later)
- Python 3.10+
- NumPy 1.20+
- SciPy 1.7+
- Xcode Command Line Tools

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue or PR.
