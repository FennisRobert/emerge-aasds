# emerge-aasds

**Apple Accelerate Sparse Direct Solver** - Lightning-fast sparse linear system solver for macOS.

## Installation

**macOS only** (requires Xcode Command Line Tools):
```bash
# Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# Install emerge-aasds
pip install emerge-aasds
```

## API

### `AccelerateInterface(factorization='lu', symmetry='nonsymmetric', verbose=0)`

**Parameters:**
- `factorization`: `'lu'` (recommended), `'qr'`, `'cholesky'`, `'ldlt'`
- `symmetry`: `'nonsymmetric'`, `'symmetric'`, `'hermitian'`
- `verbose`: `0` (silent) or `1` (timing info)

**Methods:**
- `analyse(A)` - Symbolic factorization
- `factorize(A)` - Numeric factorization (reuse for same sparsity pattern)
- `solve(b)` - Solve system, returns `(x, info)`
- `destroy()` - Cleanup (called automatically)

## Requirements

- macOS 10.13+ (High Sierra or later)
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- Xcode Command Line Tools

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue or PR.