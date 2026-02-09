/* accelerate_wrapper.c */
/* Simple C wrapper to expose Accelerate Sparse functions to Python */

#include <Accelerate/Accelerate.h>
#include <stdlib.h>

/* Wrapper for SparseConvertFromCoordinate - Double */
/* Returns pointer to heap-allocated matrix */
SparseMatrix_Double* accel_convert_from_coordinate_double(
    int rowCount, int columnCount, long blockCount, uint8_t blockSize,
    SparseAttributes_t attributes, const int *row, const int *column, 
    const double *data)
{
    SparseMatrix_Double *mat = malloc(sizeof(SparseMatrix_Double));
    *mat = SparseConvertFromCoordinate(rowCount, columnCount, blockCount, 
                                       blockSize, attributes, row, column, data);
    return mat;
}

/* Wrapper for SparseConvertFromCoordinate - Complex Double */
SparseMatrix_Complex_Double* accel_convert_from_coordinate_complex_double(
    int rowCount, int columnCount, long blockCount, uint8_t blockSize,
    SparseAttributesComplex_t attributes, const int *row, const int *column,
    const void *data)
{
    SparseMatrix_Complex_Double *mat = malloc(sizeof(SparseMatrix_Complex_Double));
    *mat = SparseConvertFromCoordinate(rowCount, columnCount, blockCount,
                                       blockSize, attributes, row, column,
                                       (const _Complex double *)data);
    return mat;
}

/* Wrapper for SparseFactor - Double */
SparseOpaqueFactorization_Double* accel_factor_double(
    SparseFactorization_t type, SparseMatrix_Double *Matrix)
{
    SparseOpaqueFactorization_Double *fact = malloc(sizeof(SparseOpaqueFactorization_Double));
    *fact = SparseFactor(type, *Matrix);
    return fact;
}

/* Wrapper for SparseFactor - Complex Double */
SparseOpaqueFactorization_Complex_Double* accel_factor_complex_double(
    SparseFactorization_t type, SparseMatrix_Complex_Double *Matrix)
{
    SparseOpaqueFactorization_Complex_Double *fact = malloc(sizeof(SparseOpaqueFactorization_Complex_Double));
    *fact = SparseFactor(type, *Matrix);
    return fact;
}

/* Wrapper for SparseSolve - Double */
void accel_solve_double(
    SparseOpaqueFactorization_Double *Factored,
    DenseMatrix_Double B, DenseMatrix_Double X)
{
    SparseSolve(*Factored, B, X);
}

/* Wrapper for SparseSolve - Complex Double */
void accel_solve_complex_double(
    SparseOpaqueFactorization_Complex_Double *Factored,
    DenseMatrix_Complex_Double B, DenseMatrix_Complex_Double X)
{
    SparseSolve(*Factored, B, X);
}

/* Wrapper for SparseRefactor - Double */
void accel_refactor_double(
    SparseMatrix_Double *Matrix,
    SparseOpaqueFactorization_Double *Factorization)
{
    SparseRefactor(*Matrix, Factorization);
}

/* Wrapper for SparseRefactor - Complex Double */
void accel_refactor_complex_double(
    SparseMatrix_Complex_Double *Matrix,
    SparseOpaqueFactorization_Complex_Double *Factorization)
{
    SparseRefactor(*Matrix, Factorization);
}

/* Wrapper for SparseCleanup - Matrix Double */
void accel_cleanup_matrix_double(SparseMatrix_Double *Matrix)
{
    if (Matrix) {
        SparseCleanup(*Matrix);
        free(Matrix);
    }
}

/* Wrapper for SparseCleanup - Matrix Complex Double */
void accel_cleanup_matrix_complex_double(SparseMatrix_Complex_Double *Matrix)
{
    if (Matrix) {
        SparseCleanup(*Matrix);
        free(Matrix);
    }
}

/* Wrapper for SparseCleanup - Factorization Double */
void accel_cleanup_factorization_double(SparseOpaqueFactorization_Double *Factored)
{
    if (Factored) {
        SparseCleanup(*Factored);
        free(Factored);
    }
}

/* Wrapper for SparseCleanup - Factorization Complex Double */
void accel_cleanup_factorization_complex_double(SparseOpaqueFactorization_Complex_Double *Factored)
{
    if (Factored) {
        SparseCleanup(*Factored);
        free(Factored);
    }
}