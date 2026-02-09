/* accelerate_wrapper.h */
#ifndef ACCELERATE_WRAPPER_H
#define ACCELERATE_WRAPPER_H

#include <Accelerate/Accelerate.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matrix creation - returns pointers */
SparseMatrix_Double* accel_convert_from_coordinate_double(
    int rowCount, int columnCount, long blockCount, uint8_t blockSize,
    SparseAttributes_t attributes, const int *row, const int *column,
    const double *data);

SparseMatrix_Complex_Double* accel_convert_from_coordinate_complex_double(
    int rowCount, int columnCount, long blockCount, uint8_t blockSize,
    SparseAttributesComplex_t attributes, const int *row, const int *column,
    const void *data);

/* Factorization - returns pointers */
SparseOpaqueFactorization_Double* accel_factor_double(
    SparseFactorization_t type, SparseMatrix_Double *Matrix);

SparseOpaqueFactorization_Complex_Double* accel_factor_complex_double(
    SparseFactorization_t type, SparseMatrix_Complex_Double *Matrix);

/* Refactorization - updates existing factorization */
void accel_refactor_double(
    SparseMatrix_Double *Matrix,
    SparseOpaqueFactorization_Double *Factorization);

void accel_refactor_complex_double(
    SparseMatrix_Complex_Double *Matrix,
    SparseOpaqueFactorization_Complex_Double *Factorization);

/* Solve - takes pointers */
void accel_solve_double(
    SparseOpaqueFactorization_Double *Factored,
    DenseMatrix_Double B, DenseMatrix_Double X);

void accel_solve_complex_double(
    SparseOpaqueFactorization_Complex_Double *Factored,
    DenseMatrix_Complex_Double B, DenseMatrix_Complex_Double X);

/* Cleanup - takes pointers */
void accel_cleanup_matrix_double(SparseMatrix_Double *Matrix);
void accel_cleanup_matrix_complex_double(SparseMatrix_Complex_Double *Matrix);
void accel_cleanup_factorization_double(SparseOpaqueFactorization_Double *Factored);
void accel_cleanup_factorization_complex_double(SparseOpaqueFactorization_Complex_Double *Factored);

#ifdef __cplusplus
}
#endif

#endif /* ACCELERATE_WRAPPER_H */