/* accelerate_wrapper.h */
#ifndef ACCELERATE_WRAPPER_H
#define ACCELERATE_WRAPPER_H

#include <Accelerate/Accelerate.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * EXISTING API (unchanged for backward compatibility)
 * ============================================================================ */

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


/* ============================================================================
 * EXTENDED API - WITH TUNABLE PARAMETERS
 * ============================================================================ */

/* Ordering methods */
typedef enum {
    ACCEL_ORDER_DEFAULT = 0,
    ACCEL_ORDER_AMD = 1,
    ACCEL_ORDER_METIS = 2,
    ACCEL_ORDER_COLAMD = 3,
    ACCEL_ORDER_MTMETIS = 4
} AccelOrderMethod;

/* Scaling methods */
typedef enum {
    ACCEL_SCALING_DEFAULT = 0,
    ACCEL_SCALING_NONE = 1
} AccelScalingMethod;

/* Factorization options */
typedef struct {
    AccelOrderMethod order_method;      /* Ordering algorithm (-1 = default) */
    AccelScalingMethod scaling_method;  /* Scaling strategy (-1 = default) */
    double pivot_tolerance;              /* Pivot threshold (-1 = auto) */
    double zero_tolerance;               /* Zero threshold (-1 = auto) */
} AccelFactorOptions;

/* Get default options (safe defaults) */
AccelFactorOptions accel_get_default_options(int is_complex);

/* Factorization WITH options - returns pointers */
SparseOpaqueFactorization_Double* accel_factor_double_ex(
    SparseFactorization_t type,
    SparseMatrix_Double *Matrix,
    const AccelFactorOptions *options  /* Pass NULL for defaults */
);

SparseOpaqueFactorization_Complex_Double* accel_factor_complex_double_ex(
    SparseFactorization_t type,
    SparseMatrix_Complex_Double *Matrix,
    const AccelFactorOptions *options  /* Pass NULL for defaults */
);

/* Refactorization WITH options */
void accel_refactor_double_ex(
    SparseMatrix_Double *Matrix,
    SparseOpaqueFactorization_Double *Factorization,
    const AccelFactorOptions *options  /* Pass NULL for defaults */
);

void accel_refactor_complex_double_ex(
    SparseMatrix_Complex_Double *Matrix,
    SparseOpaqueFactorization_Complex_Double *Factorization,
    const AccelFactorOptions *options  /* Pass NULL for defaults */
);

#ifdef __cplusplus
}
#endif

#endif /* ACCELERATE_WRAPPER_H */