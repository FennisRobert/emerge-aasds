/* accelerate_wrapper.c */
/* C wrapper to expose Accelerate Sparse functions to Python */
/* EXTENDED VERSION with tunable parameters + direct CSC construction */

#include "_wrapper.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * EXISTING API (unchanged for backward compatibility)
 * ============================================================================ */

/* Wrapper for SparseConvertFromCoordinate - Double */
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

/* Wrapper for SparseCleanup - Matrix Double (COO-created, Accelerate owns internals) */
void accel_cleanup_matrix_double(SparseMatrix_Double *Matrix)
{
    if (Matrix) {
        SparseCleanup(*Matrix);
        free(Matrix);
    }
}

/* Wrapper for SparseCleanup - Matrix Complex Double (COO-created) */
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


/* ============================================================================
 * EXTENDED API - WITH TUNABLE PARAMETERS
 * ============================================================================ */

/* Get default factor options */
AccelFactorOptions accel_get_default_options(int is_complex)
{
    AccelFactorOptions opts;
    opts.order_method = ACCEL_ORDER_DEFAULT;
    opts.scaling_method = ACCEL_SCALING_DEFAULT;
    opts.pivot_tolerance = is_complex ? 0.01 : 0.01;  /* Accelerate defaults */
    opts.zero_tolerance = -1.0;  /* Auto */
    return opts;
}

/* Helper to convert our ordering enum to Accelerate's */
static SparseOrder_t get_order_type(AccelOrderMethod method)
{
    switch (method) {
        case ACCEL_ORDER_AMD:
            return SparseOrderAMD;
        case ACCEL_ORDER_METIS:
            return SparseOrderMetis;
        case ACCEL_ORDER_COLAMD:
            return SparseOrderCOLAMD;
        case ACCEL_ORDER_MTMETIS:
            return SparseOrderMTMetis;
        case ACCEL_ORDER_DEFAULT:
        default:
            return SparseOrderDefault;
    }
}

/* Helper to build symbolic options */
static SparseSymbolicFactorOptions build_symbolic_options(const AccelFactorOptions *opts)
{
    SparseSymbolicFactorOptions sym_opts;
    
    if (opts == NULL) {
        sym_opts = _SparseDefaultSymbolicFactorOptions;
    } else {
        sym_opts = _SparseDefaultSymbolicFactorOptions;
        sym_opts.orderMethod = get_order_type(opts->order_method);
    }
    
    return sym_opts;
}

/* Helper to build numeric options */
static SparseNumericFactorOptions build_numeric_options(const AccelFactorOptions *opts, int is_complex)
{
    SparseNumericFactorOptions num_opts;
    
    if (opts == NULL) {
        num_opts = is_complex ? _SparseDefaultNumericFactorOptions_Complex_Double 
                             : _SparseDefaultNumericFactorOptions_Double;
    } else {
        num_opts = is_complex ? _SparseDefaultNumericFactorOptions_Complex_Double 
                             : _SparseDefaultNumericFactorOptions_Double;
        
        if (opts->scaling_method == ACCEL_SCALING_NONE) {
            num_opts.scalingMethod = SparseScalingUser;
            num_opts.scaling = NULL;
        } else if (opts->scaling_method == ACCEL_SCALING_DEFAULT) {
            num_opts.scalingMethod = SparseScalingDefault;
        }
        
        if (opts->pivot_tolerance >= 0.0) {
            num_opts.pivotTolerance = opts->pivot_tolerance;
        }
        
        if (opts->zero_tolerance >= 0.0) {
            num_opts.zeroTolerance = opts->zero_tolerance;
        }
    }
    
    return num_opts;
}

/* Extended factor - Double */
SparseOpaqueFactorization_Double* accel_factor_double_ex(
    SparseFactorization_t type,
    SparseMatrix_Double *Matrix,
    const AccelFactorOptions *options)
{
    SparseSymbolicFactorOptions sym_opts = build_symbolic_options(options);
    SparseNumericFactorOptions num_opts = build_numeric_options(options, 0);
    
    SparseOpaqueFactorization_Double *fact = malloc(sizeof(SparseOpaqueFactorization_Double));
    *fact = SparseFactor(type, *Matrix, sym_opts, num_opts);
    
    return fact;
}

/* Extended factor - Complex Double */
SparseOpaqueFactorization_Complex_Double* accel_factor_complex_double_ex(
    SparseFactorization_t type,
    SparseMatrix_Complex_Double *Matrix,
    const AccelFactorOptions *options)
{
    SparseSymbolicFactorOptions sym_opts = build_symbolic_options(options);
    SparseNumericFactorOptions num_opts = build_numeric_options(options, 1);
    
    SparseOpaqueFactorization_Complex_Double *fact = malloc(sizeof(SparseOpaqueFactorization_Complex_Double));
    *fact = SparseFactor(type, *Matrix, sym_opts, num_opts);
    
    return fact;
}

/* Extended refactor - Double */
void accel_refactor_double_ex(
    SparseMatrix_Double *Matrix,
    SparseOpaqueFactorization_Double *Factorization,
    const AccelFactorOptions *options)
{
    SparseNumericFactorOptions num_opts = build_numeric_options(options, 0);
    SparseRefactor(*Matrix, Factorization, num_opts);
}

/* Extended refactor - Complex Double */
void accel_refactor_complex_double_ex(
    SparseMatrix_Complex_Double *Matrix,
    SparseOpaqueFactorization_Complex_Double *Factorization,
    const AccelFactorOptions *options)
{
    SparseNumericFactorOptions num_opts = build_numeric_options(options, 1);
    SparseRefactor(*Matrix, Factorization, num_opts);
}


/* ============================================================================
 * CSC API - Direct construction from Compressed Sparse Column arrays
 *
 * Accelerate's internal representation IS CSC:
 *   SparseMatrixStructure { rowCount, columnCount, columnStarts, rowIndices,
 *                           attributes, blockSize }
 *   SparseMatrix_Double   { structure, data }
 *
 * We construct the struct directly, pointing at the caller's arrays.
 * The caller (Python) must keep those arrays alive.
 * ============================================================================ */

/* Direct CSC construction - Double */
SparseMatrix_Double* accel_create_from_csc_double(
    int rowCount, int columnCount,
    const long *columnStarts, const int *rowIndices,
    const double *data,
    int kind, int triangle)
{
    SparseMatrix_Double *mat = malloc(sizeof(SparseMatrix_Double));
    
    SparseAttributes_t attrs = {0};
    
    /* Set kind bits: SparseOrdinary=0, SparseSymmetric=2 (bits 0-1) */
    /* Accelerate uses: 0=ordinary, 2=symmetric, 3=hermitian in the kind field */
    if (kind == 3) {  /* symmetric */
        attrs._reserved = 0;
        attrs.kind = SparseSymmetric;
    } else if (kind == 4) {  /* hermitian */
        attrs._reserved = 0;
        attrs.kind = SparseSymmetric;  /* For real, symmetric == hermitian */
    }
    /* kind == 0: ordinary, attrs stays zeroed */
    
    /* Set triangle: SparseLowerTriangle if lower */
    if (kind == 3 || kind == 4) {
        if (triangle == 1) {
            attrs.triangle = SparseLowerTriangle;
        } else {
            attrs.triangle = SparseUpperTriangle;
        }
    }
    
    SparseMatrixStructure structure = {
        .rowCount = rowCount,
        .columnCount = columnCount,
        .columnStarts = (long *)columnStarts,
        .rowIndices = (int *)rowIndices,
        .attributes = attrs,
        .blockSize = 1
    };
    
    mat->structure = structure;
    mat->data = (double *)data;
    
    return mat;
}

/* Direct CSC construction - Complex Double
 *
 * SparseMatrix_Complex_Double uses SparseMatrixStructureComplex internally,
 * which differs from SparseMatrixStructure only in that its attributes field
 * is typed as SparseAttributesComplex_t instead of SparseAttributes_t.
 * We populate the struct fields through the mat-> pointer to get the
 * correct types. */
SparseMatrix_Complex_Double* accel_create_from_csc_complex_double(
    int rowCount, int columnCount,
    const long *columnStarts, const int *rowIndices,
    const void *data,
    int kind, int triangle)
{
    SparseMatrix_Complex_Double *mat = malloc(sizeof(SparseMatrix_Complex_Double));
    
    /* Zero-initialise the whole struct so all reserved/padding bits are clean */
    memset(mat, 0, sizeof(SparseMatrix_Complex_Double));
    
    /* Populate the structure fields directly through the complex type */
    mat->structure.rowCount = rowCount;
    mat->structure.columnCount = columnCount;
    mat->structure.columnStarts = (long *)columnStarts;
    mat->structure.rowIndices = (int *)rowIndices;
    mat->structure.blockSize = 1;
    
    /* Set attributes on the complex-typed attributes field */
    if (kind == 3) {  /* symmetric */
        mat->structure.attributes.kind = SparseSymmetric;
    } else if (kind == 4) {  /* hermitian */
        mat->structure.attributes.kind = SparseSymmetric;
    }
    /* kind == 0: ordinary, attributes stay zeroed */
    
    if (kind == 3 || kind == 4) {
        if (triangle == 1) {
            mat->structure.attributes.triangle = SparseLowerTriangle;
        } else {
            mat->structure.attributes.triangle = SparseUpperTriangle;
        }
    }
    
    mat->data = (_Complex double *)data;
    
    return mat;
}

/* Cleanup for CSC-created matrices.
 * Only free the wrapper struct — the arrays are owned by Python. */
void accel_cleanup_csc_matrix_double(SparseMatrix_Double *Matrix)
{
    if (Matrix) {
        /* Do NOT call SparseCleanup — we don't own the arrays */
        free(Matrix);
    }
}

void accel_cleanup_csc_matrix_complex_double(SparseMatrix_Complex_Double *Matrix)
{
    if (Matrix) {
        free(Matrix);
    }
}