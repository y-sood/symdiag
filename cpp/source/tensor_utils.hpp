#pragma once
#include <vector>
//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef abs
#undef abs
#endif

void set_dense_matrix_element(FLA_Obj A, dim_t row, dim_t col, double value);
double get_dense_matrix_element(FLA_Obj A, dim_t i, dim_t j);
void set_tensor_element_bccs(FLA_Obj T, dim_t* index, dim_t order, double value);
void set_matrix_element_bccs(FLA_Obj G, dim_t row, dim_t col, double value);
double get_tensor_element_bccs(FLA_Obj T, dim_t i, dim_t j, dim_t k);
double get_tensor_element_bccs_alt(FLA_Obj T, dim_t* index, dim_t order);
double get_matrix_element(FLA_Obj A, dim_t i, dim_t j);
void gram_schmidt(const std::vector<double>& M, std::vector<double>& U, int n);
int is_superdiagonal(const dim_t* idx, dim_t order);
void initDiagonalizableTensor(dim_t order, dim_t size[], dim_t b, FLA_Obj* obj, int n, unsigned int seed);
void initIdentityMatrix(dim_t n, dim_t bC, dim_t bA, FLA_Obj* B);
void initSymmTensor(dim_t order, dim_t size[], dim_t b, FLA_Obj* obj);
void initZeroMatrix(dim_t n, dim_t bC, dim_t bA, FLA_Obj* B);
void initIdentityDenseMatrix(dim_t n, FLA_Obj* B);
void initZeroDenseMatrix(dim_t n, FLA_Obj* B);
void setIdentityMatrix(dim_t n, FLA_Obj* G_sttsm);
void setIdentityDenseMatrix(dim_t n, FLA_Obj* G_fac);
void fill_intra_block_symmetry(FLA_Obj T, dim_t order, dim_t block_size);
void cleanup_tensor(FLA_Obj* T);
void copy_tensor_values(FLA_Obj src, FLA_Obj dst);
void copy_matrix_values(FLA_Obj src, FLA_Obj dst);
void cleanup_matrix(FLA_Obj* F);