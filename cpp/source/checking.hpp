#pragma once 
//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

double compute_tensor_difference(FLA_Obj A, FLA_Obj B);
double orthogonality_error_matrix(FLA_Obj U, dim_t n);
double tensor_trace_general(FLA_Obj T, dim_t n, dim_t order);
double offdiag_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order);
double diag_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order);
double frob_norm_sq_tensor(FLA_Obj T, dim_t n, dim_t order);
void compute_tensor_norms(FLA_Obj T, dim_t n, dim_t order, double* diag_norm, double* offdiag_norm);
void check_diagonalization(FLA_Obj T, dim_t n, dim_t order, double tolerance);
double max_abs_tensor(FLA_Obj T, dim_t n, dim_t order);
void extract_diagonal_tensor(FLA_Obj T, FLA_Obj D, dim_t n, dim_t order);
void tensor_diff_metrics( FLA_Obj A, FLA_Obj B, dim_t n, dim_t order, double* err_abs, double* err_rel, double* max_abs, double* max_rel);
void reconstruct_m(FLA_Obj Ddiag, FLA_Obj Udense, FLA_Obj Trec, dim_t n);
void check_diagonalization_with_reconstruction(FLA_Obj T_final, FLA_Obj U_final, FLA_Obj T_initial, FLA_Obj D_diag, FLA_Obj T_reconstructed, dim_t n, dim_t order, double tolerance);