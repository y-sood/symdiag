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
void reconstruct_m(FLA_Obj Ddiag, FLA_Obj Udense, FLA_Obj Trec, dim_t n, dim_t order);
// Compares T_final's diagonal times U_final (reconstruction) against T_initial without
// materializing the reconstructed tensor or a diagonal tensor: walks canonical entries only,
// weighting each by its number of distinct permutations.
void check_diagonalization_with_reconstruction(FLA_Obj T_final, FLA_Obj U_final, FLA_Obj T_initial, dim_t n, dim_t order, double tolerance);

// Diagnostic: reconstruct T_init = T_final x1 F x2 F ... xd F using ALL entries of T_final
// (the exact identity when F is orthogonal), and compare with T_initial over every entry.
// Uses two dense n^order scratch arrays; returns false (and prints why) if that is too big.
bool full_reconstruction_report(FLA_Obj T_final, FLA_Obj F, FLA_Obj T_initial, dim_t n, dim_t order);
