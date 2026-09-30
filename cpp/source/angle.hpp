#pragma once
//Flame definition
#include "FLAME.h"
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

int solve_cubic(double a, double b, double c, double d, double* roots);
double eval_obj(double c, double s, double App, double Aqq, double Apq, double Aqp);
void calculate_rotation_angle(const FLA_Obj T, const double* A_arr, dim_t order, double* c_out, double* s_out);
// Order-d angle solve (d >= 3). A[k] (k=0..d) is the pivot sub-tensor entry with k copies
// of q and d-k copies of p. Maximizes A_0'(theta)^2 + A_d'(theta)^2 exactly: the objective
// is a trig polynomial in phi=4*theta with floor(d/2) harmonics (see higher_order.md),
// whose coefficients are built from exact harmonic tables, then maximized by a grid search
// over phi followed by bisection on the derivative for each local maximum.
void calculate_rotation_angle_general(const double* A, dim_t order, double* c_out, double* s_out);
void embed_givens_rotation(FLA_Obj G, int p, int q, double c, double s);
// Applies one pivot pair's rotation directly to columns p and q of a dense factor
// matrix F: equivalent to F <- F * G^T for a G that is identity except at
// (p,p)=c, (p,q)=s, (q,p)=-s, (q,q)=c, but O(n) instead of the O(n^3) dense Gemm.
void apply_givens_rotation_to_columns(FLA_Obj F, int p, int q, double c, double s);