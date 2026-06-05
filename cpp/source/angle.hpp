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
void calculate_rotation_angle(FLA_Obj T, double* A_arr, dim_t order, double* c_out, double* s_out);
void embed_givens_rotation(FLA_Obj G, int p, int q, double c, double s);
void embed_givens_rotation_dense(FLA_Obj G, int p, int q, double c, double s);