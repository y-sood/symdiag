#pragma once
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

#include "jacobi_config.hpp"
#include "partition.hpp"

typedef struct {
    double diag_norm_sq;
    double offdiag_norm_sq;
    double frob_norm_sq;
    double ratio;
    double rel_offdiag;
    double trace;
    double max_offdiag_abs;
    double avg_diag_abs;
    double avg_offdiag_abs;
    double orthogonality_error;
    double reconstruction_error_abs;
    double reconstruction_error_rel;
    double reconstruction_max_abs;
    double reconstruction_max_rel;
} DiagonalizationReport;

void print_dense_matrix_matlab(const char* label, FLA_Obj A);
void dump_canonical_elements(const char* label, FLA_Obj T, dim_t n, dim_t order);
void print_initial_state(const JacobiConfig* config, FLA_Obj T, FLA_Obj F, JacobiPartition* vpartition, dim_t n, dim_t order);
void print_final_state(const JacobiConfig* config, FLA_Obj T, FLA_Obj F, dim_t n, dim_t order);
void print_rotation_details(int iter, int group, int pair_idx, int p, int q, double c, double s, FLA_Obj T, dim_t order);

